from collections import Counter
from copy import deepcopy

import numpy as np
import torch
from TVSD import VolumeSlicingDataset, ExpandedPaddedSegmentation
from joblib import Parallel, delayed
from medpy.io import load as medload
from sklearn.model_selection import train_test_split
from torch.nn.functional import interpolate
from torch.utils.data import ConcatDataset, Subset
from torch.utils.data import Dataset


def gather_paths(path):
    volumes, targets = [], []
    with open(path) as f:
        data_addresses = map(lambda x: x.strip('\n').split('     '), f.readlines())
    for (v, t) in data_addresses:
        volumes.append(v)
        targets.append(t)
    return volumes, targets


def convert_target(addr, converter):
    if isinstance(list(converter.keys())[0], str):
        # this is because of the restrictions in the OmegaConf. Should be resolved with 2.1 version.
        converter = {int(k): v for k,v in converter.items()}
    volume = medload(addr)[0]
    volume = np.vectorize(converter.get)(volume)
    return volume


def sklearn_train_test_split(gathered_data, random_state=None, train_volumes=None, volumes_limit=None):
    volumes_limit = volumes_limit or len(gathered_data[0])
    train_data, test_data = train_test_split(list(zip(*gathered_data))[:volumes_limit], random_state=random_state,
                                             train_size=train_volumes)
    return train_data, test_data


def get_TVSD_datasets(data_addresses, aug=None, label_converter=None, n_procs=1, **kwargs):

    def _process_onevolume(image_addr, label_addr, aug, label_converter, **kwargs):
        if label_converter is not None:
            label = convert_target(label_addr, label_converter)
        else:
            label = ExpandedPaddedSegmentation(label_addr)

        one_volume = VolumeSlicingDataset(image_addr, segmentation=label, augmentations=aug, **kwargs)
        one_volume.task = int(name[0]) if (name := image_addr.split('/')[-3]) != 'origin' else int(image_addr.split('/')[-4][0])
        return one_volume

    datasets = Parallel(n_jobs=n_procs, verbose=5)(delayed(_process_onevolume)
                                                     (image_addr, label_addr, aug, label_converter, **kwargs)
                                                     for image_addr, label_addr in data_addresses)
    return ConcatDataset(datasets)


def adaptive_choice(choose_from, choice_count):
    if choice_count <= len(choose_from):
        return np.random.choice(choose_from, choice_count, replace=False)
    else:
        subsample = [choose_from] * (choice_count // len(choose_from))  # all the full inclusions first
        subsample.append(
            np.random.choice(choose_from, choice_count % len(choose_from), replace=False))  # additional records
        return np.concatenate(subsample)


def TVSD_dataset_resample(dataset, segmented_part=1.0, empty_part=0.1):
    is_marked = np.concatenate([d.segmentation._contains_markup() for d in dataset.datasets])

    if segmented_part is None:
        segmented_part = 1.0
    if isinstance(segmented_part, float):
        segmented_part = int(is_marked.sum() * segmented_part)

    if isinstance(empty_part, float):
        empty_part = int(segmented_part * empty_part)
    elif empty_part is None:
        empty_part = (1 - is_marked).sum()

    segmented_subsample = adaptive_choice(np.where(is_marked)[0], segmented_part)
    empty_subsample = adaptive_choice(np.where(1 - is_marked)[0], empty_part)

    return Subset(dataset, np.concatenate([segmented_subsample, empty_subsample]))


def TaskWrapper_balanced_resample(dataset, upsample=5):
    tasks = dataset.tasks
    task_cnt = Counter(tasks)
    weights = np.array([1 / task_cnt[task] for task in tasks])
    weights /= weights.sum()
    n = len(dataset)
    idx = np.random.choice(n, n * upsample, replace=True, p=weights)
    return Subset(dataset, idx)


class TVSDTaskWrapper(Dataset):
    def __init__(self, base_dataset, transform=lambda x: x, downsample_lbl=False, scale_factor_lbl=0.125):
        super().__init__()
        self.base_dataset = base_dataset
        self.transform = transform
        self.downsample_lbl = downsample_lbl
        self.tasks = []

        for dataset in self.base_dataset.dataset.datasets:
            self.tasks.extend([dataset.task] * len(dataset))
        self.tasks = np.array(self.tasks)[self.base_dataset.indices]

        if isinstance(scale_factor_lbl, float):
            self.scale_factor_lbl = (scale_factor_lbl, scale_factor_lbl)
        else:
            self.scale_factor_lbl = scale_factor_lbl

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, mask = self.base_dataset.__getitem__(index)
        task = self.tasks[index]

        sample = self.transform(deepcopy({'image': image, 'mask': mask}))

        if self.downsample_lbl:  # Downsample
            sample['mask'] = interpolate(sample['mask'][None, None, :, :].float(),
                                            scale_factor=self.scale_factor_lbl, mode='nearest').squeeze().long()

        sample['mask'] = self.id2trainId(sample['mask'], task)
        return sample['image'], sample['mask'], task

    @staticmethod
    def id2trainId(label, task_id):
        if task_id <= 3:
            organ = (label == 1)
            tumor = (label == 2)
        elif task_id in [4, 5]:
            organ = np.zeros_like(label)
            tumor = (label == 1)
        elif task_id == 6:
            organ = (label == 1)
            tumor = np.zeros_like(label)
        else:
            print("Error, No such task!")
            return None

        shape = label.shape
        results_map = np.zeros((2, shape[1], shape[2])).astype(np.float32)

        results_map[0, :, :] = np.where(organ, 1, 0)
        results_map[1, :, :] = np.where(tumor, 1, 0)
        return results_map


class EmbeddingDataset(Dataset):
    def __init__(self, data_paths, id):
        super().__init__()
        self.paths = data_paths
        self.task_id = 2 * id if 'organ' in data_paths[0] else 2 * id + 1

    def __getitem__(self, item):
        path = self.paths[item]
        with open(path, 'rb') as f:
            img, lbl = torch.load(f)
        return img, lbl, self.task_id

    def __len__(self):
        return len(self.paths)


def TVSD_rebalance(dataset, upsample=2):
    tvsd_resampled_empty = TVSD_dataset_resample(dataset)
    resampled_tasks = TVSDTaskWrapper(tvsd_resampled_empty)
    balanced_tasks = TaskWrapper_balanced_resample(resampled_tasks, upsample=upsample)
    return balanced_tasks


def TVSD_rebalance_neg(dataset):
    tvsd_resampled_empty = TVSD_dataset_resample(dataset)
    resampled_tasks = TVSDTaskWrapper(tvsd_resampled_empty)
    return resampled_tasks


def TVSD_no_rebalance(dataset):
    resampled_tasks = TVSDTaskWrapper(Subset(dataset, torch.arange(len(dataset))))
    return resampled_tasks
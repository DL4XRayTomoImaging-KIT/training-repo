from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from TVSD import VolumeSlicingDataset, ExpandedPaddedSegmentation
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from medpy.io import load as medload
from glob import glob
import os
import tifffile
import albumentations as A


class ChunkedBatchIdSampler:
    def __init__(self, dset, batch_size, batch_count=None, distribution='even', temperature=None,):
        self.chunks_number = len(dset.datasets)
        self.chunk_lengths = [len(d) for d in dset.datasets]
        self.chunk_padding = np.cumsum([0] + self.chunk_lengths)

        if distribution == 'gaussian':
            self.sampler = lambda s: np.clip(np.random.randn(s)/temperature, -0.5, 0.5) + 0.5
        elif distribution == 'even':
            self.sampler = lambda s: np.random.random(s)
        else:
            raise ValueError('Unknown distribution family {distribution}')


        self.batch_size = batch_size
        self.batch_count = int(len(dset) / batch_size) if batch_count is None else batch_count

    def __iter__(self):
        for i in range(self.batch_count):
            yield self.__call__()

    def __len__(self):
        return self.batch_count

    def __call__(self):
        chunk_id = np.random.randint(self.chunks_number)
        relative_interchunk = self.sampler(self.batch_size)
        interchunk_id = (relative_interchunk * (self.chunk_lengths[chunk_id] -1)).astype(np.int)
        return interchunk_id + self.chunk_padding[chunk_id]

class DenseLocalisationDataset(Dataset):
    def __init__(self, volume, augmentation, mask=None, checkpoint_count=256):
        self.volume = volume
        self.mask = mask
        
        self.augmentation = augmentation
        self.checkpoint_count = checkpoint_count
    
    def __len__(self):
        return len(self.volume)
    
    def __getitem__(self, id):
        img = self.volume[id]
        kp = np.stack([np.random.randint(0, img.shape[1], self.checkpoint_count), 
                               np.random.randint(0, img.shape[0], self.checkpoint_count)], -1)
        kpl = np.concatenate([np.ones((self.checkpoint_count, 1))*id, kp], -1)
        
        if self.mask is not None:
            mask = self.mask[id]
        else:
            mask = np.ones_like(img, dtype=np.uint8)*255
        
        rst = self.augmentation(image=img, keypoints=kp, position=kpl, mask=mask)
        kp = np.array(rst['keypoints'])
        kpl = np.array(rst['position'])
        
        return rst['image'][None,...], kp, kpl, rst['mask'][None,...]

class ConrsativeSegmentationDataset(Dataset):
    def __init__(self, volume, augmentation, mask=None, checkpoint_count=256, crop_size=256):
        self.volume = volume
        self.mask = mask
        
        self.augmentation = augmentation
        self.checkpoint_count = checkpoint_count

        self.cropper = A.RandomCrop(width=crop_size, height=crop_size)

    def _match_keyopints(self, keypoints1, ids1, keypoints2, ids2):
        matched_ids = list(set(ids1[:, 0]) & set(ids2[:, 0]))
        ids1 = [np.where(ids1==i)[0][0] for i in matched_ids]
        ids2 = [np.where(ids2==i)[0][0] for i in matched_ids]

        return keypoints1[ids1], keypoints2[ids2]
    
    def __len__(self):
        return len(self.volume)
    
    def __getitem__(self, id):
        img = self.volume[id]

        if self.mask is not None:
            mask = self.mask[id]
        else:
            mask = np.ones_like(img, dtype=np.uint8)*255

        crpt = self.cropper(image=img, mask=mask)
        img = crpt['image']
        mask = crpt['mask']

        kp = np.stack([np.random.randint(0, img.shape[1], self.checkpoint_count), 
                               np.random.randint(0, img.shape[0], self.checkpoint_count)], -1)
        kpl = np.arange(0, self.checkpoint_count)[:, None]
        
        view1 = self.augmentation(image=img, keypoints=kp, position=kpl, mask=mask)
        view2 = self.augmentation(image=img, keypoints=kp, position=kpl, mask=mask)

        img = np.stack([view1['image'], view2['image']])
        
        kp = np.stack(self._match_keyopints(np.array(view1['keypoints']),
                                            np.array(view1['position']), 
                                            np.array(view2['keypoints']), 
                                            np.array(view2['position'])))
        msk = np.stack([view1['mask'], view2['mask']])

        return img, kp, kpl, msk

def partial_collate(samples):
    imgs, kps, poss, msks = zip(*samples)
    imgs = torch.from_numpy(np.array(imgs))
    msks = torch.from_numpy(np.array(msks))
    return imgs, kps, poss, msks

def list_collate_fn(batch):
    return list(zip(*batch))

def convert_target(addr, converter):
    if isinstance(list(converter.keys())[0], str):
        # this is because of the restrictions in the OmegaConf. Should be resolved with 2.1 version.
        converter = {int(k):v for k,v in converter.items()}
    markup = ExpandedPaddedSegmentation(addr)
    markup.data = np.vectorize(converter.get)(markup.data)
    return markup

def supervised_segmentation_target_matcher(volumes, targets):
    label_ids = [os.path.basename(i).split('.')[-2] for i in glob(targets.format('*'))]

    if '-' in label_ids[0]:
        volume_ids = [i.split('-')[1] for i in label_ids]
    else:
        volume_ids = label_ids
    
    return [volumes.format(i) for i in volume_ids], [targets.format(i) for i in label_ids]

def semi_supervised_segmentation_target_matcher(supervised_volumes, supervised_targets, unsupervised_volumes):
    supervised_volumes, supervised_targets = supervised_segmentation_target_matcher(supervised_volumes, supervised_targets)
    unsupervised_volumes = glob(unsupervised_volumes.format('*'))

    return (supervised_volumes, supervised_targets), unsupervised_volumes

def sklearn_train_test_split(gathered_data, random_state=None, train_volumes=None, volumes_limit=None):
    volumes_limit = volumes_limit or len(gathered_data[0])
    train_data, test_data = train_test_split(list(zip(*gathered_data))[:volumes_limit], random_state=random_state, train_size=train_volumes)
    return train_data, test_data

def semi_supervised_train_test_split(gathered_data, unsupervised_test_part=False, random_state=None, 
                                     train_volumes_supervised=None, train_volumes_unsupervised=None, 
                                     supervised_volumes_limit=None, unsupervised_volumes_limit=None):
    supervised_data, unsupervised_data = gathered_data
    if supervised_volumes_limit == 0:
        supervised_train, supervised_test = [], []
    else:
        supervised_train, supervised_test = sklearn_train_test_split(supervised_data, random_state, train_volumes_supervised, supervised_volumes_limit)
    
    if unsupervised_volumes_limit == 0:
        unsupervised_train, unsupervised_test = [], []
    elif unsupervised_test_part:
        unsupervised_train, unsupervised_test = sklearn_train_test_split([unsupervised_data], random_state, train_volumes_unsupervised, unsupervised_volumes_limit)
    else:
        unsupervised_train = unsupervised_data[:unsupervised_volumes_limit][:train_volumes_unsupervised]
        unsupervised_test = []
    
    return (supervised_train + unsupervised_train), (supervised_test + unsupervised_test)

def get_TVSD_datasets(data_addresses, aug=None, label_converter=None, **kwargs):
    datasets = []
    for image_addr, label_addr in data_addresses:
        if label_converter is not None:
            label = convert_target(label_addr, label_converter)
        else:
            label = ExpandedPaddedSegmentation(label_addr)
        
        datasets.append(VolumeSlicingDataset(image_addr, segmentation=label, augmentations=aug,
                                             **kwargs))
    return ConcatDataset(datasets)

def get_DLD_datasets(data_addresses, aug=None, label_converter=None, **kwargs):
    datasets = []
    for addr_batch in data_addresses:
        if isinstance(addr_batch, tuple) and len(addr_batch) == 2:
            image_addr = addr_batch[0]
            image = tifffile.imread(image_addr)
            label_addr = addr_batch[1]
            if label_converter is not None:
                label = convert_target(label_addr, label_converter)
            else:
                label = medload(label_addr)[0]
        else:
            if isinstance(addr_batch, tuple):
                addr_batch = addr_batch[0]
            image_addr = addr_batch
            image = tifffile.imread(image_addr)
            label = None
        
        datasets.append(DenseLocalisationDataset(image, augmentation=aug, mask=label, **kwargs))
    return ConcatDataset(datasets)

def get_CSD_datasets(data_addresses, aug=None, label_converter=None, **kwargs):
    datasets = []
    for addr_batch in data_addresses:
        if isinstance(addr_batch, tuple) and len(addr_batch) == 2:
            image_addr = addr_batch[0]
            image = tifffile.imread(image_addr)
            label_addr = addr_batch[1]
            if label_converter is not None:
                label = convert_target(label_addr, label_converter)
            else:
                label = medload(label_addr)[0]
        else:
            if isinstance(addr_batch, tuple):
                addr_batch = addr_batch[0]
            image_addr = addr_batch
            image = tifffile.imread(image_addr)
            label = None
        
        datasets.append(ConrsativeSegmentationDataset(image, augmentation=aug, mask=label, **kwargs))
    return ConcatDataset(datasets)

def adaptive_choice(choose_from, choice_count):
    if choice_count <= len(choose_from):
        return np.random.choice(choose_from, choice_count, replace=False)
    else:
        subsample = [choose_from]*(choice_count//len(choose_from)) # all the full inclusions first
        subsample.append(np.random.choice(choose_from, choice_count%len(choose_from), replace=False)) # additional records
        return np.concatenate(subsample)

def multiple_dataset_resample(resampling_function):
    def wrapper_resampler(datasets, multiple_datasets_mode='all', **kwargs):
        if multiple_datasets_mode == 'first':
            return [resampling_function(datasets[0], **kwargs)] + datasets[1:]
        elif multiple_datasets_mode == 'all':
            return [resampling_function(dset, **kwargs) for dset in datasets]
        elif multiple_datasets_mode == 'default':
            return [resampling_function(datasets[0], **kwargs)] + [resampling_function(dset) for dset in datasets[1:]]
    
    return wrapper_resampler

@multiple_dataset_resample
def TVSD_dataset_resample(dataset, segmented_part=1.0, empty_part=0.1):
    is_marked = np.concatenate([d.segmentation._contains_markup() for d in dataset.datasets])
    if segmented_part is None:
        segmented_part = 1.0
    if isinstance(segmented_part, float):
        segmented_part = int(is_marked.sum() * segmented_part)        
    
    if isinstance(empty_part, float):
        empty_part = int(segmented_part * empty_part)
    elif empty_part is None:
        empty_part = (1-is_marked).sum()

    segmented_subsample = adaptive_choice(np.where(is_marked)[0], segmented_part)
    empty_subsample = adaptive_choice(np.where(1-is_marked)[0], empty_part)

    return Subset(dataset, np.concatenate([segmented_subsample, empty_subsample]))

@multiple_dataset_resample
def DLD_dataset_resample(dataset, segmented_part=1.0, empty_part=0.1, unsupervised_ratio=0.2):
    is_marked = []
    if dataset.mask is None:
        idx = np.random.choice(np.arange(len(dataset)), int(unsupervised_ratio*len(dataset)))
    else:
        is_marked = dataset.mask.sum((1, 2)) > 0

        if segmented_part is None:
            segmented_part = 1.0
        if isinstance(segmented_part, float):
            segmented_part = int(is_marked.sum() * segmented_part)
    
        if isinstance(empty_part, float):
            empty_part = int(segmented_part * empty_part)
        elif empty_part is None:
            empty_part = (1-is_marked).sum()

        segmented_subsample = np.random.choice(np.where(is_marked)[0], segmented_part, replace=False)
        empty_subsample = np.random.choice(np.where(1-is_marked)[0], empty_part, replace=False)

        idx = np.concatenate([segmented_subsample, empty_subsample])

    return Subset(dataset, idx)

@multiple_dataset_resample
def resample_concatenated_DLD(dataset, **kwargs):
    resampled_dsets = [DLD_dataset_resample(d, **kwargs) for d in dataset.datasets]
    return ConcatDataset(resampled_dsets)

from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from TVSD import VolumeSlicingDataset, ExpandedPaddedSegmentation
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from medpy.io import load as medload
from glob import glob
import os

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
    
    return list(zip([volumes.format(i) for i in volume_ids], [targets.format(i) for i in label_ids]))

def same_name_target_matcher(*addrs):
    addr_dicts = []
    file_names = []
    for addr in addrs:
        addr_dicts.append({os.path.basename(i): i for i in glob(addr)})
        file_names.append(set(addr_dicts[-1].keys()))
    
    joint_filenames = set.intersection(*file_names)

    result_tuples = []
    for fn in joint_filenames:
        result_tuples.append(tuple([d[fn] for d in addr_dicts]))
    return result_tuples

def sklearn_train_test_split(gathered_data, random_state=None, train_volumes=None, volumes_limit=None):
    if volumes_limit is not None:
        gathered_data = gathered_data[:volumes_limit]
    train_data, test_data = train_test_split(gathered_data, random_state=random_state, train_size=train_volumes)
    return train_data, test_data

def get_TVSD_datasets(data_addresses, aug=None, **kwargs):
    datasets = []
    for image_addr, label_addr in data_addresses:
        datasets.append(VolumeSlicingDataset(image_addr, segmentation=label_addr, augmentations=aug,
                                             **kwargs))
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

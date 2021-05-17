from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from TVSD import VolumeSlicingDataset
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from medpy.io import load as medload
from glob import glob
import os
import tifffile

def convert_target(addr, converter):
    if isinstance(list(converter.keys())[0], str):
        # this is because of the restrictions in the OmegaConf. Should be resolved with 2.1 version.
        converter = {int(k):v for k,v in converter.items()}
    volume = medload(addr)[0]
    volume = np.vectorize(converter.get)(volume)
    return volume

def supervised_segmentation_target_matcher(volumes, targets):
    label_ids = [os.path.basename(i).split('.')[-2] for i in glob(targets.format('*'))]

    if '-' in label_ids[0]:
        volume_ids = [i.split('-')[1] for i in label_ids]
    else:
        volume_ids = label_ids
    
    return [volumes.format(i) for i in volume_ids], [targets.format(i) for i in label_ids]

def sklearn_train_test_split(gathered_data, random_state=None, train_volumes=None, volumes_limit=None):
    volumes_limit = volumes_limit or len(gathered_data[0])
    train_data, test_data = train_test_split(list(zip(*gathered_data))[:volumes_limit], random_state=random_state, train_size=train_volumes)
    return train_data, test_data

def get_TVSD_datasets(data_addresses, aug=None, label_converter=None, **kwargs):
    datasets = []
    for image_addr, label_addr in data_addresses:
        if label_converter is not None:
            label = convert_target(label_addr, label_converter)
        else:
            label = medload(label_addr)[0]
        
        datasets.append(VolumeSlicingDataset(image_addr, segmentation=label, augmentations=aug,
                                             **kwargs))
    return ConcatDataset(datasets)

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

    segmented_subsample = np.random.choice(np.where(is_marked)[0], segmented_part, replace=False)
    empty_subsample = np.random.choice(np.where(1-is_marked)[0], empty_part, replace=False)

    return Subset(dataset, np.concatenate([segmented_subsample, empty_subsample]))


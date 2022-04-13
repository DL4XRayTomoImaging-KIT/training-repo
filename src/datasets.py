from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from TVSD import VolumeSlicingDataset, ExpandedPaddedSegmentation
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from medpy.io import load as medload
from glob import glob
import os
import tifffile
from tqdm.auto import tqdm

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

def pseudo_target_matcher(multi_pseudo_targets):
  volumes, targets = [], []

  for pseudo_targets in multi_pseudo_targets:
    pseudo_ids = [os.path.basename(os.path.dirname(i)) for i in glob(pseudo_targets.format('*'))]

    # remove organ name form target to get volume name
    dir_name, file_name = os.path.split(pseudo_targets)
    file_name = '_'.join(file_name.split('_')[1:])
    pseudo_volumes = os.path.join(dir_name, file_name)

    path_formatter = lambda path_, ids: [path_.format(i) for i in ids]
    volumes.extend(path_formatter(pseudo_volumes, pseudo_ids))
    targets.extend(path_formatter(pseudo_targets, pseudo_ids))

  return volumes, targets
  
def get_enlarged_dataset(volumes, targets, pseudo_targets):
  volumes, targets = supervised_segmentation_target_matcher(volumes, targets)
  pseudo_volumes, pseudo_targets = pseudo_target_matcher(pseudo_targets)
  return volumes+pseudo_volumes, targets+pseudo_targets

def sklearn_train_test_split(gathered_data, random_state=None, train_volumes=None, volumes_limit=None):
    volumes_limit = volumes_limit or len(gathered_data[0])
    train_data, test_data = train_test_split(list(zip(*gathered_data))[:volumes_limit], random_state=random_state, train_size=train_volumes)
    return train_data, test_data

def get_TVSD_datasets(data_addresses, aug=None, label_converter=None, **kwargs):
    datasets = []
    for image_addr, label_addr in tqdm(data_addresses, desc='applying dataset function'):
        try:
          if label_converter is not None:
              label = convert_target(label_addr, label_converter)
          else:
              label = ExpandedPaddedSegmentation(label_addr)
          
          datasets.append(VolumeSlicingDataset(image_addr, segmentation=label, augmentations=aug,
                                               **kwargs))
        except Exception as e:  # arises when mask contains no organs
          print(f'img_addr: {image_addr}')
          print(f'mask_addr: {label_addr}')
          print(f'error: {e}')
    return ConcatDataset(datasets)

def adaptive_choice(choose_from, choice_count):
    if choice_count <= len(choose_from):
        return np.random.choice(choose_from, choice_count, replace=False)
    else:
        subsample = [choose_from]*(choice_count//len(choose_from)) # all the full inclusions first
        subsample.append(np.random.choice(choose_from, choice_count%len(choose_from), replace=False)) # additional records
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
        empty_part = (1-is_marked).sum()

    segmented_subsample = adaptive_choice(np.where(is_marked)[0], segmented_part)
    empty_subsample = adaptive_choice(np.where(1-is_marked)[0], empty_part)

    return Subset(dataset, np.concatenate([segmented_subsample, empty_subsample]))

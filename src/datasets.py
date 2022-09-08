from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from TVSD import VolumeSlicingDataset, ExpandedPaddedSegmentation
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from glob import glob
import os
import cv2
import json
from tqdm.auto import tqdm
import src.filters as filters


def convert_target(addr, converter):
    if isinstance(list(converter.keys())[0], str):
        # this is because of the restrictions in the OmegaConf. Should be resolved with 2.1 version.
        converter = {int(k):v for k,v in converter.items()}
    markup = ExpandedPaddedSegmentation(addr)
    markup.data = np.vectorize(converter.get)(markup.data)
    return markup

path_formatter = lambda path_, ids: [path_.format(i) for i in ids]

def supervised_segmentation_target_matcher(volumes, targets):
    label_ids = [os.path.basename(i).split('.')[-2] for i in glob(targets.format('*'))]
    if '-' in label_ids[0]:
        volume_ids = [i.split('-')[1] for i in label_ids]
    else:
        volume_ids = label_ids
    return list(zip(path_formatter(volumes, volume_ids), path_formatter(targets, label_ids)))

def pseudo_target_matcher(volumes, targets):
    label_ids = [os.path.basename(os.path.dirname(i)) for i in glob(targets.format('*'))]
    return list(zip(path_formatter(volumes, label_ids), path_formatter(targets, label_ids)))
    
def self_matcher(volumes):
  volume_ids = glob(volumes.format('*'))
  return [(volume_id,volume_id) for volume_id in volume_ids]

def sklearn_train_test_split(gathered_data, random_state=None, train_volumes=None, volumes_limit=None):
    if volumes_limit is not None:
        gathered_data = gathered_data[:volumes_limit]
    train_data, test_data = train_test_split(gathered_data, random_state=random_state, train_size=train_volumes)
    return train_data, test_data

def get_TVSD_datasets(data_addresses, aug=None, **kwargs):
    datasets = []
    
    if 'filter_function' in kwargs:
        filter_function = kwargs.pop('filter_function')
        filter_kwargs = kwargs.pop('filter_kwargs', None)
        filter_kwargs = filter_kwargs or {}
        filter_function = getattr(filters, filter_function)
            
        for image_addr, label_addr in tqdm(data_addresses, desc='getting filtered TVSD datasets'):
            exclude_slices = filter_function(label_addr, **filter_kwargs)
            TVSD_dataset = my_VolumeSlicingDataset(image_addr, segmentation=label_addr, augmentations=aug,
                                               exclude_slices=exclude_slices, **kwargs)
            datasets.append(TVSD_dataset)
    else:
        for image_addr, label_addr in tqdm(data_addresses, desc='getting TVSD datasets'):
            TVSD_dataset = VolumeSlicingDataset(image_addr, segmentation=label_addr, augmentations=aug,
                                               **kwargs)
            datasets.append(TVSD_dataset)
            
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
def TVSD_dataset_resample(dataset, segmented_part=1.0, empty_part=0.1, contains_markup_path=None):

    if contains_markup_path is not None:
        with open(contains_markup_path) as f:
            contains_markup_data = json.load(f)

    is_marked = []
    for d in tqdm(dataset.datasets, desc='resampling TVSD datasets'):
        if contains_markup_path is None:
            ne_slices = d.segmentation._contains_markup()
        else:
            ne_slices = contains_markup_data[d.segmentation.file_addr]
            
        ne_slices = np.array(ne_slices, dtype=int)
        if hasattr(d, 'exclude_slices'):
          # would be none for slices in original data
          if d.exclude_slices is not None:
            ne_slices[d.exclude_slices] = -1
            
        is_marked.append(ne_slices)         
    is_marked = np.concatenate(is_marked)
        
    if segmented_part is None:
        segmented_part = 1.0
    if isinstance(segmented_part, float):
        segmented_part = int((is_marked==1).sum() * segmented_part)        
    
    if isinstance(empty_part, float):
        empty_part = int(segmented_part * empty_part)
    elif empty_part is None:
        empty_part = (is_marked==0).sum()

    segmented_subsample = adaptive_choice(np.where(is_marked==1)[0], segmented_part)
    empty_subsample = adaptive_choice(np.where(is_marked==0)[0], empty_part)
    collective_subsample = np.concatenate([segmented_subsample, empty_subsample])

    return Subset(dataset, collective_subsample)



######################


def classification_addr_slices(sliceQuality_data_path):
    with open(sliceQuality_data_path, 'r') as fp:
        sliceQuality_data = json.load(fp)

    labelledSlices = []
    lbl_dict = {'good': 0, 'bad': 1}
    for msk_addr, slcs in sliceQuality_data.items():
      img_addr = msk_addr.replace('brain_scaled', 'scaled')
      for slc_id, slc_lbl in slcs.items():
        if slc_lbl in ['good', 'bad']:  # ignore mid
          labelledSlices.append( ((img_addr, msk_addr, int(slc_id)), lbl_dict[slc_lbl]) )
          
    return labelledSlices    

class my_VolumeSlicingDataset(VolumeSlicingDataset):
    def __init__(self, *args, exclude_slices=None, **kwargs):
        VolumeSlicingDataset.__init__(self, *args, **kwargs)
        self.exclude_slices = exclude_slices

class LabelledSlicesDataset(Dataset):
    """Manually Labelled Slices dataset."""

    def __init__(self, labelledSlices, aug=None, **kwargs):
        """
        Args:
            labelledSlices (list): list of tuples (img_addr, msk_addr, slc_id, slc_lbl)
        """
          
        self.labelledSlices = []
        n_classes = 10  # give each class a separate channel
        for (img_addr, msk_addr, slc_id), slc_lbl in tqdm(labelledSlices, desc='getting classification datasets'):
          TVSD_dataset = VolumeSlicingDataset(img_addr, segmentation=msk_addr, augmentations=aug,
                                             **kwargs)
          img, msk = TVSD_dataset[slc_id]
          # img = cv2.resize(img[0], dsize=(512, 512))
          # msk = cv2.resize(msk[0], dsize=(512, 512))
          # img, msk = np.expand_dims(img, 0), np.expand_dims(msk, 0)
          
          msk_channels = np.zeros((n_classes, *msk.shape[1:]))
          img = np.concatenate((img, msk_channels), dtype=img.dtype)
          
          for class_ in range(n_classes):
            idx = (msk == class_)[0]                                    
            img[class_ + 1][idx] = 1
          
          self.labelledSlices.append((img, slc_lbl))

    def __len__(self):
        return len(self.labelledSlices)

    def __getitem__(self, idx):
        slc, slc_lbl = self.labelledSlices[idx]
        return slc, slc_lbl
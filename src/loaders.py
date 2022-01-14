import src.datasets as datasets
import src.augmentations as augmentations
from torch.utils.data import DataLoader
import os

def generic_loaders(data_gatherer_name='supervised_segmentation_target_matcher', data_gatherer_kwargs=None,
                train_test_split_function='sklearn_train_test_split', seed=None, train_test_split_kwargs=None,
                aug_name='none_aug',
                dataset_function_name='get_TVSD_datasets', dataset_kwargs=None,
                dataset_rebalance_function_name=None, dataset_rebalance_kwargs=None,
                collate_fn_name=None,
                sampler_function_name=None, sampler_function_kwargs=None,
                dataloader_kwargs=None):       
    
    # gather data
    data_gatherer_kwargs = data_gatherer_kwargs or {}
    gathered_data = getattr(datasets, data_gatherer_name)(**data_gatherer_kwargs)

    # create train-test split, limiting the overall loading capacity
    data_splitter_kwargs = {'random_state': seed}
    data_splitter_kwargs.update(train_test_split_kwargs or {})
    train_data, test_data = getattr(datasets, train_test_split_function)(gathered_data, **data_splitter_kwargs)

    # define augmentation, since it should be passed to the dataset
    aug = getattr(augmentations, aug_name)

    # construct train and test datasets itself
    dataset_kwargs = dataset_kwargs or {}
    train_set = getattr(datasets, dataset_function_name)(train_data, aug=aug, **dataset_kwargs)
    test_set = getattr(datasets, dataset_function_name)(test_data, aug=aug, **dataset_kwargs)

    # rebalance dataset
    if dataset_rebalance_function_name is not None:
        dataset_rebalance_kwargs = dataset_rebalance_kwargs or {}
        train_set, test_set = getattr(datasets, dataset_rebalance_function_name)([train_set, test_set], **dataset_rebalance_kwargs)
    
    # select collate function
    collate_fn = getattr(datasets, collate_fn_name) if collate_fn_name is not None else None

    # get samplers
    if sampler_function_name is not None:
        train_sampler = getattr(datasets, sampler_function_name)(train_set, **sampler_function_kwargs)
        train_loader_kw = {'batch_sampler': train_sampler}
        test_sampler = getattr(datasets, sampler_function_name)(test_set, **sampler_function_kwargs)
        test_loader_kw = {'batch_sampler': test_sampler}
    else:
        train_loader_kw = {'drop_last': True, 'shuffle': True}
        test_loader_kw = {'drop_last': True, 'shuffle': True}
    
    # get s
    train_loader_kw.update({'num_workers': 16, 'pin_memory': True})
    test_loader_kw.update({'num_workers': 16, 'pin_memory': True})

    if dataloader_kwargs is not None:
        train_loader_kw.update(dataloader_kwargs)
        test_loader_kw.update(dataloader_kwargs)

    # get data loaders
    train_loader = DataLoader(train_set, collate_fn=collate_fn, **train_loader_kw)
    test_loader = DataLoader(test_set, collate_fn=collate_fn, **test_loader_kw)

    return {'train': train_loader, 'valid': test_loader}, None

def triple_presliced_loaders(paths,
                train_test_split_function='sklearn_train_test_split', seed=None, train_test_split_kwargs=None,
                aug_name='none_aug',
                dataset_function_name='get_TVSD_datasets', dataset_kwargs=None,
                dataloader_kwargs=None):       
    
    # gather data
    volumes_full, targets_full = datasets.supervised_segmentation_target_matcher(paths['volumes'], paths['targets_full'])
    volumes_sliced, targets_sliced = datasets.supervised_segmentation_target_matcher(paths['volumes'], paths['targets_sliced'])

    volumes = {os.path.basename(t).split('.')[0]:v for t,v in zip(targets_full, volumes_full)}
    fulls_targets = {os.path.basename(t).split('.')[0]:t for t in targets_full}
    sliceds_targets = {os.path.basename(t).split('.')[0]:t for t in targets_sliced}

    ids = list(volumes.keys())

    volumes = [volumes[i] for i in ids]
    fulls_targets = [fulls_targets[i] for i in ids]
    sliceds_targets = [sliceds_targets[i] for i in ids]

    gathered_data = (volumes, fulls_targets, sliceds_targets)


    # create train-test split, limiting the overall loading capacity
    data_splitter_kwargs = {'random_state': seed}
    data_splitter_kwargs.update(train_test_split_kwargs or {})
    train_data, test_data = getattr(datasets, train_test_split_function)(gathered_data, **data_splitter_kwargs)
    print('SpecLogOccSearch> ', [os.path.basename(i[0]) for i in train_data])

    valid_data = [(i[0], i[1]) for i in train_data] # interpolation
    train_data = [(i[0], i[2]) for i in train_data] # training
    test_data = [(i[0], i[1]) for i in test_data]  # extrapolation

    # define augmentation, since it should be passed to the dataset
    aug = getattr(augmentations, aug_name)

    # construct train and test datasets itself
    dataset_kwargs = dataset_kwargs or {}
    train_set = getattr(datasets, dataset_function_name)(train_data, aug=aug, **dataset_kwargs)
    valid_set = getattr(datasets, dataset_function_name)(valid_data, aug=augmentations.none_aug, **dataset_kwargs)
    test_set = getattr(datasets, dataset_function_name)(test_data, aug=augmentations.none_aug, **dataset_kwargs)

    # rebalance dataset
    small_valid_set = datasets.TVSD_dataset_resample([valid_set], segmented_part=0.1, empty_part=0)[0]
    train_set, valid_set, test_set = datasets.TVSD_dataset_resample([train_set, valid_set, test_set], empty_part=0, multiple_datasets_mode='all')
    
    train_loader_kw = {'drop_last': False, 'shuffle': True, 'num_workers': 16, 'pin_memory': True}
    test_loader_kw = {'drop_last': True, 'shuffle': True, 'num_workers': 16, 'pin_memory': True}

    if dataloader_kwargs is not None:
        train_loader_kw.update(dataloader_kwargs)
        test_loader_kw.update(dataloader_kwargs)

    # get data loaders
    train_loader = DataLoader(train_set, **train_loader_kw)
    valid_loader = DataLoader(valid_set, **test_loader_kw)
    test_loader = DataLoader(test_set, **test_loader_kw)
    small_test_loader = DataLoader(small_valid_set, **test_loader_kw)

    return {'train': train_loader, 'valid': small_test_loader}, {'infer_extrapolation': test_loader, 'infer_interpolation': valid_loader}

def triple_loaders(data_gatherer_name='supervised_segmentation_target_matcher', data_gatherer_kwargs=None,
                train_test_split_function='sklearn_train_test_split', seed=None, train_test_split_kwargs=None,
                aug_name='none_aug',
                dataset_function_name='get_TVSD_datasets', dataset_kwargs=None,
                dataset_rebalance_function_name=None, dataset_rebalance_kwargs=None,
                collate_fn_name=None,
                dataloader_kwargs=None):       
    
    # gather data
    data_gatherer_kwargs = data_gatherer_kwargs or {}
    gathered_data = getattr(datasets, data_gatherer_name)(**data_gatherer_kwargs)

    # create train-test split, limiting the overall loading capacity
    data_splitter_kwargs = {'random_state': seed}
    data_splitter_kwargs.update(train_test_split_kwargs or {})
    train_data, test_data = getattr(datasets, train_test_split_function)(gathered_data, **data_splitter_kwargs)

    # define augmentation, since it should be passed to the dataset
    aug = getattr(augmentations, aug_name)

    # construct train and test datasets itself
    dataset_kwargs = dataset_kwargs or {}
    train_set = getattr(datasets, dataset_function_name)(train_data, aug=aug, **dataset_kwargs)
    test_set = getattr(datasets, dataset_function_name)(test_data, aug=aug, **dataset_kwargs)

    # rebalance dataset
    if dataset_rebalance_function_name is not None:
        dataset_rebalance_kwargs = dataset_rebalance_kwargs or {}
        dataset_rebalance_kwargs['multiple_datasets_mode'] = 'default' # just to ease configs, since this function is for special case anyways
        train_set, valid_set, test_set = getattr(datasets, dataset_rebalance_function_name)([train_set, train_set, test_set], **dataset_rebalance_kwargs)
    
    # select collate function
    collate_fn = getattr(datasets, collate_fn_name) if collate_fn_name is not None else None


    train_loader_kw = {'drop_last': True, 'shuffle': True}
    test_loader_kw = {'drop_last': True, 'shuffle': True}

    train_loader_kw.update({'num_workers': 16, 'pin_memory': True})
    test_loader_kw.update({'num_workers': 16, 'pin_memory': True})

    if dataloader_kwargs is not None:
        train_loader_kw.update(dataloader_kwargs)
        test_loader_kw.update(dataloader_kwargs)

    # get data loaders
    train_loader = DataLoader(train_set, collate_fn=collate_fn, **train_loader_kw)
    valid_loader = DataLoader(valid_set, collate_fn=collate_fn, **test_loader_kw)
    test_loader = DataLoader(test_set, collate_fn=collate_fn, **test_loader_kw)

    return {'train': train_loader, 'valid': valid_loader}, {'infer_extrapolation': test_loader}

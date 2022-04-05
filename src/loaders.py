import src.datasets as datasets
import src.augmentations as augmentations
from torch.utils.data import DataLoader
from copy import deepcopy

def gather_one_collection(matcher_funcion='supervised_segmentation_target_matcher', matcher_kwargs=None, 
                    split_function='sklearn_train_test_split', seed=None, split_kwargs=None, names=['train', 'valid']):

    # gather data
    matcher_kwargs = matcher_kwargs or {}
    gathered_data = getattr(datasets, matcher_funcion)(**matcher_kwargs)

    # create train-test split, limiting the overall loading capacity
    if split_function is not None:
        data_splitter_kwargs = {'random_state': seed}
        data_splitter_kwargs.update(split_kwargs or {})
        gathered_data = getattr(datasets, split_function)(gathered_data, **data_splitter_kwargs)

        return {n: v for n,v in zip(names, gathered_data)}
    else:
        return {names: gathered_data}

def gather_collections(dicts_of_parameters, seed):
    gathered_collections = {}
    if not isinstance(dicts_of_parameters, list):
        dicts_of_parameters = [dicts_of_parameters]

    for parameters_set in dicts_of_parameters:
        if not ('seed' in parameters_set.keys()):
            parameters_set['seed'] = seed
        gathered_collections.update(gather_one_collection(**parameters_set))
    return gathered_collections


def get_one_loader(data, aug_name='none_aug',
                  dataset_function_name='get_TVSD_datasets', dataset_kwargs=None,
                  dataset_rebalance_function_name=None, dataset_rebalance_kwargs=None,
                  collate_fn_name=None,
                  sampler_function_name=None, sampler_function_kwargs=None,
                  dataloader_kwargs=None):
    
    # define augmentation, since it should be passed to the dataset
    aug = getattr(augmentations, aug_name)

    # construct train and test datasets itself
    dataset_kwargs = dataset_kwargs or {}
    dataset = getattr(datasets, dataset_function_name)(data, aug=aug, **dataset_kwargs)

    # rebalance dataset
    if dataset_rebalance_function_name is not None:
        dataset_rebalance_kwargs = dataset_rebalance_kwargs or {}
        dataset = getattr(datasets, dataset_rebalance_function_name)([dataset], **dataset_rebalance_kwargs)[0]
    
    # select collate function
    collate_fn = getattr(datasets, collate_fn_name) if collate_fn_name is not None else None

    # get samplers
    if sampler_function_name is not None:
        sampler = getattr(datasets, sampler_function_name)(dataset, **sampler_function_kwargs)
        loader_kw = {'batch_sampler': sampler}
    else:
        loader_kw = {'drop_last': True, 'shuffle': True}
    
    # get s
    loader_kw.update({'num_workers': 16, 'pin_memory': True})

    if dataloader_kwargs is not None:
        loader_kw.update(dataloader_kwargs)

    # get data loaders
    loader = DataLoader(dataset, collate_fn=collate_fn, **loader_kw)

    return loader


def get_loaders(dicts_of_parameters, dict_of_data):
    if not isinstance(dicts_of_parameters, list):
        dicts_of_parameters = [dicts_of_parameters]
    
    in_loop = {}
    post_loop = {}

    parameters_by_name = {}
    for parameters_set in dicts_of_parameters:
        apply_to = parameters_set.pop('dataset_names')

        if not isinstance(apply_to, list):
            apply_to = [apply_to]

        for dataset_name in apply_to:
            parameters_by_name[dataset_name] = deepcopy(parameters_set)
        
    if 'default' in parameters_by_name.keys():
        default_dataset_params = parameters_by_name.pop('default')

        for dataset_name in parameters_by_name.keys():
            parameters_by_name[dataset_name] = default_dataset_params | parameters_by_name[dataset_name]

    for dataset_name, parameters_set in parameters_by_name.items():
        if parameters_set.pop('in_loop', True):
            in_loop[dataset_name] = get_one_loader(dict_of_data[dataset_name], **parameters_set)
        else:
            post_loop[dataset_name] = get_one_loader(dict_of_data[dataset_name], **parameters_set)
    
    return in_loop, post_loop

def generic_loaders(gatherers, datasets, seed):
    addresses = gather_collections(gatherers, seed)
    loaders_in_loop, loaders_post_loop = get_loaders(datasets, addresses)
    return loaders_in_loop, loaders_post_loop
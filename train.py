# imports

import sys, os

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import numpy as np
import json

from tqdm.auto import tqdm
from matplotlib import pyplot as plt

from glob import glob

from sklearn.model_selection import train_test_split

#from src.datasets import MultiTiffSegmentation, ExpandedPaddedMarkup, Tiff3D
#from src.predefined_augmentations import light_aug

from medpy.io import load as medload

#from src.lovasz_losses import lovasz_softmax, iou
#
#from src.segmenting_helpers import get_weighted_sampler_by_mark, get_weighted_sampler_by_sum

import torch
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler, Subset
from torch.utils.data import DistributedSampler


import torch
from torch import nn

import kornia

import wandb

from catalyst.dl import Runner, SupervisedRunner
from catalyst import dl, metrics
from catalyst.contrib.callbacks.wandb_logger import WandbLogger
from catalyst.callbacks.metric import BatchMetricCallback
from collections import defaultdict

from torch.nn import NLLLoss2d, CrossEntropyLoss

import hydra
from omegaconf import DictConfig, OmegaConf

import src.augmentations
import src.models
import src.datasets
import src.losses
import src.callbacks
import src.runners

from functools import partial

from skimage.measure import label

from src.hydra_helpers import cfg_ut, check_exclusion

# definitions

tn = lambda t: t.detach().cpu().numpy()

get_files_by_ids = lambda ids, address: [address.format(i) for i in ids]

@cfg_ut('model', 'initializing model')
def get_model(network_type, *args, **kwargs):
    model = getattr(src.models, network_type)(*args, **kwargs)
    trainable_parameters = sum(p.numel() for p in model.parameters())
    print(f'trainable parameters in model: {trainable_parameters}')
    return model

@cfg_ut('checkpoint', 'loading checkpoint')
def load_weights(model, checkpoint_addr=None,
                 checkpoint_key=None, model_key=None,
                 load_function=None, **load_function_params):

    if checkpoint_addr is None:
        return model

    state_dict = torch.load(checkpoint_addr)

    if checkpoint_key is not None:
        state_dict = state_dict[checkpoint_key]
    if model_key is not None:
        model = getattr(model, model_key)

    if load_function is None:
        model.load_state_dict(state_dict)
    else:
        getattr(src.models, load_function)(model, state_dict, **load_function_params)

@cfg_ut('dataset', 'loading datasets')
def get_loaders(data_gatherer_name='supervised_segmentation_target_matcher', data_gatherer_kwargs=None,
                train_test_split_function='sklearn_train_test_split', random_state=None, train_test_split_kwargs=None,
                aug_name='none_aug',
                dataset_function_name='get_TVSD_datasets', dataset_kwargs=None,
                dataset_rebalance_function_name=None, dataset_rebalance_kwargs=None,
                collate_fn_name=None,
                sampler_function_name=None, sampler_function_kwargs=None,
                dataloader_kwargs=None):       
    
    # gather data
    data_gatherer_kwargs = data_gatherer_kwargs or {}
    gathered_data = getattr(src.datasets, data_gatherer_name)(**data_gatherer_kwargs)

    # create train-test split, limiting the overall loading capacity
    data_splitter_kwargs = {'random_state': random_state}
    data_splitter_kwargs.update(train_test_split_kwargs or {})
    train_data, test_data = getattr(src.datasets, train_test_split_function)(gathered_data, **data_splitter_kwargs)

    # define augmentation, since it should be passed to the dataset
    aug = getattr(src.augmentations, aug_name)

    # construct train and test datasets itself
    dataset_kwargs = dataset_kwargs or {}
    train_set = getattr(src.datasets, dataset_function_name)(train_data, aug=aug, **dataset_kwargs)
    test_set = getattr(src.datasets, dataset_function_name)(test_data, aug=aug, **dataset_kwargs)

    # rebalance dataset
    if dataset_rebalance_function_name is not None:
        dataset_rebalance_kwargs = dataset_rebalance_kwargs or {}
        train_set, test_set = getattr(src.datasets, dataset_rebalance_function_name)([train_set, test_set], **dataset_rebalance_kwargs)
    
    # select collate function
    collate_fn = getattr(src.datasets, collate_fn_name) if collate_fn_name is not None else None

    # get samplers
    if sampler_function_name is not None:
        train_sampler = getattr(src.datasets, sampler_function_name)(train_set, **sampler_function_kwargs)
        train_loader_kw = {'batch_sampler': train_sampler}
        test_sampler = getattr(src.datasets, sampler_function_name)(test_set, **sampler_function_kwargs)
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

    return train_loader, test_loader

@cfg_ut('optimizer', 'preparing optimizer')
def get_optimizer(model, lr=3e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()

    return optimizer

@cfg_ut('criterion', 'getting criterion')
def get_criterion(criterion_name, criterion_hyper_parameters=None):
    criterion_hyper_parameters = criterion_hyper_parameters or {}
    criterion = getattr(src.losses, criterion_name)(**criterion_hyper_parameters)
    return criterion

@cfg_ut('logger', 'initializing logger')
def get_logger(project_name, log_cfg, experiment_name=None):
    if project_name is not None:
        wanlogger = WandbLogger(metric_names=None, project=project_name, name=experiment_name, config=log_cfg)
        return wanlogger

@cfg_ut('callbacks', 'getting callbacks', force=True)
def get_callbacks(logger, *args):
    callbacks = []
    if logger is not None:
        callbacks.append(logger)
    
    for callbac_config in args:
        kwgs = (callbac_config['kwargs'] or {})if 'kwargs' in  callbac_config.keys() else {}
        new_callback = getattr(src.callbacks, callbac_config['name'])(**kwgs)

        if isinstance(new_callback, list):
            callbacks += new_callback
        else:
            callbacks.append(new_callback)
    
    return callbacks

@cfg_ut('runner', 'initializing runner', force=True)
def get_runner(runner_name='SupervisedRunner', runner_kwargs=None):
    runner_kwargs = runner_kwargs or {}
    return getattr(src.runners, runner_name)(**runner_kwargs)

@cfg_ut('training', 'starting overall training', force=True)
def do_training(runner, model, criterion, optimizer, train_loader, test_loader, callbacks, **kwargs):
    torch.backends.cudnn.benchmark = True

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders={"train": train_loader, "valid": test_loader},
        callbacks=callbacks,
        **kwargs)

@hydra.main(config_path='training_configs', config_name="config")
def overall_training(cfg : DictConfig) -> None:
    excluded = check_exclusion(cfg=cfg)
    if excluded:
        print("this job is marked to be excluded!")
        print(OmegaConf.to_yaml(cfg['model']))
        print(OmegaConf.to_yaml(cfg['training']))
    else:
        model = get_model(cfg=cfg)
        load_weights(model, cfg=cfg)
        train_loader, test_loader = get_loaders(cfg=cfg)
        optimizer = get_optimizer(model, cfg=cfg)
        criterion = get_criterion(cfg=cfg)
        logger = get_logger(cfg=cfg, log_cfg=cfg)
        callbacks = get_callbacks(logger, cfg=cfg)
        runner = get_runner(cfg=cfg)
        do_training(runner, model, criterion, optimizer, train_loader, test_loader, callbacks, cfg=cfg)

if __name__ == "__main__":
    overall_training()

# imports
import torch
from torch.utils.data import DataLoader

import torch

from catalyst.loggers.wandb import WandbLogger

import hydra
from omegaconf import DictConfig, OmegaConf
from src.hydra_helpers import cfg_ut, check_exclusion

import src.augmentations
import src.models
import src.loaders
import src.losses
import src.callbacks
import src.runners



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
def get_loaders(loaders_function='generic_loaders', **kwargs):
    return getattr(src.loaders, loaders_function)(**kwargs)

@cfg_ut('optimizer', 'preparing optimizer')
def get_optimizer(model, optimizer_name, optimizer_hyper_parameters=None, 
                  submodel_to_optimize=None, parameters_to_include=None, parameters_to_exclude=None):
    optimizer_hyper_parameters = optimizer_hyper_parameters or {}

    if submodel_to_optimize is not None:
        if isinstance(submodel_to_optimize, str):
            model = model[submodel_to_optimize]
    
    if (parameters_to_include is None) and (parameters_to_exclude is None):
        params = model.parameters()
    else:
        params = model.named_parameters()
        if parameters_to_include is not None:
            params = {n:p for n,p in params if n.startswith(tuple(parameters_to_include))}
        if parameters_to_exclude is not None:
            params = {n:p for n,p in params if not n.startswith(tuple(parameters_to_exclude))}
        params = params.values()

    optimizer = getattr(torch.optim, optimizer_name)(params, **optimizer_hyper_parameters)
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
        wanlogger = WandbLogger(project=project_name, name=experiment_name, config=log_cfg)
        return {'wandb': wanlogger}
    else:
        return None

@cfg_ut('callbacks', 'getting callbacks', force=True)
def get_callbacks(*args):
    callbacks = [] 
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
    if runner_name == 'SupervisedRunner':
        kwgs = {'input_key':"features", 'output_key': "logits", 'target_key': "targets", 'loss_key': "loss"}
    else:
        kwgs = {}
    runner_kwargs = runner_kwargs or {}
    kwgs.update(runner_kwargs)
    
    return getattr(src.runners, runner_name)(**kwgs)

@cfg_ut('training', 'starting overall training', force=True)
def do_training(runner, model, criterion, optimizer, loaders, logger, callbacks, num_steps=None, **kwargs):
    torch.backends.cudnn.benchmark = True

    train_stage_loaders, inference_stage_loaders = loaders

    if (num_steps is not None) and ('num_epochs' in kwargs.keys()) and (kwargs['num_epochs'] is not None):
        raise Exception('both `num_epochs` and `num_steps` should not be configured together!')
    
    if num_steps is not None:
        kwargs['num_epochs'] = num_steps // len(train_stage_loaders['train'])

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=train_stage_loaders,
        callbacks=callbacks,
        loggers=logger,
        **kwargs)
    
    if inference_stage_loaders is not None:
        assert len([i for i in inference_stage_loaders.keys() if i.startswith('train')]) == 0, 'There should be no train datasets on the last evaluation step!'
        
        if 'valid' not in inference_stage_loaders.keys():
            inference_stage_loaders['valid'] = train_stage_loaders['valid']

        kwargs['num_epochs'] = 1
        kwargs['verbose'] = True
        runner.train(
            model=model,
            criterion=criterion,
            loaders=inference_stage_loaders,
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
        loaders = get_loaders(cfg=cfg)
        optimizer = get_optimizer(model, cfg=cfg)
        criterion = get_criterion(cfg=cfg)
        logger = get_logger(cfg=cfg, log_cfg=cfg)
        callbacks = get_callbacks(cfg=cfg)
        runner = get_runner(cfg=cfg)
        do_training(runner, model, criterion, optimizer, loaders, logger, callbacks, cfg=cfg)

if __name__ == "__main__":
    overall_training()

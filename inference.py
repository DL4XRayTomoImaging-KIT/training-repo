from builtins import breakpoint, getattr
from numpy.lib.nanfunctions import nanpercentile
import torch
from torch import nn
from train import get_model, load_weights
import numpy as np
from functools import partial
from src.augmentations import none_aug
from flexpand import Expander, Matcher
import os
import tifffile

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
from scipy.stats import mode

import src.datasets
from torch.utils.data import DataLoader
import src.augmentations

from univread import read as imread


# what to configure:
# network loading config (this will share loading function with train)
# configuration for the flexpand (this could be loaded as a separate configuration)
# processing parameters (batch size, activation function)
# configuration for the postfix of file to save (what should we do if none provided, maybe required?)

def get_processing_dataloader(data, dset_name, dset_kwargs, batch_size):
    dc = getattr(src.datasets, dset_name)
    dset_kwargs = OmegaConf.to_container(dset_kwargs, resolve=True)
    if 'aug' in dset_kwargs.keys():
        dset_kwargs['aug'] = getattr(src.augmentations, dset_kwargs['aug'])
    if 'augmentation' in dset_kwargs.keys():
        dset_kwargs['augmentation'] = getattr(src.augmentations, dset_kwargs['augmentation'])
    dset = dc(data, **dset_kwargs)
    dldr = DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    return dldr

def no_agg(batch, callback):
    return callback(batch)

def noisy_agg(batch, callback, samples_count, noise_multiplier=0, agg_func='mean'):
    local_preds = []
    for i in range(samples_count):
        batch_copy = batch + torch.FloatTensor(np.random.randn(*batch.detach().cpu().numpy().shape)*noise_multiplier)
        local_preds.append(callback(batch_copy))
    agg_func = getattr(np, agg_func)
    return agg_func(np.stack(local_preds), 0)


def smart_selector_agg(batch, callback, samples_count=50, noise_multiplier=0.2, agg_func='mean', std_multiplier=2):
    img_to_new_s = []
    for i in range(samples_count):
        batch_copy = batch + torch.FloatTensor(np.random.randn(*batch.detach().cpu().numpy().shape)*noise_multiplier)
        img_to_new_s.append(callback(batch_copy))

    img_to_new_s = np.array(img_to_new_s)
    agg_func = getattr(np, agg_func)
    
    img_to_new_n = callback(batch)
    
    exchange_map = (img_to_new_n < (img_to_new_s.mean(0) - img_to_new_s.std(0)*std_multiplier)) | (img_to_new_n > (img_to_new_s.mean(0) + img_to_new_s.std(0)*std_multiplier))
    
    img_to_new_n[exchange_map] = np.median(img_to_new_s, 0)[exchange_map]
    return img_to_new_n

class Segmenter:
    def __init__(self, model_config, checkpoint_config, processing_parameters, dataset_parameters):
        self.model = get_model(**model_config)
        load_weights(self.model, **checkpoint_config)
        self.model = nn.DataParallel(self.model)
        self.model.to(torch.device('cuda:0'))
        self.model.train(False)

        self.batch_size = processing_parameters['batch_size']

        self.dataset_parameters = dataset_parameters

        activation_function_name = processing_parameters.get('activation_function', 'argmax') or 'argmax'
        self.activation_function = partial(getattr(torch, activation_function_name), dim=1)
        self.image_grain = processing_parameters.get('image_grain', 32) or 32
        self.mode_3d = processing_parameters.get('mode_3d', False) or False

        aggregation_function_name = dataset_parameters.get('agg_func', 'no_agg') or 'no_agg'
        self.agg_func = partial(globals()[aggregation_function_name], **(dataset_parameters.get('agg_func_kwargs', {}) or {}))
        
        self.output_dtype = processing_parameters.get('dtype')

    def process_one_batch(self, batch):
        with torch.no_grad():
            # batch = none_aug(image=batch)['image']
            if batch.ndim < 4:
                batch = batch[:, None, ...] # add channels dimension
            # else:
            #     batch = np.moveaxis(batch, -1, 1) # move channels dimension
            batch = batch[:, :, :batch.shape[2]//self.image_grain*self.image_grain, :batch.shape[3]//self.image_grain*self.image_grain]
            # batch = torch.from_numpy(batch)
            batch = batch.to(torch.device('cuda:0'))
            pred = self.activation_function(self.model(batch))
            pred = pred.detach().cpu().numpy()
            return pred

    def process_one_axis(self, volume, axis=0):
        predictions = []

        if axis != 0:
            volume = np.moveaxis(volume, axis, 0)
        
        dim_1 = int(np.ceil(volume.shape[-2]/self.image_grain)*self.image_grain)
        dim_2 = int(np.ceil(volume.shape[-1]/self.image_grain)*self.image_grain)

        tmp_volume = np.pad(volume, ((0, 0), (0, dim_1-volume.shape[-2]), (0, dim_2-volume.shape[-1])), mode='reflect')
        
        dldr = get_processing_dataloader(tmp_volume, self.dataset_parameters['name'], self.dataset_parameters['kwargs'], self.batch_size)

        for b in dldr:
            predictions.append(self.agg_func(b[0], self.process_one_batch))
        predictions = np.concatenate(predictions)


        predictions = predictions[..., :volume.shape[-2], :volume.shape[-1]]
        
        if axis != 0:
            predictions = np.moveaxis(predictions, 0, axis)
        
        if self.output_dtype is not None:
            predictions = predictions.astype(self.output_dtype)

        return predictions

    def process_one_volume(self, volume):
        if self.mode_3d:
            predictions = np.zeros_like(volume, dtype=self.output_dtype)
            for axis in [0, 1, 2]:
                predictions += self.process_one_axis(volume, axis)
                print('axis processed ', axis)
            predictions = predictions//3
            print('mean estimated')
        else:
            predictions = self.process_one_axis(volume)

        return predictions

def generate_in_out_pairs(expander_config, saving_config):
    exp = Expander()
    in_list = exp(**expander_config)

    mtch = Matcher()
    pairs_list = mtch(in_list, **saving_config)

    return pairs_list

def load_process_save(segmenter, input_addr, output_addr):
    img = imread(input_addr)
    msk = segmenter.process_one_volume(img)
    if not output_addr.endswith(('.tif', '.tiff')):
        output_addr += '.tif'
    tifffile.imwrite(output_addr, msk)

@hydra.main(config_path='inference_configs', config_name="config")
def inference(cfg : DictConfig) -> None:
    seger = Segmenter(cfg['model'], cfg['checkpoint'], cfg['processing'], cfg['dataset']['dataset_config'])
    pairs = generate_in_out_pairs(cfg['dataset']['source'], cfg['dataset']['destination'])
    for inp_addr, outp_addr in tqdm(pairs):
        load_process_save(seger, inp_addr, outp_addr)

if __name__ == "__main__":
    inference()

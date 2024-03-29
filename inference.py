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
from omegaconf import DictConfig
from tqdm.auto import tqdm
from scipy.stats import mode

# what to configure:
# network loading config (this will share loading function with train)
# configuration for the flexpand (this could be loaded as a separate configuration)
# processing parameters (batch size, activation function)
# configuration for the postfix of file to save (what should we do if none provided, maybe required?)

class Segmenter:
    def __init__(self, model_config, checkpoint_config, processing_parameters):
        self.model = get_model(**model_config)
        load_weights(self.model, **checkpoint_config)
        self.model = nn.DataParallel(self.model)
        self.model.to(torch.device('cuda:0'))
        self.model.train(False)

        self.batch_size = processing_parameters['batch_size']

        activation_function_name = processing_parameters.get('activation_function', 'argmax') or 'argmax'
        self.activation_function = partial(getattr(torch, activation_function_name), dim=1)
        self.image_grain = processing_parameters.get('image_grain', 32) or 32
        self.mode_3d = processing_parameters.get('mode_3d', False) or False
        self.output_dtype = processing_parameters.get('dtype')

    def process_one_batch(self, batch):
        with torch.no_grad():
            batch = none_aug(image=batch)['image']
            if batch.ndim < 4:
                batch = batch[:, None, ...] # add channels dimension
            else:
                batch = np.moveaxis(batch, -1, 1) # move channels dimension
            batch = batch[:, :, :batch.shape[2]//self.image_grain*self.image_grain, :batch.shape[3]//self.image_grain*self.image_grain]
            batch = torch.from_numpy(batch)
            batch = batch.to(torch.device('cuda:0'))
            pred = self.activation_function(self.model(batch))
            pred = pred.detach().cpu().numpy()
            return pred

    def process_one_axis(self, volume, axis=0):
        predictions = []

        if axis != 0:
            volume = np.moveaxis(volume, axis, 0)
        
        for b in range(int(np.ceil(len(volume)/ self.batch_size))):
            batch = volume[self.batch_size*b : self.batch_size*(b+1)]
            pred = self.process_one_batch(batch)
            predictions.append(pred)
        predictions = np.concatenate(predictions)
        if predictions.ndim == 4:
            predictions = np.pad(predictions, ((0, 0), (0, 0), (0, volume.shape[1]-predictions.shape[2]), (0, volume.shape[2]-predictions.shape[3])))
        elif predictions.ndim == 3:
            predictions = np.pad(predictions, ((0, 0), (0, volume.shape[1]-predictions.shape[1]), (0, volume.shape[2]-predictions.shape[2])))
        
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
    img = tifffile.imread(input_addr)
    msk = segmenter.process_one_volume(img)
    tifffile.imwrite(output_addr, msk)

@hydra.main(config_path='inference_configs', config_name="config")
def inference(cfg : DictConfig) -> None:
    seger = Segmenter(cfg['model'], cfg['checkpoint'], cfg['processing'])
    pairs = generate_in_out_pairs(cfg['dataset']['source'], cfg['dataset']['destination'])
    for inp_addr, outp_addr in tqdm(pairs):
        load_process_save(seger, inp_addr, outp_addr)

if __name__ == "__main__":
    inference()

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

# what to configure:
# network loading config (this will share loading function with train)
# configuration for the flexpand (this could be loaded as a separate configuration)
# processing parameters (batch size, activation function)
# configuration for the postfix of file to save (what should we do if none provided, maybe required?)

class Segmenter:
    def __init__(self, model_config, checkpoint_config, processing_parameters):
        self.model = get_model(**model_config)
        load_weights(self.model, **checkpoint_config)
        self.model.train(False)
        self.model = nn.DataParallel(self.model)
        self.model.to(torch.device('cuda:0'))
        self.model.train(False)

        self.batch_size = processing_parameters['batch_size']

        if ('activation_function' in processing_parameters) and (processing_parameters['activation_function'] is not None):
            activation_function_name = processing_parameters['activation_function']
        else:
            activation_function_name = 'argmax'
        self.activation_function = partial(getattr(torch, activation_function_name), dim=1)

        if ('image_grain' in processing_parameters) and (processing_parameters['image_grain'] is not None):
            self.image_grain = processing_parameters['image_grain']
        else:
            self.image_grain = 32
        
        if ('dtype' in processing_parameters):
            self.output_dtype = processing_parameters['dtype']
        else:
            self.output_dtype = None


    def process_one_volume(self, volume):
        predictions = []
        with torch.no_grad():
            for b in range(int(np.ceil(len(volume)/ self.batch_size))):
                batch = volume[self.batch_size*b : self.batch_size*(b+1)]
                batch = batch[:, :batch.shape[1]//self.image_grain*self.image_grain, :batch.shape[2]//self.image_grain*self.image_grain]
                batch = none_aug(image=batch)['image']
                batch = torch.from_numpy(batch[:, None, ...])
                batch = batch.to(torch.device('cuda:0'))
                pred = self.activation_function(self.model(batch))
                pred = pred.detach().cpu().numpy()
                predictions.append(pred)
        predictions = np.concatenate(predictions)
        if predictions.ndim == 4:
            predictions = np.pad(predictions, ((0, 0), (0, 0), (0, volume.shape[1]-predictions.shape[2]), (0, volume.shape[2]-predictions.shape[3])))
        elif predictions.ndim == 3:
            predictions = np.pad(predictions, ((0, 0), (0, volume.shape[1]-predictions.shape[1]), (0, volume.shape[2]-predictions.shape[2])))

        if self.output_dtype is not None:
            predictions = predictions.astype(self.output_dtype)

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

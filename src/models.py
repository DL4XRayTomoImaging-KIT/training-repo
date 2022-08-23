from segmentation_models_pytorch import * # we need everything from the smp

# now we define our own models
import torch
from torch import nn
from torchvision.models import alexnet
from torchvision.models.resnet import conv1x1, conv3x3, BasicBlock
from einops import rearrange

def get_recommended_batch_size(parameters_count, image_side):
    base_batch_size = 1
    base_batch_size *= int((512/image_side)**2) # correction for the crop size
    base_batch_size *= max(torch.cuda.device_count(), 1) # correction for the devices count
    base_batch_size = int(base_batch_size * (40_000_000 / parameters_count)) # correction for the model size

def partial_load(model, state_dict, renew_parameters=None, drop_parameters=None):
    mp = set(model.state_dict().keys())
    dp = set(state_dict.keys())

    if drop_parameters is not None:
        keys_to_pop = []
        for parname in drop_parameters:
            keys_to_pop += [k for k in dp if k.startswith(parname)]
        for key in keys_to_pop:
            state_dict.pop(key)
    dp = set(state_dict.keys())

    not_updated = list(mp-dp)
    if renew_parameters is not None:
        not_updated += renew_parameters
    model_side_addendum = {k:v for k,v in model.state_dict().items() if k in not_updated}
    state_dict.update(model_side_addendum)

    model.load_state_dict(state_dict)

def load_to_smp(model, state_dict, renew_parameters=None, drop_parameters=None):
    if not('fc.weight' in state_dict.keys()):
        state_dict['fc.weight'] = None
    if not('fc.bias' in state_dict.keys()):
        state_dict['fc.bias'] = None

    partial_load(model, state_dict, renew_parameters, drop_parameters)

from segmentation_models_pytorch import * # we need everything from the smp

# now we define our own models
import torch
from torch import nn
from torchvision.models.resnet import conv1x1, conv3x3, BasicBlock
from einops import rearrange

class EUnet(Unet):
    def __init__(self, *args, embedding_depth=512, nonlinear_heads=False, metric_head=False, **kwargs):
        real_classes = kwargs['classes']
        kwargs['classes'] = embedding_depth
        super().__init__(*args, **kwargs)
        if nonlinear_heads:
            self.classification_head = nn.Sequential(nn.Conv2d(in_channels=embedding_depth, out_channels=embedding_depth, kernel_size=1), 
                                                     nn.ReLU(), 
                                                     nn.Conv2d(in_channels=embedding_depth, out_channels=real_classes, kernel_size=1))
        else:
            self.classification_head = nn.Conv2d(in_channels=embedding_depth, out_channels=real_classes, kernel_size=1)
        if metric_head:
            if nonlinear_heads:
                self.metric_head = nn.Sequential(nn.Conv2d(in_channels=embedding_depth, out_channels=2048, kernel_size=1), 
                                                 nn.BatchNorm2d(2048), nn.ReLU(), 
                                                 nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1),
                                                 nn.BatchNorm2d(2048), nn.ReLU(),
                                                 nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1))
            else:
                self.metric_head = nn.Conv2d(in_channels=embedding_depth, out_channels=embedding_depth, kernel_size=1)
        else:
            self.metric_head = lambda x: x
    
    def forward(self, *args, **kwargs):
        e, b_ = super().forward(*args, **kwargs)
        c = self.classification_head(e)
        m = self.metric_head(e)
        
        return m, c

def get_recommended_batch_size(parameters_count, image_side):
    base_batch_size = 1
    base_batch_size *= int((512/image_side)**2) # correction for the crop size
    base_batch_size *= max(torch.cuda.device_count(), 1) # correction for the devices count
    base_batch_size = int(base_batch_size * (40_000_000 / parameters_count)) # correction for the model size

def partial_load(model, state_dict, renew_parameters=None):
    mp = set(model.state_dict().keys())
    dp = set(state_dict.keys())

    not_updated = list(mp-dp)
    if renew_parameters is not None:
        not_updated += renew_parameters

    model_side_addendum = {k:v for k,v in model.state_dict().items() if k in not_updated}
    state_dict.update(model_side_addendum)
    model.load_state_dict(state_dict)

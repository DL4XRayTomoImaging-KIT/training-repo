from segmentation_models_pytorch import * # we need everything from the smp

# now we define our own models
import torch
from torch import nn
from torchvision.models.resnet import conv1x1, conv3x3, BasicBlock
from einops import rearrange

class MyPersonalResnet(nn.Module):
    def __init__(self, layers=[2,2,2,2], block=BasicBlock, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, strides=None, in_channels=3, normalised_output=True, 
                 class_conversion=True, space_collapse=True, prelayers=None):
        super().__init__()

        self.prep = prelayers

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.out_channels = num_classes
        self.class_conversion = class_conversion
        self.space_collapse = space_collapse

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        if strides is None:
            strides = [2, 1, 2, 2, 2]
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=strides[0], padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[1])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[2],
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[3],
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[4],
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if normalised_output:
            self.fc = nn.Sequential(nn.Linear(512 * block.expansion, num_classes), nn.BatchNorm1d(num_classes))
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, collapse_space=None, convert_classes=None):
        x = self.class_activation_map(x, class_conversion=convert_classes if convert_classes is not None else self.class_conversion)
        if collapse_space is None:
            collapse_space = self.space_collapse
        if collapse_space:
            x = self.avgpool(x)[:,:,0,0]
        return x

    def class_activation_map(self, x, class_conversion=True):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if class_conversion:
            x_ = rearrange(x, 'b c h w -> (b h w) c')
            x_ = self.fc(x_)
            x = rearrange(x_, '(b h w) c -> b c h w', b=x.shape[0], h=x.shape[2], w=x.shape[3])      
        return x

    def get_model_raw_featuremap(self, x, include_layers_count=0):
        self.conv1.stride = (1, 1)
        x = self.conv1(x)
        self.conv1.stride = (2, 2)
        x = self.bn1(x)

        if include_layers_count > 0:
            x = self.relu(x)
            x = self.maxpool(x)

            layers_list = [self.layer1, self.layer2, self.layer3, self.layer4]
            for i, l in zip(range(include_layers_count), layers_list):
                x = l(x)
        return x

# special case -- doing MaskRCNN loader as a function.

from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN

def mask_rcnn(encoder_name, encoder_weights, in_channels, classes):
    backbone = MyPersonalResnet(in_channels=in_channels, num_classes=512, normalised_output=False, space_collapse=False)
    if encoder_weights is not None:
        backbone.load_state_dict(torch.load(encoder_weights)['model_state_dict'])
    backbone = BackboneWithFPN(backbone, {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}, [64, 128, 256, 512], 512)
    model = MaskRCNN(backbone, num_classes=classes, image_mean=[0.5], image_std=[0.15]) #TODO: remove hardcoding of mean and std

    return model


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

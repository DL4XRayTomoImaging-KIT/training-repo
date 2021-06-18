from batchgenerators.transforms import Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, \
    BrightnessTransform, ContrastAugmentationTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform, SpatialTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.abstract_transforms import AbstractTransform
import math
import numpy as np
from skimage.transform import resize
from albumentations.augmentations.transforms import CropNonEmptyMaskIfExists, ImageOnlyTransform,\
    HorizontalFlip, VerticalFlip, GaussianBlur, GaussNoise, RandomBrightnessContrast, RandomGamma, \
    ToFloat, Downscale, Normalize
from albumentations import Compose as AlbuCompose
from copy import deepcopy


class NormalizeB(ImageOnlyTransform):
    def __init__(self, shift=0., scale=255., data_key='image', label_key='seg'):
        super().__init__()
        self.shift = shift
        self.scale = scale
        self.data_key = data_key
        self.label_key = label_key

    def __call__(self, **data_dict):
        for k in [self.data_key]:
            image = data_dict[k]
            image -= self.shift
            data_dict[k] = image / self.scale
        return data_dict


class ToFloatB(ImageOnlyTransform):
    def __init__(self, data_key='image', label_key='seg'):
        super().__init__()
        self.data_key = data_key
        self.label_key = label_key

    def __call__(self, **data_dict):
        for k in [self.data_key]:
            image = data_dict[k].astype(np.float32)
            data_dict[k] = image
        return data_dict


class ScaleDown(AbstractTransform):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, data, data_key, label_key, crop_d, crop_h):
        if np.random.rand(1) <= self.p:
            scale = np.random.uniform(0.6, 1, 1)
            for k in [data_key, label_key]:
                c, rows, cols = data[k].shape
                add_rows, add_cols = math.ceil(rows * (1 - scale)) // 2, math.ceil(cols * (1 - scale)) // 2
                new_img = np.zeros((c, rows + 2 * add_rows, cols + 2 * add_cols))
                new_img[:, add_rows: -add_rows, add_cols: -add_cols] = data[k]
                data[k] = resize(new_img, (c, rows, cols))
        return data


crop_in_mask = AlbuCompose([CropNonEmptyMaskIfExists(280, 149)])

strong_aug_albu = AlbuCompose([CropNonEmptyMaskIfExists(280, 149),
                              HorizontalFlip(p=0.1),
                              VerticalFlip(p=0.1),
                              GaussNoise(var_limit=(0, 0.1), p=0.1),
                              GaussianBlur(sigma_limit=(0.5, 1.), p=0.2),
                              RandomBrightnessContrast(p=0.15),
                              Downscale(scale_min=0.5, scale_max=0.99, interpolation=0, p=0.25),
                              RandomGamma(p=0.15),
                              Normalize(mean=np.array([0.3922]), std=np.array([0.2753]), always_apply=True),
                              ToFloat(always_apply=True)])

strong_aug = Compose([MirrorTransform(data_key="image", label_key='mask'),
                      SpatialTransform(None, patch_center_dist_from_border=100,
                                       data_key="image", label_key="mask",
                                       do_elastic_deform=False, do_rotation=False,
                                       do_scale=True, scale=(0.65, 1.), random_crop=False),
                      GaussianNoiseTransform(p_per_sample=0.1, data_key="image"),
                      GaussianBlurTransform(blur_sigma=(0.5, 1.), different_sigma_per_channel=True,
                                            p_per_channel=0.5, p_per_sample=0.2, data_key="image"),
                      BrightnessMultiplicativeTransform((0.75, 1.25), p_per_sample=0.15, data_key="image"),
                      BrightnessTransform(0.0, 0.1, True, p_per_sample=0.15, p_per_channel=0.5, data_key="image"),
                      ContrastAugmentationTransform(p_per_sample=0.15, data_key="image"),
                      SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5,
                                                     order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                     ignore_axes=None, data_key="image"),
                      GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True, retain_stats=True,
                                     p_per_sample=0.15, data_key="image"),
                      NormalizeB(data_key="image", label_key="mask")
                      ])

spatial_aug = Compose([SpatialTransform(None, patch_center_dist_from_border=100,
                                        data_key="image", label_key="mask",
                                        do_elastic_deform=False, do_rotation=False,
                                        do_scale=True, scale=(0.65, 1.), random_crop=False)])

test_albu = AlbuCompose([Normalize(mean=np.array([0.3922]), std=np.array([0.2753]), always_apply=True)])

test = Compose([ToFloatB(), NormalizeB(data_key="image", label_key="mask")])
import numpy as np
from albumentations.augmentations.transforms import CropNonEmptyMaskIfExists, \
    HorizontalFlip, VerticalFlip, GaussianBlur, GaussNoise, RandomBrightnessContrast, RandomGamma, \
    ToFloat, Downscale, Normalize
from albumentations import Compose as AlbuCompose


crop_in_mask = AlbuCompose([CropNonEmptyMaskIfExists(280, 149)])

strong_aug = AlbuCompose([HorizontalFlip(p=0.1),
                          VerticalFlip(p=0.1),
                          GaussNoise(var_limit=(0, 0.1), p=0.1),
                          GaussianBlur(sigma_limit=(0.5, 1.), p=0.2),
                          RandomBrightnessContrast(p=0.15),
                          Downscale(scale_min=0.5, scale_max=0.99, interpolation=0, p=0.25),
                          RandomGamma(p=0.15),
                          Normalize(mean=np.array([0.3922]), std=np.array([0.2753]), always_apply=True),
                          ToFloat(always_apply=True)])

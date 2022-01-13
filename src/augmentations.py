import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import albumentations.augmentations.transforms as A
from albumentations import Compose, OneOf, KeypointParams


strong_aug = Compose([OneOf([A.Blur(blur_limit=15),
                             A.MedianBlur(blur_limit=15),
                             A.MotionBlur(blur_limit=15)], p=0.5),
                      OneOf([A.CLAHE(),
                             A.Equalize()], p=0.2),
                      OneOf([A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4),
                             A.RandomGamma(gamma_limit=(30, 170)),
                             A.Solarize(threshold=(128-64, 128+64))], p=0.5),
                      OneOf([A.GridDistortion(distort_limit=0.5),
                             A.GlassBlur(max_delta=10)], p=0.5),
                      A.GaussNoise(var_limit=(50, 100), p=0.5),
                      A.ToFloat(always_apply=True)], p=0.99)

medium_aug = Compose([OneOf([A.Blur(blur_limit=5),
                             A.MedianBlur(blur_limit=5),
                             A.MotionBlur(blur_limit=5)], p=0.5),
                      OneOf([A.CLAHE(),
                             A.Equalize()], p=0.2),
                      OneOf([A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                             A.RandomGamma(gamma_limit=(90, 110))], p=0.5),
                      OneOf([A.GridDistortion(distort_limit=0.2),
                             A.GlassBlur(max_delta=5)], p=0.5),
                      A.GaussNoise(var_limit=(20, 50), p=0.5),
                      A.ToFloat(always_apply=True)], p=0.99)

light_aug = Compose([ OneOf([A.Blur(blur_limit=3),
                             A.MedianBlur(blur_limit=3),
                             A.MotionBlur(blur_limit=3)], p=0.5),
                      OneOf([A.CLAHE(),
                             A.Equalize()], p=0.2),
                      OneOf([A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                             A.RandomGamma(gamma_limit=(90, 110))], p=0.5),
                      A.GaussNoise(var_limit=(20, 50), p=0.5),
                      A.ToFloat(always_apply=True)], p=0.99)

none_aug = A.ToFloat(always_apply=True)

medium_aug_rot = Compose([OneOf([A.Blur(blur_limit=5),
                             A.MedianBlur(blur_limit=5),
                             A.MotionBlur(blur_limit=5)], p=0.5),
                      OneOf([A.CLAHE(),
                             A.Equalize()], p=0.2),
                      OneOf([A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                             A.RandomGamma(gamma_limit=(90, 110))], p=0.5),
                      OneOf([A.GridDistortion(distort_limit=0.2),
                             A.ShiftScaleRotate(),
                             A.GlassBlur(max_delta=5)], p=0.5),
                      A.GaussNoise(var_limit=(20, 50), p=0.5),
                      A.ToFloat(always_apply=True)], p=0.99)

medium_aug_bbx = Compose([   OneOf([A.Blur(blur_limit=5),
                                     A.MedianBlur(blur_limit=5),
                                     A.MotionBlur(blur_limit=5)], p=0.8),
                              OneOf([A.CLAHE(),
                                     A.Equalize()], p=0.2),
                              OneOf([A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                                     A.RandomGamma(gamma_limit=(90, 110))], p=0.8),
                              OneOf([A.ShiftScaleRotate(),
                                     A.GlassBlur(max_delta=5)], p=0.8),
                              A.GaussNoise(var_limit=(20, 50), p=0.8),
                              A.ToFloat(always_apply=True)],
                           keypoint_params=KeypointParams(format='xy', label_fields=['position'], remove_invisible=True, angle_in_degrees=True))

low_aug_bbx = Compose([A.ShiftScaleRotate(),
                       A.ToFloat(always_apply=True)],
                       keypoint_params=KeypointParams(format='xy', label_fields=['position'], remove_invisible=True, angle_in_degrees=True))

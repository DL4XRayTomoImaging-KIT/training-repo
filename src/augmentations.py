from turtle import position
from audioop import alaw2lin
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

medium_aug_rot_float = Compose([
                        OneOf([A.Blur(blur_limit=5),
                               A.MedianBlur(blur_limit=5),
                               A.MotionBlur(blur_limit=5)], p=0.5),
                      OneOf([A.GridDistortion(distort_limit=0.2),
                             A.ShiftScaleRotate()], p=0.5),
                      A.GaussNoise(var_limit=(0.05, 0.2), p=0.5)], p=0.99)


aug_rot_float = Compose([
                      OneOf([A.GridDistortion(distort_limit=0.2),
                             A.ShiftScaleRotate()], p=0.5)], p=0.99)

medium_aug_rot_float_atro = Compose([
                        A.RandomCrop(256, 256, always_apply=True), 
                        OneOf([A.Blur(blur_limit=5),
                               A.MedianBlur(blur_limit=5),
                               A.MotionBlur(blur_limit=5)], p=0.5),
                      OneOf([A.GridDistortion(distort_limit=0.2),
                             A.ShiftScaleRotate()], p=0.5),
                      A.GaussNoise(var_limit=(0.05, 0.2), p=0.5)], p=0.99, additional_targets={'image_1': 'image'})

medium_aug_rot_float_atro_reco = Compose([
                        A.RandomCrop(256, 256, always_apply=True), 
                        OneOf([A.Blur(blur_limit=5),
                               A.MedianBlur(blur_limit=5),
                               A.MotionBlur(blur_limit=5)], p=0.5),
                            OneOf([A.GridDistortion(distort_limit=0.2),
                                   A.ShiftScaleRotate()], p=0.5)], p=0.99, additional_targets={'image_1': 'image'})

medium_aug_rot_float_atro_reco = Compose([
                        A.RandomCrop(256, 256, always_apply=True), 
                        OneOf([A.Blur(blur_limit=5),
                               A.MedianBlur(blur_limit=5),
                               A.MotionBlur(blur_limit=5)], p=0.5),
                            OneOf([A.GridDistortion(distort_limit=0.2),
                                   A.ShiftScaleRotate()], p=0.5)], p=0.99, additional_targets={'image_1': 'image'})
                     
infer_aug_float_atro = Compose([
                        A.PadIfNeeded(512, 512, always_apply=True),
                        A.RandomCrop(512, 512, always_apply=True)], p=0.99, additional_targets={'image_1': 'image'})

import os
import re
import torch
from glob import glob
from tqdm.auto import tqdm
from src.callbacks import iou
os.chdir("/home/ws/er5241/Repos/training-repo/")

from inference import *

src_fls = glob('/autofs/HD-LSDF/sd20d002/segmentations/workshop/medaka_decropped/*')

#classes = 10
#dst_addr = '/autofs/HD-LSDF/sd20d002/segmentations/workshop/brain_decropped/*'
#checkpoint_addr = '/mnt/data/machine-learning/logs/medaka-supervised/2021-09-08_23-31-03/logdir/checkpoints/best.pth'
#converter = None

classes = 7
dst_addr = '/autofs/HD-LSDF/sd20d002/segmentations/workshop/heartkidney_decropped/*'
checkpoint_addr = '/mnt/data/machine-learning/logs/medaka-supervised/2021-09-10_15-08-52/logdir/checkpoints/best.pth'
converter = {'0':0, '4':1, '5':2, '6':3, '7':4, '9':5, '10':6}


model = {'network_type': 'Unet', 'encoder_name': 'resnet18', 'classes': classes, 'in_channels': 1}
checkpoint = {'checkpoint_addr': checkpoint_addr, 'checkpoint_key': 'model_state_dict'}
processing = {'batch_size': 1, 'dtype': 'uint8'}
seger = Segmenter(model_config=model, checkpoint_config=checkpoint, processing_parameters=processing)


for src_fl in src_fls:
    id = re.findall(r'/([\d]*).tif', src_fl)[0]
    dst_fls = glob(f'{dst_addr}{id}.tif')
    
    if not dst_fls:
        #print(f'skipped {src_fl}')
        continue
    
    dst_fl = dst_fls[0]
    
    img = tifffile.imread(src_fl)
    predn = seger.process_one_volume(img)
    
    lbl = tifffile.imread(dst_fl)
    if converter is not None:
        for k,v in converter.items():
            lbl[lbl==int(k)] = v
    
    print(id, iou(predn,lbl,classes))

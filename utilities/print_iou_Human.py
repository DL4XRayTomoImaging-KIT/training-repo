import os
import re
import torch
from glob import glob
from src.callbacks import iou
os.chdir("/home/ws/er5241/Repos/training-repo/")

from inference import *

src_fls = glob('/autofs/HD-LSDF/sd20d002/segmentations/workshop/medaka_decropped/*')

#classes = 10
#dst_addr = '/autofs/HD-LSDF/sd20d002/segmentations/workshop/brain_decropped/*'
#converter = None

classes = 7
dst_addr = '/autofs/HD-LSDF/sd20d002/segmentations/workshop/heartkidney_decropped/*'
converter = {'0':0, '4':1, '5':2, '6':3, '7':4, '9':5, '10':6}

for src_fl in src_fls:
    id = re.findall(r'/(\d*).tif', src_fl)[0]
    dst_fls = glob(f'{dst_addr}{id}.tif')
    
    if len(dst_fls) < 2:
        #print(f'skipped {src_fl}')
        continue
    
    dst_fl = dst_fls[0]
    
    img = dst_fls[1]
    predn = tifffile.imread(img)
    lbl = tifffile.imread(dst_fl)
    
    if converter is not None:
        for k,v in converter.items():
            lbl[lbl==int(k)] = v
            predn[predn==int(k)] = v
    
    print(id, iou(predn,lbl,classes))

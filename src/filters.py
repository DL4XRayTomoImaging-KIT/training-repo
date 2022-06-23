import os
import json

def threshold_filter(label_addr, threshold, segm_data_path = '/home/ws/er5241/Repos/training-repo/utilities/segm_data.json'):
    with open(segm_data_path, 'r') as fp:
      segm_data = json.load(fp)
    segm_sum = segm_data[label_addr]['sum']
    return True if segm_sum >= threshold else False
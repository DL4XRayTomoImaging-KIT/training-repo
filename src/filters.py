import os
import json


# function to filter addrs and only keep masks having sum greater than a certain threshold
def volume_sum_filter(gathered_data, threshold=1e6, segm_data_path="/home/ws/er5241/Repos/training-repo/utilities/segm_data.json"):
    with open(segm_data_path, 'r') as fp:
        segm_data = json.load(fp)
      
    filtered_data = []
    for img_addr, msk_addr in gathered_data:
        if segm_data[msk_addr]['sum'] >= threshold:
            filtered_data.append((img_addr, msk_addr))
    return filtered_data
    
# function to filter slices and only keep those whose loss was less than a certain threshold when passed to autoencoder
def slice_loss_filter(dataset, threshold=3):
    loss_data_path="/home/ws/er5241/Repos/training-repo/utilities/autoenc_losses.json"
    with open(loss_data_path, 'r') as fp:
        loss_data = json.load(fp)

    slices_loss = loss_data[dataset.volume.file_addr]
    keep_slices = [ True if slice_loss < threshold else False for slice_loss in slices_loss ]
    exclude_slices = [not slc for slc in keep_slices]

    return exclude_slices
    
# function to filter slices and only keep those which the classifier categorized as good
classifi_result_path="/home/ws/er5241/Repos/training-repo/utilities/classifier_result.json"
with open(classifi_result_path, 'r') as fp:
    classifi_result = json.load(fp)
import numpy as np
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def classifier_filter(msk_addr, threshold=0.5):
    # keep all slices of original markup
    if not msk_addr in classifi_result:
        return None
    else:
        slice_preds = np.array(list(classifi_result[msk_addr].values()))
        slice_preds = sigmoid(slice_preds)
        keep_slices = []
        for slice_pred in slice_preds:
            # slice_pred[0] -> good
            # slice_pred[1] -> bad
            if slice_pred[0] > threshold:
                keep_slices.append(True)
            else:
                keep_slices.append(False)
        #keep_slices = list(classifi_result[msk_addr].values()) 
        exclude_slices = [not slc for slc in keep_slices]
        return exclude_slices
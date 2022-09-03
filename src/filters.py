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
def slice_loss_filter(dataset, threshold=3, loss_data_path="/home/ws/er5241/Repos/training-repo/utilities/autoenc_losses.json"):
    with open(loss_data_path, 'r') as fp:
      loss_data = json.load(fp)

    slices_loss = loss_data[dataset.volume.file_addr]
    keep_slices = [ True if slice_loss < threshold else False for slice_loss in slices_loss ]
    exclude_slices = [not slc for slc in keep_slices]

    return exclude_slices
    
# function to filter slices and only keep those which the classifier categorized as good
def classifier_filter(dataset, classifi_result_path="/home/ws/er5241/Repos/training-repo/utilities/classifier_result.json"):
    with open(classifi_result_path, 'r') as fp:
        classifi_result = json.load(fp)
        
    msk_addr = dataset.segmentation.file_addr
    # keep all slices of original markup
    if not msk_addr in classifi_result:
        keep_slices = [True] * len(dataset.segmentation)
    else:
        keep_slices = list(classifi_result[msk_addr].values())
      
    exclude_slices = [not slc for slc in keep_slices]
    return exclude_slices
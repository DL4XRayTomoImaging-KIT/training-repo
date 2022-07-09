import os
import json


# function to filter addrs and only keep masks having sum greater than a certain threshold
def volume_sum_filter(gathered_data, threshold=1e6, segm_data_path='/home/ws/er5241/Repos/training-repo/utilities/segm_data.json'):
    with open(segm_data_path, 'r') as fp:
        segm_data = json.load(fp)
      
    filtered_data = []
    for img_addr, msk_addr in gathered_data:
        if segm_data[msk_addr]['sum'] >= threshold:
            filtered_data.append((img_addr, msk_addr))
    return filtered_data
    
# function to filter slices and only keep those whose loss was less than a certain threshold when passed to autoencoder
def slice_loss_filter(dataset, threshold=3, loss_data_path='/home/ws/er5241/Repos/training-repo/utilities/autoenc_losses.json'):
    with open(loss_data_path, 'r') as fp:
        loss_data = json.load(fp)
    
    keep_slices, exclude_slices = [], []
    for d in dataset.datasets:
        slices_loss = loss_data[d.volume.file_addr]
        dataset_keep = [slice_id for slice_id, slice_loss in enumerate(slices_loss) if slice_loss < threshold]
        dataset_exclude = [slice_id for slice_id in range(len(slices_loss)) if slice_id not in dataset_keep]
        
        keep_slices.extend(dataset_keep)
        exclude_slices.extend(dataset_exclude)

    return exclude_slices
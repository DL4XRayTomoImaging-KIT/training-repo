import os
import nibabel as nib
import numpy as np
from medpy.metric import hd95
from skimage.measure import label as LAB

# adapded from https://github.com/jianpengz/DoDNet/blob/c22687bffb36e218f0239f6bbd2c8971088a7a9c/a_DynConv/postp.py


def continuous_region_extract_organ(label, keep_region_nums):  # keep_region_nums=1
    mask = np.zeros_like(label)
    regions = np.where(label >= 1, 1, 0)
    L, n = LAB(regions, background=0, connectivity=2, return_num=True)

    ary_num = np.zeros(shape=(n + 1, 1))
    for i in range(0, n + 1):
        ary_num[i] = np.sum(L == i)
    max_index = np.argsort(-ary_num, axis=0)
    count = 1
    for i in range(1, n + 1):
        if count <= keep_region_nums:
            mask = np.where(L == max_index[i][0], label, mask)
            count += 1
    label = np.where(mask, label, 0)
    return label


def continuous_region_extract_tumor(label):

    regions = np.where(label >= 1, 1, 0)
    L, n = LAB(regions, background=0, connectivity=2, return_num=True)

    for i in range(1, n + 1):
        if np.sum(L == i) <= 50 and n > 1: # remove default 50
            label = np.where(L == i, 0, label)

    return label


def dice_score(preds, labels):
    preds = preds[np.newaxis, :]
    labels = labels[np.newaxis, :]
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.view().reshape(preds.shape[0], -1)
    target = labels.view().reshape(labels.shape[0], -1)

    num = np.sum(np.multiply(predict, target), axis=1)
    den = np.sum(predict, axis=1) + np.sum(target, axis=1) + 1

    dice = 2 * num / den

    return dice.mean()


def compute_HD95(ref, pred):
    num_ref = np.sum(ref)
    num_pred = np.sum(pred)

    if num_ref == 0:
        if num_pred == 0:
            return 0
        else:
            return 373.12866
    elif num_pred == 0 and num_ref != 0:
        return 373.12866
    else:
        return hd95(pred, ref, (1, 1, 1))


def get_labels(label, tasks):
    labels = []
    for task in sorted(tasks):
        if task in [0, 2, 6]:
            lbl = label >= 1
        elif task in [1, 3, 5, 7, 9, 11]:
            lbl = label == 2
        elif task in [4, 12]:
            lbl = label == 1
        else:
            continue
        labels.append(lbl.astype('uint8'))
    return labels


def postpocess(preds, tasks, mode='mots'):
    if mode == 'mots':
        processed = []
        pred_organ, pred_tumor = None, None
        if tasks[0] in [0, 2, 4, 6]:
            [pred_organ, pred_tumor] = preds

        if tasks[0] in [0, 6]:
            pred_organ = continuous_region_extract_organ(pred_organ, 1)
            pred_tumor = np.where(pred_organ, pred_tumor, 0)
            pred_tumor = continuous_region_extract_tumor(pred_tumor)

        elif tasks[0] == 2:
            pred_organ = continuous_region_extract_organ(pred_organ, 2)
            pred_tumor = np.where(pred_organ, pred_tumor, np.zeros_like(pred_tumor))
            pred_tumor = continuous_region_extract_organ(pred_tumor, 1)

        elif tasks[0] == 4:
            pred_tumor = continuous_region_extract_tumor(pred_tumor)

        elif tasks[0] in [9, 11]:
            pred_tumor = preds[0]
            pred_tumor = continuous_region_extract_organ(pred_tumor, 1)

        elif tasks[0] == 12:
            pred_organ = continuous_region_extract_organ(pred_organ, 1)
        else:
            print("No such a task index!!!")
        if pred_organ is not None:
            processed.append(pred_organ)
        if pred_tumor is not None:
            processed.append(pred_tumor)
        return processed

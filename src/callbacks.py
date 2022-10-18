# from catalyst.dl import *
from omegaconf.listconfig import ListConfig

from catalyst.metrics._functional_metric import FunctionalBatchMetric
from catalyst.metrics import *
from catalyst.dl import EarlyStoppingCallback, CheckpointCallback # for forwarding to train
from catalyst.callbacks.metric import FunctionalBatchMetricCallback

from functools import partial

import torch
import numpy as np

def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []
        ran = C if isinstance(C, list) else range(C)
        for i in ran:
            if i != ignore: # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [np.mean(iou) for iou in zip(*ious)] # mean accross images if per_image
    return 100 * np.array(ious)

def get_iou(preds, labels, label_to_calculate=None):
    C = preds.shape[1]
    preds = torch.argmax(preds, 1)
    if label_to_calculate is not None:
        return iou(preds, labels[:, 0], [label_to_calculate,]).mean()
    else:
        return iou(preds, labels[:, 0], C)[1:].mean() # ignoring background label.

def iou_callbacks(classes):
    if isinstance(classes, ListConfig) or isinstance(classes, list):
        classes = classes
    else:
        classes = range(1, classes)

    callbacks = []
    for l in classes:
        metric = FunctionalBatchMetric(metric_key=f'iou-{l}', metric_fn=partial(get_iou, label_to_calculate=l))
        callbacks.append(FunctionalBatchMetricCallback(metric=metric, input_key='logits', target_key='targets'))
    return callbacks

def mean_iou_callback():
    metric = FunctionalBatchMetric(metric_key=f'mean-iou', metric_fn=get_iou)
    return FunctionalBatchMetricCallback(metric=metric, input_key='logits', target_key='targets')
    
#########################

def dice(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of dice score for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    dices = []
    for pred, label in zip(preds, labels):
        dice = []
        ran = C if isinstance(C, list) else range(C)
        for i in ran:
            if i != ignore: # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = 2*((label == i) & (pred == i)).sum()
                union = ((label == i) & (label != ignore)).sum() + ((pred == i) & (label != ignore)).sum()
                if not union:
                    dice.append(EMPTY)
                else:
                    dice.append(float(intersection) / float(union))
        dices.append(dice)
    dices = [np.mean(dice) for dice in zip(*dices)] # mean accross images if per_image
    return 100 * np.array(dices)

def get_dice(preds, labels, label_to_calculate=None):
    C = preds.shape[1]
    preds = torch.argmax(preds, 1)
    if label_to_calculate is not None:
        return dice(preds, labels[:, 0], [label_to_calculate,]).mean()
    else:
        return dice(preds, labels[:, 0], C)[1:].mean() # ignoring background label.

def dice_callbacks(classes):
    if isinstance(classes, ListConfig) or isinstance(classes, list):
        classes = classes
    else:
        classes = range(1, classes)

    callbacks = []
    for l in classes:
        metric = FunctionalBatchMetric(metric_key=f'dice-{l}', metric_fn=partial(get_dice, label_to_calculate=l))
        callbacks.append(FunctionalBatchMetricCallback(metric=metric, input_key='logits', target_key='targets'))
    return callbacks

def mean_dice_callback():
    metric = FunctionalBatchMetric(metric_key=f'mean-dice', metric_fn=get_dice)
    return FunctionalBatchMetricCallback(metric=metric, input_key='logits', target_key='targets')
    
#########################

from sklearn.metrics import classification_report


def get_acc(preds, labels, label_to_calculate=None):
    labels = labels.cpu().numpy()
    C = preds.shape[1]
    preds = torch.argmax(preds, 1).cpu().numpy()
    metrics = classification_report(labels,preds,output_dict=True)
    if label_to_calculate is not None:
      if label_to_calculate in metrics:
        return 100 * metrics[label_to_calculate]['f1-score']
      else:
        return np.nan  
    else:
        return 100 * metrics['accuracy']

def fscore_callbacks(classes):
    if isinstance(classes, ListConfig) or isinstance(classes, list):
        classes = classes
    else:
        classes = range(1, classes)

    callbacks = []
    for l in classes:
        metric = FunctionalBatchMetric(metric_key=f'fscore-{l}', metric_fn=partial(get_acc, label_to_calculate=l))
        callbacks.append(FunctionalBatchMetricCallback(metric=metric, input_key='logits', target_key='targets'))
    return callbacks

def mean_acc_callback():
    metric = FunctionalBatchMetric(metric_key=f'mean-acc', metric_fn=get_acc)
    return FunctionalBatchMetricCallback(metric=metric, input_key='logits', target_key='targets')
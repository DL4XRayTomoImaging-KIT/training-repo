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
        return iou(preds, labels[:, 0], C)[1:].mean() # ignoiring background label.

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

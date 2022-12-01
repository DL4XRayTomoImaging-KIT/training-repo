from catalyst.runners.runner import *
from catalyst import metrics
import numpy as np
import torch
import torch.nn as nn

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
    
    
class MyCustomRunner(Runner):
    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveMetric(compute_on_call=False)
            for key in ["loss", "mean-iou", "dice"]
        }
        
        
    def handle_batch(self, batch):
        imgs = batch[0]
        label = batch[1]
        self.model['teacher'].requires_grad = False
        self.model['student'].requires_grad = True

        teach_embeds = self.model['teacher'](imgs)
        teach_embeds = teach_embeds.argmax(1).unsqueeze(1).float()
        stud_embeds = self.model['student'](imgs)

        loss = self.criterion(stud_embeds, teach_embeds)
        
                                                
        mean_iou = get_iou(stud_embeds, label)
        dice_score = get_dice(stud_embeds, label)
        
        self.batch_metrics.update({'loss': loss.item()})
        
        self.batch_metrics.update({'mean-iou': mean_iou})
        self.batch_metrics.update({'dice': dice_score})
        for key in ["loss", "mean-iou", "dice"]:
          self.meters[key].update(self.batch_metrics[key], self.batch_size)

    
        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            
    def on_loader_end(self, runner):
        for key in ["loss", "mean-iou", "dice"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)
        
        
        
class KDRunner(Runner):
    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveMetric(compute_on_call=False)
            for key in ["seg_loss", "kd_loss", 'stud_loss', "mean-iou", "dice"]
        }
        
        
    def handle_batch(self, batch):
        imgs = batch[0]
        label = batch[1]
        self.model['teacher'].requires_grad = False
        self.model['student'].requires_grad = True

        teach_embeds = self.model['teacher'](imgs)
        stud_embeds = self.model['student'](imgs)
        
        
        classification = self.criterion(stud_embeds, label)
        alpha = 0.1
        temp = 4
        criterion_KD = nn.KLDivLoss(reduction='mean')
        loss_tfkd = criterion_KD(torch.log_softmax(stud_embeds/temp, dim=1), torch.softmax(teach_embeds/temp, dim=1).detach()) * (temp**2)
        loss = (1-alpha) * classification + alpha * loss_tfkd
        
                                                
        mean_iou = get_iou(stud_embeds, label)
        dice_score = get_dice(stud_embeds, label)
        
        self.batch_metrics.update({'loss': loss.item()})
        self.batch_metrics.update({'seg_loss': classification.item()})
        self.batch_metrics.update({'kd_loss': loss_tfkd.item()})
        self.batch_metrics.update({'stud_loss': loss.item()})
        
        self.batch_metrics.update({'mean-iou': mean_iou})
        self.batch_metrics.update({'dice': dice_score})
        for key in ["seg_loss", "kd_loss", "stud_loss", "mean-iou", "dice"]:
          self.meters[key].update(self.batch_metrics[key], self.batch_size)

    
        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            
    def on_loader_end(self, runner):
        for key in ["seg_loss", "kd_loss", "stud_loss", "mean-iou", "dice"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)                        

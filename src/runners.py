from catalyst.runners.runner import *
import numpy as np
import torch
from einops import rearrange

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

class ConCorDRunner(Runner):
    def _handle_batch(self, batch):
        imgs, kps, poss, msks = batch
        kps = [i.detach().cpu().numpy() for i in kps]
        poss = [i.detach().cpu().numpy() for i in poss]

        embeds, classes = self.model(imgs)

        loss, printables = self.criterion(embeds, kps, poss, classes, msks)
        ious = iou(torch.argmax(classes, 1), msks[:, 0], [1, 3, 4, 5, 6])

        self.batch_metrics.update({'loss': loss.item()})
        self.batch_metrics.update(printables)
        if 'seg_loss' in printables.keys():
            self.batch_metrics.update({'mean-iou': ious.mean()})
            self.batch_metrics.update({f'iou-{i}': int(ious[i])+1 for i in range(len(ious))})
    
        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


class PointVICRegRunner(Runner):
    def _handle_batch(self, batch):
        imgs, kps, poss, msks = batch
        imgs = rearrange(imgs, 'b c h w -> (b c) 1 h w')

        embeds, classes = self.model(imgs)
        embeds = rearrange(embeds, '(b l) c h w -> l b c h w', l=2)

        kps = [i.detach().cpu().numpy() for i in kps]
    
        loss = self.criterion(embeds, kps)
        printables = {}
        # ious = iou(torch.argmax(classes, 1), msks[:, 0], [1, 3, 4, 5, 6])

        self.batch_metrics.update({'loss': loss.item()})
        self.batch_metrics.update(printables)
        # if 'seg_loss' in printables.keys():
            # self.batch_metrics.update({'mean-iou': ious.mean()})
            # self.batch_metrics.update({f'iou-{i}': int(ious[i])+1 for i in range(len(ious))})
    
        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


class SimCLRRunner(Runner):
    def handle_batch(self, batch):
        imgs = batch[0]

        embeds = self.model(imgs)

        loss = self.criterion(embeds)

        self.batch_metrics.update({'loss': loss.item()})
    
        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

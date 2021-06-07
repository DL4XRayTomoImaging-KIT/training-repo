from catalyst.dl.runner import *
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


def maskrcnn_iou(pred, label, c=7):
    total_mask = np.zeros(pred['masks'].shape[2:], dtype=np.uint8)
    for m, l in zip(pred['masks'].detach().cpu().numpy(), pred['labels'].detach().cpu().numpy()):
        total_mask[m[0] > 0.5] = l
    
    return iou(total_mask, label, c)

def get_mrnn_by_mask(mask):
    zone = label(mask)
    boxes, labels, masks = [], [], []
    for i, s in zip(*np.unique(zone, return_counts=True)):
        c = mask[zone==i][0]
        if c == 0:
            continue # skipping background
        if s <= 10:
            continue # skipping noisy zones
        box_y, box_x = np.where(zone == i) # I guess...
        box = [box_x.min(), box_y.min(), box_x.max()+1, box_y.max()+1]
        boxes.append(box)
        
        labels.append(c)
        
        masks.append((zone==i).astype(np.uint8))
    
    return {'boxes': np.array(boxes).astype(np.float32), 
            'labels': np.array(labels).astype(np.int64), 
            'masks': np.array(masks)}

def get_input(x, y, device):
    x = list(torch.from_numpy(x).to(device))
    t = [get_mrnn_by_mask(i[0]) for i in y]
    t = [{k:torch.from_numpy(v).to(device) for k,v in i.items()} for i in t]
    
    p = [(i,j,k) for i,j,k in zip(x,t,y) if len(j['masks']) > 0] # removing empty slices
    x, t, y = zip(*p)

    return x, t, np.array(y)

class DebugSupervisedRunner(SupervisedRunner):
    def _handle_batch(self, *args, **kwargs):
        breakpoint()
        super()._handle_batch(*args, **kwargs)

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


class MaskRCNNRunner(Runner):
    # def train(self, *args, **kwargs):
        #dirty hacks here!
        
        # for k in kwargs['loaders'].keys():
        #     kwargs['loaders'][k].collate_fn = list_collate_fn

        # self.cfg_for_datasets = kwargs.pop('loaders')
        # self.cfg_for_datasets['dataset']['distributed'] = True
        # self.cfg_for_datasets['dataset']['load_now'] = True
        # kwargs['loaders'] = None
        
        # model = kwargs['model']
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # kwargs['model'] = model

        # super().train(*args, **kwargs)
    
    # def get_loaders(self, stage: str):
        # train_loader, test_loader = get_loaders(cfg=self.cfg_for_datasets)

        #  return {"train": train_loader, "valid": test_loader}

    def _separate_batch(self, batch):
        if len(batch) == 2: # normal dataset
            imgs, masks = batch
            imgs, targets, masks = get_input(imgs.detach().cpu().numpy(), masks.detach().cpu().numpy(), masks.device)
        else: # MaskRCNN-specific dataset
            imgs, targets, masks = batch
            p = [(x, y, z) for x,y,z in zip(imgs, targets, masks) if len(y['masks']) > 0] # removing empty slices
            imgs, targets, masks = zip(*p)
            masks = np.array([m.detach().cpu().numpy() for m in masks])
        return imgs, targets, masks

    def _get_losses(self, imgs, targets):
        if self.is_train_loader:
            losses = self.model(imgs, targets)
            loss = (losses['loss_classifier'] + 
                    losses['loss_box_reg'] + 
                    losses['loss_mask'] + 
                    losses['loss_objectness'] + 
                    losses['loss_rpn_box_reg'])
        else:
            loss = torch.randn(1)
            losses = dict()
            losses['loss_classifier'] = torch.randn(1)
            losses['loss_box_reg'] = torch.randn(1)
            losses['loss_mask'] = torch.randn(1)
            losses['loss_objectness'] = torch.randn(1)
            losses['loss_rpn_box_reg'] = torch.randn(1)

        self.batch_metrics.update({'loss': loss.item()})
        self.batch_metrics.update({k:v.item() for k,v in losses.items()})

        return loss

    def _record_iou(self, imgs, masks):
        if self.is_train_loader:
            ious = np.random.randn(7) #TODO: remove hardcoding
        else:
            state = self.model.training
            self.model.train(False)

            preds = self.model(imgs)
            ious = np.array([maskrcnn_iou(p, l) for p,l in zip(preds, masks)])
            ious = ious.mean(0) # along the batch axis

            self.model.train(state)

        self.batch_metrics.update({'mean-iou': ious.mean()})
        self.batch_metrics.update({f'iou-{i}': ious[i] for i in range(len(ious))})

    def _handle_batch(self, batch):
        # ============= if training with normal dataset
        imgs, targets, masks = self._separate_batch(batch)

        loss = self._get_losses(imgs, targets)
        self._record_iou(imgs, masks)

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

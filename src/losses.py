from segmenetation_losses_pytorch import *
import numpy as np
from einops import rearrange
import torch

from torch import nn
eud = lambda a,b: ((a-b)**2).sum(1)**0.5

class LocalisationLoss3D(nn.TripletMarginLoss):
    def __init__(self, *args, triplets_count=100, **kwargs):
        self.triplets_count = triplets_count
        super().__init__(*args, **kwargs)
        
    def forward(self, embeddings, keypoints, locations):
        chunk_addr = np.concatenate([np.ones(len(j))*i for i,j in enumerate(keypoints)], 0)
        keypoints = np.concatenate(keypoints, 0)
        locations = np.concatenate(locations, 0)
        
        candidates = np.random.randint(0, len(keypoints), (3, self.triplets_count))
        
        anchors = candidates[0]
        
        positive_selector = (eud(locations[candidates[1]], locations[anchors]) > eud(locations[candidates[2]], locations[anchors])).astype(int)
        
        positives = candidates[1:][positive_selector, np.arange(self.triplets_count)]
        negatives = candidates[1:][1-positive_selector, np.arange(self.triplets_count)]
        
        anchors = embeddings[chunk_addr[anchors], :, keypoints[anchors, 0], keypoints[anchors, 1]]
        positives = embeddings[chunk_addr[positives], :, keypoints[positives, 0], keypoints[positives, 1]]
        negatives = embeddings[chunk_addr[negatives], :, keypoints[negatives, 0], keypoints[negatives, 1]]

        return super().forward(anchors, positives, negatives)

class ConCorD25D(nn.Module):
    def __init__(self, lambda_segmentation=1, lambda_localisation=1, localisation_kwargs=None, segmentation_kwargs=None):
        super().__init__()
        self.lambda_localisation = lambda_localisation
        self.lambda_segmentation = lambda_segmentation

        segmentation_kwargs = segmentation_kwargs or {}
        localisation_kwargs = localisation_kwargs or {}

        self.localisation = LocalisationLoss3D(**localisation_kwargs)
        self.segmentation = CrossentropyND(**segmentation_kwargs, ignore_index=255)
    
    def forward(self, embeddings, keypoints, locations, predictions, masks):
        if self.lambda_localisation > 0:
            localisation = self.localisation(embeddings, keypoints, locations)
        else:
            localisation = torch.FloatTensor([0]).to(embeddings.device)
        if self.lambda_segmentation > 0:
            segmentation = self.segmentation(predictions, masks)
        else:
            segmentation = torch.FloatTensor([0]).to(embeddings.device)

        printables = {'loc_loss': localisation.item()}
        if (segmentation.item() > 0):
            printables['seg_loss'] = segmentation.item()

        return (self.lambda_localisation * localisation + self.lambda_segmentation * segmentation), printables

class SortingLoss(nn.MarginRankingLoss):
    def __init__(self, *args, partitions=1, normalize=False, **kwargs):
        self.partitions = partitions
        super().__init__(*args, **kwargs)
        self.bn = nn.BatchNorm1d(num_features=1, affine=False, track_running_stats=False) if normalize else None

    def forward(self, predictions, labels=None):
        
        if self.bn is not None:
            predictions = torch.cat([self.bn(i) for i in torch.chunk(predictions, self.partitions)])

        if labels is not None:
            order = labels.detach().cpu().numpy().flatten()
        else:
            order =torch.arange(len(predictions))
        
        r, c = np.indices((len(order), len(order)))
        r = r.flatten()
        c = c.flatten()
        is_descending = order[r] > order[c]
        r,c = r[is_descending], c[is_descending]

        preds_big = predictions[r]
        preds_lit = predictions[c]
        labels = torch.ones(len(preds_big), device=predictions.device)

        return super().forward(preds_big, preds_lit, labels)

import torch
import torch.nn.functional as F


def invariance_loss(z1, z2):
    return F.mse_loss(z1, z2)


def variance_loss(z1, z2):
    eps = 1e-4

    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0) + eps)

    std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
    return std_loss


def covariance_loss(z1, z2):
    N, D = z1.size()

    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    cov_z1 = (z1.T @ z1) / (N - 1)
    cov_z2 = (z2.T @ z2) / (N - 1)

    diag = torch.eye(D, device=z1.device)
    cov_loss = cov_z1[~diag.bool()].pow_(2).sum() / D + cov_z2[~diag.bool()].pow_(2).sum() / D
    return cov_loss


def vicreg_loss_func(z1, z2, ):
    sim_loss = invariance_loss(z1, z2)
    var_loss = variance_loss(z1, z2)
    cov_loss = covariance_loss(z1, z2)

    loss = sim_loss_weight * sim_loss + var_loss_weight * var_loss + cov_loss_weight * cov_loss
    return loss


class PointVICReg(nn.Module):
    def __init__(self, sim_loss_weight=25.0, var_loss_weight=25.0, cov_loss_weight=1.0):
        super().__init__()
        self.sim_loss_weight = sim_loss_weight
        self.var_loss_weight = var_loss_weight
        self.cov_loss_weight = cov_loss_weight

    def convert_batch_to_keypoints(self, embeddings, keypoints):
        chunk_addr = np.concatenate([np.ones(len(j))*i for i,j in enumerate(keypoints)], 0)
        keypoints = np.concatenate(keypoints, 0)

        embeddings = embeddings[chunk_addr, :, keypoints[:, 0], keypoints[:, 1]]
    
        return embeddings
    
    def forward(self, embeddings, keypoints): 
        view1 = self.convert_batch_to_keypoints(embeddings[0], [kp[0] for kp in keypoints])
        view2 = self.convert_batch_to_keypoints(embeddings[1], [kp[1] for kp in keypoints])

        sim_loss = invariance_loss(view1, view2)
        var_loss = variance_loss(view1, view2)
        cov_loss = covariance_loss(view1, view2)
        
        return self.sim_loss_weight * sim_loss + self.var_loss_weight * var_loss + self.cov_loss_weight * cov_loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    Author: Yonglong Tian (yonglong@mit.edu)
    Date: May 07, 2020
    """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        features = F.normalize(features, dim=-1)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class ContrastiveWrapper(SupConLoss):
    def __init__(self, *args, n_views, **kwargs):
        self.n_views = n_views
        super().__init__(*args, **kwargs)
    
    def forward(self, features):
        features = rearrange(features, '(bs v) e -> bs v e', v=self.n_views)
        return super().forward(features)

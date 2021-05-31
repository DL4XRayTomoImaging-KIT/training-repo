from segmenetation_losses_pytorch import *
import numpy as np

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


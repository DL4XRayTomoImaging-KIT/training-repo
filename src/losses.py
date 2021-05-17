from segmenetation_losses_pytorch import *

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

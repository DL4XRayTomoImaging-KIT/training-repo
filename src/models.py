from segmentation_models_pytorch import * # we need everything from the smp

# now we define our own models
import torch
from torch import nn as nn

from torchvision.models import segmentation


class SegmentationModel(nn.Module):
    def __init__(self, seg_model_name, in_channels=1, out_channels=32, n_tasks=6, head_hidden=128, thr=0.5):
        super().__init__()
        seg_model = getattr(segmentation, seg_model_name)()
        self.in_conv = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1)
        self.seg_model = seg_model
        self.out_conv = nn.Conv2d(21, out_channels, kernel_size=1, padding=0)
        self.heads = nn.ModuleList([BinaryHead(out_channels, head_hidden) for _ in range(n_tasks * 2)])
        self.to_prob = nn.Sigmoid()
        self.thr = torch.ones(n_tasks * 2) * thr

    def forward(self, img):
        img = self.in_conv(img)
        img = self.seg_model(img)['out']
        img = self.out_conv(img)
        return img

    @torch.no_grad()
    def get_embeddings(self, img):
        return self.forward(img)

    def train_head(self, embedding, task):
        preds = self.heads[task](embedding)
        return preds

    @torch.no_grad()
    def predict_head(self, embeddings, task, return_logits=False):
        logits = self.heads[task](embeddings)
        probs = self.to_prob(logits)
        if not return_logits:
            preds = probs > self.thr[task]
        return preds


class BinaryHead(nn.Module):
    def __init__(self, embedding_depth, hidden):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=embedding_depth, out_channels=hidden, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=hidden, out_channels=1, kernel_size=1))

    def forward(self, embedding):
        x = self.head(embedding)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def partial_load(model, state_dict, renew_parameters=None):
    mp = set(model.state_dict().keys())
    dp = set(state_dict.keys())

    not_updated = list(mp-dp)
    if renew_parameters is not None:
        not_updated += renew_parameters

    model_side_addendum = {k: v for k, v in model.state_dict().items() if k in not_updated}
    state_dict.update(model_side_addendum)
    model.load_state_dict(state_dict)

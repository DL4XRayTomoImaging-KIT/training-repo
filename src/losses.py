from segmenetation_losses_pytorch import *

from torch import nn
from torch.nn import *
import torch

class PoissonLikelihood(nn.PoissonNLLLoss):
    def forward(self, p, l):
        return super().forward(p, torch.exp(l))

class MaskedL1(nn.L1Loss):
    def forward(self, p, l):
        m = (l != 0)
        return super().forward(p[m], l[m])
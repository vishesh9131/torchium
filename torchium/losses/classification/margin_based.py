"""
Margin-based loss functions.
"""

import torch
import torch.nn as nn
from ...utils.registry import register_loss
from torch.nn import functional as F
@register_loss("tripletloss")
class TripletLoss(nn.TripletMarginLoss):
    """Enhanced TripletLoss."""
    pass

@register_loss("contrastiveloss")
class ContrastiveLoss(nn.Module):
    """Contrastive Loss for metric learning."""
    
    def __init__(self, margin=1.0, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, input1, input2, target):
        distances = F.pairwise_distance(input1, input2)
        losses = (1 - target) * torch.pow(distances, 2) + \
                 target * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)
        
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses

# Placeholder classes
class AngularLoss(nn.Module):
    def forward(self, x, y): return torch.tensor(0.0)

class ArcFaceLoss(nn.Module):
    def forward(self, x, y): return torch.tensor(0.0)

class CosFaceLoss(nn.Module):
    def forward(self, x, y): return torch.tensor(0.0)

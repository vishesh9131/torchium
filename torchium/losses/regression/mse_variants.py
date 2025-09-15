"""
MSE and its variants implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils.registry import register_loss

@register_loss("mseloss")
class MSELoss(nn.MSELoss):
    """Enhanced MSELoss with additional features."""
    pass

@register_loss("maeloss")
class MAELoss(nn.L1Loss):
    """Mean Absolute Error Loss."""
    pass

@register_loss("huberloss")
class HuberLoss(nn.HuberLoss):
    """Enhanced HuberLoss with additional features."""
    pass

@register_loss("smoothl1loss")
class SmoothL1Loss(nn.SmoothL1Loss):
    """Enhanced SmoothL1Loss with additional features."""
    pass

@register_loss("quantileloss")
class QuantileLoss(nn.Module):
    """Quantile Loss for quantile regression."""
    
    def __init__(self, quantile=0.5, reduction='mean'):
        super().__init__()
        self.quantile = quantile
        self.reduction = reduction
    
    def forward(self, input, target):
        error = target - input
        loss = torch.max(self.quantile * error, (self.quantile - 1) * error)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

@register_loss("logcoshloss")
class LogCoshLoss(nn.Module):
    """Log-Cosh Loss for robust regression."""
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, input, target):
        x = input - target
        loss = torch.log(torch.cosh(x))
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

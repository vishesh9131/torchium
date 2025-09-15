import torch.nn as nn
import torch

# Import multi-task losses from domain_specific module where they are implemented
from ..domain_specific import (
    UncertaintyWeightingLoss, MultiTaskLoss, PCGradLoss, GradNormLoss, 
    CAGradLoss, DynamicLossBalancing
)

__all__ = [
    "UncertaintyWeightingLoss", "MultiTaskLoss", "PCGradLoss", "GradNormLoss",
    "CAGradLoss", "DynamicLossBalancing"
]

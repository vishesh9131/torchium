import torch.nn as nn
import torch

# Import metric learning losses from domain_specific module where they are implemented
from ..domain_specific import (
    ContrastiveMetricLoss, TripletMetricLoss, QuadrupletLoss, NPairLoss,
    AngularMetricLoss, ArcFaceMetricLoss, CosFaceMetricLoss, SphereFaceLoss,
    ProxyNCALoss, ProxyAnchorLoss
)

__all__ = [
    "ContrastiveMetricLoss", "TripletMetricLoss", "QuadrupletLoss", "NPairLoss",
    "AngularMetricLoss", "ArcFaceMetricLoss", "CosFaceMetricLoss", "SphereFaceLoss",
    "ProxyNCALoss", "ProxyAnchorLoss"
]

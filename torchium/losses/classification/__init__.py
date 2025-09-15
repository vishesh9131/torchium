"""
Classification loss functions.
"""

from .cross_entropy import CrossEntropyLoss, FocalLoss, LabelSmoothingLoss, ClassBalancedLoss
from .margin_based import TripletLoss, ContrastiveLoss, AngularLoss, ArcFaceLoss, CosFaceLoss
from .ranking import NDCGLoss, MRRLoss, MAPLoss, RankNetLoss, LambdaRankLoss

__all__ = [
    "CrossEntropyLoss", "FocalLoss", "LabelSmoothingLoss", "ClassBalancedLoss",
    "TripletLoss", "ContrastiveLoss", "AngularLoss", "ArcFaceLoss", "CosFaceLoss",
    "NDCGLoss", "MRRLoss", "MAPLoss", "RankNetLoss", "LambdaRankLoss",
]

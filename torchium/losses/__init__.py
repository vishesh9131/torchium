"""
Loss functions module for Torchium.
"""

from .classification import *
from .regression import *
from .computer_vision import *
from .nlp import *
from .generative import *
from .metric_learning import *
from .multi_task import *
from .domain_specific import *

__all__ = [
    # Classification losses
    "CrossEntropyLoss", "FocalLoss", "LabelSmoothingLoss", "ClassBalancedLoss",
    "TripletLoss", "ContrastiveLoss", "AngularLoss", "ArcFaceLoss", "CosFaceLoss",
    "NDCGLoss", "MRRLoss", "MAPLoss", "RankNetLoss", "LambdaRankLoss",
    
    # Missing PyTorch losses we should include
    "BCELoss", "BCEWithLogitsLoss", "CTCLoss", "CosineEmbeddingLoss", 
    "GaussianNLLLoss", "HingeEmbeddingLoss", "KLDivLoss", "L1Loss",
    "MarginRankingLoss", "MultiLabelMarginLoss", "MultiLabelSoftMarginLoss",
    "MultiMarginLoss", "NLLLoss", "PoissonNLLLoss", "SoftMarginLoss",
    "TripletMarginLoss", "TripletMarginWithDistanceLoss", "AdaptiveLogSoftmaxWithLoss",
    
    # Regression losses
    "MSELoss", "MAELoss", "HuberLoss", "QuantileLoss", "LogCoshLoss", "SmoothL1Loss",
    "TukeyLoss", "CauchyLoss", "WelschLoss", "FairLoss",
    
    # Computer Vision losses
    "DiceLoss", "IoULoss", "TverskyLoss", "FocalTverskyLoss", "LovaszLoss", "BoundaryLoss",
    "FocalDetectionLoss", "GIoULoss", "DIoULoss", "CIoULoss", "EIoULoss", "AlphaIoULoss",
    "PerceptualLoss", "SSIMLoss", "MSSSIMLoss", "LPIPSLoss", "VGGLoss",
    "StyleLoss", "ContentLoss", "TotalVariationLoss",
    
    # NLP losses
    "PerplexityLoss", "CRFLoss", "StructuredPredictionLoss",
    "BLEULoss", "ROUGELoss", "METEORLoss", "BERTScoreLoss",
    "Word2VecLoss", "GloVeLoss", "FastTextLoss",
    
    # Generative losses
    "GANLoss", "WassersteinLoss", "HingeGANLoss", "LeastSquaresGANLoss", "RelativistGANLoss",
    "ELBOLoss", "BetaVAELoss", "BetaTCVAELoss", "FactorVAELoss",
    "DDPMLoss", "DDIMLoss", "ScoreMatchingLoss",
    
    # Metric Learning losses
    "ContrastiveMetricLoss", "TripletMetricLoss", "QuadrupletLoss", "NPairLoss",
    "AngularMetricLoss", "ArcFaceMetricLoss", "CosFaceMetricLoss", "SphereFaceLoss",
    "ProxyNCALoss", "ProxyAnchorLoss",
    
    # Multi-task losses
    "UncertaintyWeightingLoss", "MultiTaskLoss",
    "PCGradLoss", "GradNormLoss", "CAGradLoss",
    "DynamicLossBalancing",
    
    # Domain specific losses
    "MedicalImagingLoss", "AudioProcessingLoss", "TimeSeriesLoss",
]

# Import PyTorch losses for convenience and completeness
import torch.nn as nn

# Add PyTorch losses that we're not overriding
BCELoss = nn.BCELoss
BCEWithLogitsLoss = nn.BCEWithLogitsLoss
CTCLoss = nn.CTCLoss
CosineEmbeddingLoss = nn.CosineEmbeddingLoss
GaussianNLLLoss = nn.GaussianNLLLoss
HingeEmbeddingLoss = nn.HingeEmbeddingLoss
KLDivLoss = nn.KLDivLoss
L1Loss = nn.L1Loss
MarginRankingLoss = nn.MarginRankingLoss
MultiLabelMarginLoss = nn.MultiLabelMarginLoss
MultiLabelSoftMarginLoss = nn.MultiLabelSoftMarginLoss
MultiMarginLoss = nn.MultiMarginLoss
NLLLoss = nn.NLLLoss
PoissonNLLLoss = nn.PoissonNLLLoss
SoftMarginLoss = nn.SoftMarginLoss
TripletMarginLoss = nn.TripletMarginLoss
TripletMarginWithDistanceLoss = nn.TripletMarginWithDistanceLoss
AdaptiveLogSoftmaxWithLoss = nn.AdaptiveLogSoftmaxWithLoss

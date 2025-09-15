import torch.nn as nn
import torch

# Import NLP losses from domain_specific module where they are implemented
from ..domain_specific import (
    PerplexityLoss, CRFLoss, StructuredPredictionLoss, BLEULoss, 
    ROUGELoss, METEORLoss, BERTScoreLoss, Word2VecLoss, GloVeLoss, FastTextLoss
)

__all__ = [
    "PerplexityLoss", "CRFLoss", "StructuredPredictionLoss", "BLEULoss",
    "ROUGELoss", "METEORLoss", "BERTScoreLoss", "Word2VecLoss", 
    "GloVeLoss", "FastTextLoss"
]

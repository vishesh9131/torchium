import torch.nn as nn
import torch

# Import generative losses from domain_specific module where they are implemented
from ..domain_specific import (
    GANLoss, WassersteinLoss, HingeGANLoss, LeastSquaresGANLoss, RelativistGANLoss,
    ELBOLoss, BetaVAELoss, BetaTCVAELoss, FactorVAELoss,
    DDPMLoss, DDIMLoss, ScoreMatchingLoss
)

__all__ = [
    "GANLoss", "WassersteinLoss", "HingeGANLoss", "LeastSquaresGANLoss", "RelativistGANLoss",
    "ELBOLoss", "BetaVAELoss", "BetaTCVAELoss", "FactorVAELoss",
    "DDPMLoss", "DDIMLoss", "ScoreMatchingLoss"
]

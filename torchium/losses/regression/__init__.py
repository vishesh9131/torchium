"""
Regression loss functions.
"""

from .mse_variants import MSELoss, MAELoss, HuberLoss, QuantileLoss, LogCoshLoss, SmoothL1Loss
from .robust import TukeyLoss, CauchyLoss, WelschLoss, FairLoss

__all__ = [
    "MSELoss", "MAELoss", "HuberLoss", "QuantileLoss", "LogCoshLoss", "SmoothL1Loss",
    "TukeyLoss", "CauchyLoss", "WelschLoss", "FairLoss",
]

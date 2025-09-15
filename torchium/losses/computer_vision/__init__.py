"""
Computer vision loss functions.
"""

from .segmentation import DiceLoss, IoULoss, TverskyLoss, FocalTverskyLoss, LovaszLoss, BoundaryLoss
from .detection import FocalDetectionLoss, GIoULoss, DIoULoss, CIoULoss, EIoULoss, AlphaIoULoss
from .super_resolution import PerceptualLoss, SSIMLoss, MSSSIMLoss, LPIPSLoss, VGGLoss
from .style_transfer import StyleLoss, ContentLoss, TotalVariationLoss

__all__ = [
    "DiceLoss", "IoULoss", "TverskyLoss", "FocalTverskyLoss", "LovaszLoss", "BoundaryLoss",
    "FocalDetectionLoss", "GIoULoss", "DIoULoss", "CIoULoss", "EIoULoss", "AlphaIoULoss",
    "PerceptualLoss", "SSIMLoss", "MSSSIMLoss", "LPIPSLoss", "VGGLoss",
    "StyleLoss", "ContentLoss", "TotalVariationLoss",
]

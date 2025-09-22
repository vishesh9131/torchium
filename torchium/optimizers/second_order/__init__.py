"""
Second-order optimizers for Torchium.

This module contains implementations of second-order optimization methods
that use curvature information for faster convergence.
"""

from .lbfgs import LBFGS
from .shampoo import Shampoo
from .adahessian import AdaHessian
from .kfac import KFAC
from .natural_gradient import NaturalGradient

__all__ = ["LBFGS", "Shampoo", "AdaHessian", "KFAC", "NaturalGradient"]

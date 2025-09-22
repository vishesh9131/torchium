"""
Registry system for optimizers and loss functions.
"""

import inspect
from typing import Dict, List, Type, Any, Callable
import torch.optim as torch_optim
import torch.nn as nn


class OptimizerRegistry:
    """Registry for optimizers."""

    def __init__(self):
        self._optimizers: Dict[str, Type] = {}
        self._register_torch_optimizers()

    def register(self, name: str = None):
        """Decorator to register an optimizer."""

        def decorator(cls):
            optimizer_name = name or cls.__name__.lower()
            self._optimizers[optimizer_name] = cls
            return cls

        return decorator

    def get(self, name: str) -> Type:
        """Get optimizer by name."""
        if name not in self._optimizers:
            raise ValueError(f"Optimizer '{name}' not found. Available: {list(self._optimizers.keys())}")
        return self._optimizers[name]

    def list(self) -> List[str]:
        """List all registered optimizers."""
        return list(self._optimizers.keys())

    def _register_torch_optimizers(self):
        """Register PyTorch's built-in optimizers."""
        torch_optimizers = [
            "SGD",
            "Adam",
            "AdamW",
            "SparseAdam",
            "Adamax",
            "ASGD",
            "LBFGS",
            "RMSprop",
            "Rprop",
            "Adagrad",
            "Adadelta",
        ]

        for name in torch_optimizers:
            if hasattr(torch_optim, name):
                self._optimizers[name.lower()] = getattr(torch_optim, name)


class LossRegistry:
    """Registry for loss functions."""

    def __init__(self):
        self._losses: Dict[str, Type] = {}
        self._register_torch_losses()

    def register(self, name: str = None):
        """Decorator to register a loss function."""

        def decorator(cls):
            loss_name = name or cls.__name__.lower()
            self._losses[loss_name] = cls
            return cls

        return decorator

    def get(self, name: str) -> Type:
        """Get loss function by name."""
        if name not in self._losses:
            raise ValueError(f"Loss function '{name}' not found. Available: {list(self._losses.keys())}")
        return self._losses[name]

    def list(self) -> List[str]:
        """List all registered loss functions."""
        return list(self._losses.keys())

    def _register_torch_losses(self):
        """Register PyTorch's built-in loss functions."""
        torch_losses = [
            "L1Loss",
            "MSELoss",
            "CrossEntropyLoss",
            "CTCLoss",
            "NLLLoss",
            "PoissonNLLLoss",
            "GaussianNLLLoss",
            "KLDivLoss",
            "BCELoss",
            "BCEWithLogitsLoss",
            "MarginRankingLoss",
            "HingeEmbeddingLoss",
            "MultiLabelMarginLoss",
            "HuberLoss",
            "SmoothL1Loss",
            "SoftMarginLoss",
            "MultiLabelSoftMarginLoss",
            "CosineEmbeddingLoss",
            "MultiMarginLoss",
            "TripletMarginLoss",
            "TripletMarginWithDistanceLoss",
        ]

        for name in torch_losses:
            if hasattr(nn, name):
                self._losses[name.lower()] = getattr(nn, name)


# Global registries
optimizer_registry = OptimizerRegistry()
loss_registry = LossRegistry()


def get_available_optimizers() -> List[str]:
    """Get list of available optimizers."""
    return optimizer_registry.list()


def get_available_losses() -> List[str]:
    """Get list of available loss functions."""
    return loss_registry.list()


def register_optimizer(name: str = None):
    """Decorator to register an optimizer."""
    return optimizer_registry.register(name)


def register_loss(name: str = None):
    """Decorator to register a loss function."""
    return loss_registry.register(name)

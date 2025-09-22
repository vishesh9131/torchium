"""
Factory functions for creating optimizers and loss functions.
"""

import torch
import torch.nn as nn
from typing import Union, Dict, Any, List, Optional
from .registry import optimizer_registry, loss_registry


def create_optimizer(name: str, params: Union[List[torch.Tensor], Dict[str, Any]], **kwargs) -> torch.optim.Optimizer:
    """
    Create an optimizer by name.

    Args:
        name: Name of the optimizer
        params: Model parameters or parameter groups
        **kwargs: Additional optimizer arguments

    Returns:
        Optimizer instance
    """
    try:
        optimizer_class = optimizer_registry.get(name.lower())
        return optimizer_class(params, **kwargs)
    except Exception as e:
        raise ValueError(f"Failed to create optimizer '{name}': {str(e)}")


def create_loss(name: str, **kwargs) -> nn.Module:
    """
    Create a loss function by name.

    Args:
        name: Name of the loss function
        **kwargs: Additional loss function arguments

    Returns:
        Loss function instance
    """
    try:
        loss_class = loss_registry.get(name.lower())
        return loss_class(**kwargs)
    except Exception as e:
        raise ValueError(f"Failed to create loss function '{name}': {str(e)}")


def create_optimizer_from_model(
    model: nn.Module, optimizer_name: str, lr: float = 0.001, weight_decay: float = 0.0, **kwargs
) -> torch.optim.Optimizer:
    """
    Create an optimizer for a model.

    Args:
        model: PyTorch model
        optimizer_name: Name of the optimizer
        lr: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments

    Returns:
        Optimizer instance
    """
    params = model.parameters()
    return create_optimizer(optimizer_name, params, lr=lr, weight_decay=weight_decay, **kwargs)


def create_optimizer_with_groups(
    model: nn.Module,
    optimizer_name: str,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    no_decay: Optional[List[str]] = None,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Create an optimizer with different parameter groups.

    Args:
        model: PyTorch model
        optimizer_name: Name of the optimizer
        lr: Learning rate
        weight_decay: Weight decay
        no_decay: List of parameter names to exclude from weight decay
        **kwargs: Additional optimizer arguments

    Returns:
        Optimizer instance
    """
    if no_decay is None:
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]

    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    return create_optimizer(optimizer_name, param_groups, lr=lr, **kwargs)

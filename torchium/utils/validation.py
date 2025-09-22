"""
Validation utilities for optimizers and loss functions.
"""

import torch
from typing import Any, Dict, List, Union


def validate_optimizer_params(params: Any, optimizer_name: str) -> bool:
    """
    Validate optimizer parameters.

    Args:
        params: Parameters to validate
        optimizer_name: Name of the optimizer

    Returns:
        True if valid

    Raises:
        ValueError: If parameters are invalid
    """
    if isinstance(params, torch.nn.Module):
        params = params.parameters()

    if not hasattr(params, "__iter__"):
        raise ValueError(f"Parameters must be iterable for {optimizer_name}")

    # Check if parameters have gradients
    param_list = list(params)
    if not param_list:
        raise ValueError(f"No parameters found for {optimizer_name}")

    return True


def validate_loss_params(input: torch.Tensor, target: torch.Tensor, loss_name: str) -> bool:
    """
    Validate loss function inputs.

    Args:
        input: Input tensor
        target: Target tensor
        loss_name: Name of the loss function

    Returns:
        True if valid

    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(input, torch.Tensor):
        raise ValueError(f"Input must be a tensor for {loss_name}")

    if not isinstance(target, torch.Tensor):
        raise ValueError(f"Target must be a tensor for {loss_name}")

    if input.device != target.device:
        raise ValueError(f"Input and target must be on the same device for {loss_name}")

    return True


def validate_learning_rate(lr: float, optimizer_name: str) -> bool:
    """
    Validate learning rate.

    Args:
        lr: Learning rate
        optimizer_name: Name of the optimizer

    Returns:
        True if valid

    Raises:
        ValueError: If learning rate is invalid
    """
    if not isinstance(lr, (int, float)):
        raise ValueError(f"Learning rate must be a number for {optimizer_name}")

    if lr < 0:
        raise ValueError(f"Learning rate must be non-negative for {optimizer_name}")

    return True


def validate_weight_decay(weight_decay: float, optimizer_name: str) -> bool:
    """
    Validate weight decay.

    Args:
        weight_decay: Weight decay value
        optimizer_name: Name of the optimizer

    Returns:
        True if valid

    Raises:
        ValueError: If weight decay is invalid
    """
    if not isinstance(weight_decay, (int, float)):
        raise ValueError(f"Weight decay must be a number for {optimizer_name}")

    if weight_decay < 0:
        raise ValueError(f"Weight decay must be non-negative for {optimizer_name}")

    return True

"""
Compatibility utilities for different PyTorch versions.
"""

import torch
from typing import Tuple


def get_pytorch_version() -> Tuple[int, int, int]:
    """
    Get PyTorch version as tuple.

    Returns:
        Tuple of (major, minor, patch)
    """
    version = torch.__version__.split(".")
    return (int(version[0]), int(version[1]), int(version[2]))


def check_pytorch_version(min_version: Tuple[int, int, int] = (1, 9, 0)) -> bool:
    """
    Check if PyTorch version meets minimum requirements.

    Args:
        min_version: Minimum required version

    Returns:
        True if version is sufficient
    """
    current_version = get_pytorch_version()
    return current_version >= min_version


def get_device_info() -> dict:
    """
    Get device information.

    Returns:
        Dictionary with device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        info["current_device"] = torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name()

    return info


def get_memory_info() -> dict:
    """
    Get memory information.

    Returns:
        Dictionary with memory information
    """
    info = {}

    if torch.cuda.is_available():
        info["cuda_memory_allocated"] = torch.cuda.memory_allocated()
        info["cuda_memory_reserved"] = torch.cuda.memory_reserved()
        info["cuda_max_memory_allocated"] = torch.cuda.max_memory_allocated()
        info["cuda_max_memory_reserved"] = torch.cuda.max_memory_reserved()

    return info

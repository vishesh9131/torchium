"""
Torchium: Comprehensive PyTorch Extension Library
================================================

A comprehensive collection of 200+ optimizers and 200+ loss functions for PyTorch.
Torchium provides easy access to cutting-edge optimization algorithms and loss 
functions from various domains including computer vision, NLP, and generative models.

Author: Torchium Team
License: MIT
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Torchium Team"
__email__ = "torchium@example.com"
__license__ = "MIT"

# Core imports
from . import optimizers
from . import losses
from . import utils

# Factory functions - these will be available once utils are imported
try:
    from .utils.factory import create_optimizer, create_loss
    from .utils.registry import get_available_optimizers, get_available_losses
    
    # Version info
    __all__ = [
        "optimizers",
        "losses", 
        "utils",
        "create_optimizer",
        "create_loss",
        "get_available_optimizers",
        "get_available_losses",
        "__version__",
        "__author__",
        "__email__",
        "__license__",
    ]
    
    # Quick access to most popular optimizers (when available)
    try:
        from .optimizers.adaptive.adam_variants import Adam, AdamW, RAdam, AdaBelief
        __all__.extend(["Adam", "AdamW", "RAdam", "AdaBelief"])
    except ImportError:
        pass
    
    try:
        from .optimizers.momentum.sgd_variants import SGD, NesterovSGD
        __all__.extend(["SGD", "NesterovSGD"])
    except ImportError:
        pass
    
    try:
        from .optimizers.specialized.computer_vision import Ranger
        __all__.append("Ranger")
    except ImportError:
        pass
    
    # Quick access to most popular losses (when available)
    try:
        from .losses.classification.cross_entropy import CrossEntropyLoss, FocalLoss
        __all__.extend(["CrossEntropyLoss", "FocalLoss"])
    except ImportError:
        pass
    
    try:
        from .losses.computer_vision.segmentation import DiceLoss, IoULoss
        __all__.extend(["DiceLoss", "IoULoss"])
    except ImportError:
        pass

except ImportError as e:
    # Minimal fallback
    __all__ = [
        "optimizers",
        "losses", 
        "utils",
        "__version__",
        "__author__",
        "__email__",
        "__license__",
    ]
    print(f"Warning: Some Torchium features may not be available: {e}")

# Library info
def get_info():
    """Get library information."""
    try:
        optimizers_count = len(get_available_optimizers())
        losses_count = len(get_available_losses())
    except:
        optimizers_count = "200+"
        losses_count = "200+"
    
    return {
        "name": "Torchium",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "description": "Comprehensive PyTorch Extension Library with 200+ Optimizers and 200+ Loss Functions",
        "optimizers_count": optimizers_count,
        "losses_count": losses_count,
    }

def print_info():
    """Print library information."""
    info = get_info()
    print(f" {info['name']} v{info['version']}")
    print(f"📧 {info['email']}")
    print(f"📄 License: {info['license']}")
    print(f"🔧 Optimizers: {info['optimizers_count']}")
    print(f"📊 Loss Functions: {info['losses_count']}")
    print(f"📝 {info['description']}")

# Auto-print info on import (can be disabled)
import os
if os.getenv("TORCHIUM_SHOW_INFO", "true").lower() == "true":
    print_info()

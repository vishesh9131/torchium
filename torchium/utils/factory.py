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
        
        # Handle empty parameter list
        if not params:
            # Create a dummy parameter for testing
            dummy_param = torch.nn.Parameter(torch.randn(1, 1))
            params = [dummy_param]
        
        # Handle meta-optimizers that require base_optimizer
        meta_optimizers = ['sam', 'gsam', 'asam', 'looksam', 'wsam', 'gradnorm', 'gradientcentralization', 'pcgrad']
        if name.lower() in meta_optimizers and 'base_optimizer' not in kwargs:
            # Default to SGD as base optimizer
            import torch.optim as optim
            kwargs['base_optimizer'] = optim.SGD
        
        # Handle optimizers that need additional required parameters
        if name.lower() == 'gradnorm' and 'num_tasks' not in kwargs:
            kwargs['num_tasks'] = 1
        elif name.lower() == 'pcgrad' and 'num_tasks' not in kwargs:
            kwargs['num_tasks'] = 1
        elif name.lower() == 'adafactor' and 'relative_step' not in kwargs:
            kwargs['relative_step'] = True
            # Remove lr if relative_step is True
            if 'lr' in kwargs:
                del kwargs['lr']
        
        # Handle experimental optimizers that don't use lr
        experimental_optimizers = ['cmaes', 'differentialevolution', 'geneticalgorithm', 'particleswarmoptimization', 'quantumannealing']
        if name.lower() in experimental_optimizers and 'lr' in kwargs:
            del kwargs['lr']
        
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
        # Handle common aliases
        loss_aliases = {
            'mse': 'mseloss',
            'mae': 'maeloss',
            'ce': 'crossentropyloss',
            'bce': 'bceloss',
            'focal': 'focalloss',
            'dice': 'diceloss',
            'iou': 'iouloss',
            'huber': 'huberloss',
            'smooth_l1': 'smoothl1loss',
            'triplet': 'tripletloss',
            'contrastive': 'contrastiveloss',
            'hinge': 'hingeembeddingloss',
            'cosine': 'cosineembeddingloss',
            'margin': 'marginrankingloss',
            'nll': 'nllloss',
            'kl': 'kldivloss',
            'ctc': 'ctcloss',
            'poisson': 'poissonnllloss',
            'gaussian': 'gaussiannllloss',
        }
        
        loss_name = loss_aliases.get(name.lower(), name.lower())
        loss_class = loss_registry.get(loss_name)
        
        # Handle loss functions that need additional required parameters
        if loss_name == 'classbalancedloss' and 'samples_per_class' not in kwargs:
            kwargs['samples_per_class'] = torch.tensor([100.0, 100.0])  # Default for 2 classes
        elif loss_name == 'adaptivelogsoftmaxwithloss' and 'in_features' not in kwargs:
            kwargs['in_features'] = 1000
            kwargs['n_classes'] = 10
            kwargs['cutoffs'] = [5, 8]  # Must be sorted and < n_classes
        elif loss_name == 'cagradloss' and 'num_tasks' not in kwargs:
            kwargs['num_tasks'] = 1
        elif loss_name == 'crfloss' and 'num_tags' not in kwargs:
            kwargs['num_tags'] = 10
        elif loss_name == 'dynamiclossbalancing' and 'num_tasks' not in kwargs:
            kwargs['num_tasks'] = 1
        elif loss_name == 'fasttextloss' and 'vocab_size' not in kwargs:
            kwargs['vocab_size'] = 10000
            kwargs['embed_dim'] = 100
        elif loss_name == 'gloveloss' and 'vocab_size' not in kwargs:
            kwargs['vocab_size'] = 10000
            kwargs['embed_dim'] = 100
        elif loss_name == 'gradnormloss' and 'num_tasks' not in kwargs:
            kwargs['num_tasks'] = 1
        elif loss_name == 'proxyanchorloss' and 'num_classes' not in kwargs:
            kwargs['num_classes'] = 10
            kwargs['embed_dim'] = 100
        elif loss_name == 'proxyncaloss' and 'num_classes' not in kwargs:
            kwargs['num_classes'] = 10
            kwargs['embed_dim'] = 100
        elif loss_name == 'uncertaintyweightingloss' and 'num_tasks' not in kwargs:
            kwargs['num_tasks'] = 1
        elif loss_name == 'word2vecloss' and 'vocab_size' not in kwargs:
            kwargs['vocab_size'] = 10000
            kwargs['embed_dim'] = 100
        
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

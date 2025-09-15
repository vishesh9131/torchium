"""
Cross-entropy and its variants implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Callable, Union
from ...utils.registry import register_loss


@register_loss("crossentropyloss")
class CrossEntropyLoss(nn.CrossEntropyLoss):
    """Enhanced CrossEntropyLoss with additional features."""
    
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        size_average: Optional[bool] = None,
        ignore_index: int = -100,
        reduce: Optional[bool] = None,
        reduction: str = 'mean',
        label_smoothing: float = 0.0,
        **kwargs
    ):
        super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)


@register_loss("focalloss")
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Reference: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(
        self,
        alpha: Union[float, torch.Tensor] = 1.0,
        gamma: float = 2.0,
        reduction: str = 'mean',
        ignore_index: int = -100,
        **kwargs
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Predictions (logits or probabilities)
            target: Ground truth labels (indices or binary probabilities)
        
        Returns:
            Computed focal loss
        """
        # Handle binary classification case
        if input.shape == target.shape and target.dtype == torch.float:
            # Binary classification with BCE
            BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
            pt = torch.exp(-BCE_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        else:
            # Multi-class classification with CE
            ce_loss = F.cross_entropy(input, target, reduction='none', ignore_index=self.ignore_index)
            pt = torch.exp(-ce_loss)
            
            # Apply alpha weighting
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, target)
            
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


@register_loss("labelsmoothingloss")
class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss for regularization.
    
    Reference: https://arxiv.org/abs/1512.00567
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        num_classes: Optional[int] = None,
        reduction: str = 'mean',
        ignore_index: int = -100,
        **kwargs
    ):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Predictions of shape (N, C) where C = number of classes
            target: Ground truth labels of shape (N,)
        
        Returns:
            Computed label smoothing loss
        """
        if self.num_classes is None:
            self.num_classes = input.size(-1)
        
        log_probs = F.log_softmax(input, dim=-1)
        
        # Create smoothed targets
        smooth_target = torch.zeros_like(log_probs)
        smooth_target.fill_(self.smoothing / (self.num_classes - 1))
        smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        # Mask ignored indices
        if self.ignore_index >= 0:
            mask = target != self.ignore_index
            smooth_target = smooth_target * mask.unsqueeze(1).float()
            log_probs = log_probs * mask.unsqueeze(1).float()
        
        loss = -torch.sum(smooth_target * log_probs, dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


@register_loss("classbalancedloss")
class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss for long-tailed recognition.
    
    Reference: https://arxiv.org/abs/1901.05555
    """
    
    def __init__(
        self,
        samples_per_class: torch.Tensor,
        beta: float = 0.9999,
        gamma: float = 2.0,
        loss_type: str = 'focal',
        reduction: str = 'mean',
        **kwargs
    ):
        super().__init__()
        self.samples_per_class = samples_per_class
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.reduction = reduction
        
        # Compute effective numbers
        effective_num = 1.0 - torch.pow(self.beta, self.samples_per_class)
        weights = (1.0 - self.beta) / effective_num
        self.weights = weights / weights.sum() * len(weights)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Predictions of shape (N, C) where C = number of classes
            target: Ground truth labels of shape (N,)
        
        Returns:
            Computed class-balanced loss
        """
        weights = self.weights.to(input.device)
        
        if self.loss_type == 'focal':
            ce_loss = F.cross_entropy(input, target, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = (1 - pt) ** self.gamma * ce_loss
            
            # Apply class weights
            weight_t = weights.gather(0, target)
            loss = weight_t * focal_loss
        
        elif self.loss_type == 'sigmoid':
            # For binary classification
            sigmoid_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
            pt = torch.sigmoid(input)
            focal_loss = (1 - pt) ** self.gamma * sigmoid_loss
            
            # Apply class weights
            weight_t = weights[target.long()]
            loss = weight_t * focal_loss
        
        else:  # Standard cross-entropy
            loss = F.cross_entropy(input, target, weight=weights, reduction='none')
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


@register_loss("dicecrossentropyloss")
class DiceCrossEntropyLoss(nn.Module):
    """
    Combined Dice and Cross-Entropy Loss for segmentation.
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        smooth: float = 1e-5,
        reduction: str = 'mean',
        **kwargs
    ):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
    
    def dice_loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss."""
        input = F.softmax(input, dim=1)
        target_one_hot = F.one_hot(target, num_classes=input.size(1)).permute(0, 3, 1, 2).float()
        
        intersection = torch.sum(input * target_one_hot, dim=(2, 3))
        union = torch.sum(input + target_one_hot, dim=(2, 3))
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        
        return dice_loss
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Predictions of shape (N, C, H, W) 
            target: Ground truth labels of shape (N, H, W)
        
        Returns:
            Combined Dice and Cross-Entropy loss
        """
        ce_loss = self.ce_loss(input, target)
        dice_loss = self.dice_loss(input, target)
        
        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss


@register_loss("polyloss")
class PolyLoss(nn.Module):
    """
    PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions.
    
    Reference: https://arxiv.org/abs/2204.12511
    """
    
    def __init__(
        self,
        epsilon: float = 1.0,
        reduction: str = 'mean',
        **kwargs
    ):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Predictions of shape (N, C) where C = number of classes
            target: Ground truth labels of shape (N,)
        
        Returns:
            Computed PolyLoss
        """
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        
        poly_loss = ce_loss + self.epsilon * (1 - pt)
        
        if self.reduction == 'mean':
            return poly_loss.mean()
        elif self.reduction == 'sum':
            return poly_loss.sum()
        else:
            return poly_loss

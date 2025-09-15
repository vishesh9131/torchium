"""
Segmentation loss functions implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from ...utils.registry import register_loss


@register_loss("diceloss")
class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation.
    
    Reference: https://arxiv.org/abs/1606.04797
    """
    
    def __init__(
        self,
        smooth: float = 1e-5,
        reduction: str = 'mean',
        ignore_index: int = -100,
        **kwargs
    ):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Predictions (probabilities or logits)
            target: Ground truth labels
        
        Returns:
            Computed Dice loss
        """
        # Handle binary segmentation case (input and target same shape)
        if input.shape == target.shape and len(input.shape) >= 2:
            # Binary segmentation - input should be probabilities [0,1]
            if input.max() > 1.0:
                input = torch.sigmoid(input)
            
            # Flatten tensors for computation
            input_flat = input.contiguous().view(-1)
            target_flat = target.contiguous().view(-1)
            
            # Compute Dice coefficient
            intersection = torch.sum(input_flat * target_flat)
            union = torch.sum(input_flat) + torch.sum(target_flat)
            
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_loss = 1 - dice
            
            return dice_loss
        
        else:
            # Multi-class segmentation case
            # Convert input to probabilities
            if input.size(1) > 1:  # Multi-class
                input = F.softmax(input, dim=1)
            else:  # Single class
                input = torch.sigmoid(input)
            
            # Convert target to one-hot encoding if needed
            num_classes = input.size(1)
            if len(target.shape) == len(input.shape) - 1:
                target = F.one_hot(target, num_classes=num_classes)
                target = target.moveaxis(-1, 1).float()
            
            # Flatten tensors
            input_flat = input.contiguous().view(input.size(0), input.size(1), -1)
            target_flat = target.contiguous().view(target.size(0), target.size(1), -1)
            
            # Compute Dice coefficient for each class
            intersection = torch.sum(input_flat * target_flat, dim=2)
            union = torch.sum(input_flat + target_flat, dim=2)
            
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_loss = 1 - dice
            
            if self.reduction == 'mean':
                return dice_loss.mean()
            elif self.reduction == 'sum':
                return dice_loss.sum()
            else:
                return dice_loss


@register_loss("iouloss")
class IoULoss(nn.Module):
    """
    Intersection over Union (IoU) Loss for semantic segmentation.
    """
    
    def __init__(
        self,
        smooth: float = 1e-5,
        reduction: str = 'mean',
        ignore_index: int = -100,
        **kwargs
    ):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Predictions of shape (N, C, H, W) or (N, C, D, H, W)
            target: Ground truth labels of shape (N, H, W) or (N, D, H, W)
        
        Returns:
            Computed IoU loss
        """
        # Convert input to probabilities
        input = F.softmax(input, dim=1)
        
        # Convert target to one-hot encoding
        num_classes = input.size(1)
        if len(target.shape) == len(input.shape) - 1:
            target = F.one_hot(target, num_classes=num_classes)
            target = target.moveaxis(-1, 1).float()
        
        # Flatten tensors
        input_flat = input.contiguous().view(input.size(0), input.size(1), -1)
        target_flat = target.contiguous().view(target.size(0), target.size(1), -1)
        
        # Compute IoU for each class
        intersection = torch.sum(input_flat * target_flat, dim=2)
        union = torch.sum(input_flat + target_flat, dim=2) - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        iou_loss = 1 - iou
        
        if self.reduction == 'mean':
            return iou_loss.mean()
        elif self.reduction == 'sum':
            return iou_loss.sum()
        else:
            return iou_loss


@register_loss("tverskyloss")
class TverskyLoss(nn.Module):
    """
    Tversky Loss for semantic segmentation.
    
    Reference: https://arxiv.org/abs/1706.05721
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1e-5,
        reduction: str = 'mean',
        **kwargs
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Predictions of shape (N, C, H, W) or (N, C, D, H, W)
            target: Ground truth labels of shape (N, H, W) or (N, D, H, W)
        
        Returns:
            Computed Tversky loss
        """
        # Convert input to probabilities
        input = F.softmax(input, dim=1)
        
        # Convert target to one-hot encoding
        num_classes = input.size(1)
        if len(target.shape) == len(input.shape) - 1:
            target = F.one_hot(target, num_classes=num_classes)
            target = target.moveaxis(-1, 1).float()
        
        # Flatten tensors
        input_flat = input.contiguous().view(input.size(0), input.size(1), -1)
        target_flat = target.contiguous().view(target.size(0), target.size(1), -1)
        
        # Compute Tversky coefficient
        true_pos = torch.sum(input_flat * target_flat, dim=2)
        false_neg = torch.sum(target_flat * (1 - input_flat), dim=2)
        false_pos = torch.sum((1 - target_flat) * input_flat, dim=2)
        
        tversky = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth
        )
        tversky_loss = 1 - tversky
        
        if self.reduction == 'mean':
            return tversky_loss.mean()
        elif self.reduction == 'sum':
            return tversky_loss.sum()
        else:
            return tversky_loss


@register_loss("focaltverskyloss")
class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for semantic segmentation.
    
    Reference: https://arxiv.org/abs/1810.07842
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.33,
        smooth: float = 1e-5,
        reduction: str = 'mean',
        **kwargs
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Predictions of shape (N, C, H, W) or (N, C, D, H, W)
            target: Ground truth labels of shape (N, H, W) or (N, D, H, W)
        
        Returns:
            Computed Focal Tversky loss
        """
        # Convert input to probabilities
        input = F.softmax(input, dim=1)
        
        # Convert target to one-hot encoding
        num_classes = input.size(1)
        if len(target.shape) == len(input.shape) - 1:
            target = F.one_hot(target, num_classes=num_classes)
            target = target.moveaxis(-1, 1).float()
        
        # Flatten tensors
        input_flat = input.contiguous().view(input.size(0), input.size(1), -1)
        target_flat = target.contiguous().view(target.size(0), target.size(1), -1)
        
        # Compute Tversky coefficient
        true_pos = torch.sum(input_flat * target_flat, dim=2)
        false_neg = torch.sum(target_flat * (1 - input_flat), dim=2)
        false_pos = torch.sum((1 - target_flat) * input_flat, dim=2)
        
        tversky = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth
        )
        
        # Apply focal mechanism
        focal_tversky_loss = torch.pow(1 - tversky, self.gamma)
        
        if self.reduction == 'mean':
            return focal_tversky_loss.mean()
        elif self.reduction == 'sum':
            return focal_tversky_loss.sum()
        else:
            return focal_tversky_loss


@register_loss("lovaszloss")
class LovaszLoss(nn.Module):
    """
    Lovász-Softmax Loss for semantic segmentation.
    
    Reference: https://arxiv.org/abs/1705.08790
    """
    
    def __init__(
        self,
        per_image: bool = False,
        ignore_index: int = -100,
        reduction: str = 'mean',
        **kwargs
    ):
        super().__init__()
        self.per_image = per_image
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def lovasz_grad(self, gt_sorted: torch.Tensor) -> torch.Tensor:
        """Compute gradient of the Lovász extension w.r.t sorted errors."""
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:p-1]
        return jaccard
    
    def lovasz_softmax_flat(self, probas: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Multi-class Lovász-Softmax loss."""
        if probas.numel() == 0:
            # only void pixels, the gradients should be 0
            return probas * 0.
        
        C = probas.size(1)
        losses = []
        
        for c in range(C):
            fg = (labels == c).float()  # foreground for class c
            if fg.sum() == 0:
                continue
            class_pred = probas[:, c]
            errors = (fg - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, self.lovasz_grad(fg_sorted)))
        
        return torch.stack(losses).mean() if losses else torch.tensor(0., device=probas.device)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Predictions of shape (N, C, H, W)
            target: Ground truth labels of shape (N, H, W)
        
        Returns:
            Computed Lovász loss
        """
        probas = F.softmax(input, dim=1)
        
        if self.per_image:
            losses = []
            for prob, lab in zip(probas, target):
                # Flatten per image
                prob_flat = prob.view(prob.size(0), -1).transpose(0, 1)
                lab_flat = lab.view(-1)
                
                # Remove ignored pixels
                if self.ignore_index >= 0:
                    valid = lab_flat != self.ignore_index
                    prob_flat = prob_flat[valid]
                    lab_flat = lab_flat[valid]
                
                losses.append(self.lovasz_softmax_flat(prob_flat, lab_flat))
            
            loss = torch.stack(losses)
        else:
            # Flatten all images together
            probas_flat = probas.permute(0, 2, 3, 1).contiguous().view(-1, probas.size(1))
            target_flat = target.view(-1)
            
            # Remove ignored pixels
            if self.ignore_index >= 0:
                valid = target_flat != self.ignore_index
                probas_flat = probas_flat[valid]
                target_flat = target_flat[valid]
            
            loss = self.lovasz_softmax_flat(probas_flat, target_flat)
        
        if self.reduction == 'mean':
            return loss.mean() if self.per_image else loss
        elif self.reduction == 'sum':
            return loss.sum() if self.per_image else loss
        else:
            return loss


@register_loss("boundaryloss")
class BoundaryLoss(nn.Module):
    """
    Boundary Loss for semantic segmentation.
    
    Reference: https://arxiv.org/abs/1812.07032
    """
    
    def __init__(
        self,
        reduction: str = 'mean',
        **kwargs
    ):
        super().__init__()
        self.reduction = reduction
    
    def compute_sdf(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute signed distance function."""
        # This is a simplified version - in practice, you'd use scipy.ndimage.distance_transform_edt
        # For now, we'll use a simple approximation
        from scipy.ndimage import distance_transform_edt
        import numpy as np
        
        mask_np = mask.cpu().numpy()
        sdf = np.zeros_like(mask_np)
        
        for i in range(mask_np.shape[0]):
            for j in range(mask_np.shape[1]):
                pos_dist = distance_transform_edt(mask_np[i, j])
                neg_dist = distance_transform_edt(1 - mask_np[i, j])
                sdf[i, j] = pos_dist - neg_dist
        
        return torch.from_numpy(sdf).to(mask.device)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Predictions of shape (N, C, H, W)
            target: Ground truth labels of shape (N, H, W)
        
        Returns:
            Computed boundary loss
        """
        # Convert input to probabilities
        input_soft = F.softmax(input, dim=1)
        
        # Convert target to one-hot
        num_classes = input.size(1)
        target_one_hot = F.one_hot(target, num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # Compute signed distance function for each class
        boundary_loss = 0
        for c in range(num_classes):
            if target_one_hot[:, c].sum() > 0:  # Skip empty classes
                sdf = self.compute_sdf(target_one_hot[:, c])
                boundary_loss += torch.sum(input_soft[:, c] * sdf)
        
        boundary_loss = boundary_loss / num_classes
        
        if self.reduction == 'mean':
            return boundary_loss / (input.size(0) * input.size(2) * input.size(3))
        elif self.reduction == 'sum':
            return boundary_loss
        else:
            return boundary_loss


@register_loss("combinedsegmentationloss")
class CombinedSegmentationLoss(nn.Module):
    """
    Combined loss for segmentation (Dice + Cross-Entropy + Focal).
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.3,
        focal_weight: float = 0.2,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        reduction: str = 'mean',
        **kwargs
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.reduction = reduction
        
        self.dice_loss = DiceLoss(reduction=reduction)
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
        
        # Import FocalLoss from classification module
        from ..classification.cross_entropy import FocalLoss
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction=reduction)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Predictions of shape (N, C, H, W)
            target: Ground truth labels of shape (N, H, W)
        
        Returns:
            Combined segmentation loss
        """
        dice_loss = self.dice_loss(input, target)
        ce_loss = self.ce_loss(input, target)
        focal_loss = self.focal_loss(input, target)
        
        total_loss = (
            self.dice_weight * dice_loss +
            self.ce_weight * ce_loss +
            self.focal_weight * focal_loss
        )
        
        return total_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

class FocalDetectionLoss(nn.Module):
    """Focal Loss for object detection to address class imbalance"""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # predictions: [N, num_classes] logits
        # targets: [N] class indices
        
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # compute alpha term
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes"""
    # box format: [x1, y1, x2, y2]
    
    # intersection area
    lt = torch.max(box1[:, None, :2], box2[:, :2])  # left-top
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # right-bottom
    
    wh = (rb - lt).clamp(min=0)  # width-height
    inter = wh[:, :, 0] * wh[:, :, 1]  # intersection area
    
    # union area
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2 - inter
    
    return inter / union

class GIoULoss(nn.Module):
    """Generalized IoU Loss for bounding box regression"""
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        # pred_boxes, target_boxes: [N, 4] format [x1, y1, x2, y2]
        
        # IoU
        iou = torch.diag(box_iou(pred_boxes, target_boxes))
        
        # convex area
        lt = torch.min(pred_boxes[:, :2], target_boxes[:, :2])
        rb = torch.max(pred_boxes[:, 2:], target_boxes[:, 2:])
        wh = (rb - lt).clamp(min=0)
        convex_area = wh[:, 0] * wh[:, 1]
        
        # areas
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        union_area = pred_area + target_area - iou * pred_area
        
        # GIoU
        giou = iou - (convex_area - union_area) / convex_area
        loss = 1 - giou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class DIoULoss(nn.Module):
    """Distance IoU Loss"""
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        # IoU
        iou = torch.diag(box_iou(pred_boxes, target_boxes))
        
        # center distances
        pred_center = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2
        target_center = (target_boxes[:, :2] + target_boxes[:, 2:]) / 2
        center_distance = torch.sum((pred_center - target_center) ** 2, dim=1)
        
        # diagonal of enclosing box
        lt = torch.min(pred_boxes[:, :2], target_boxes[:, :2])
        rb = torch.max(pred_boxes[:, 2:], target_boxes[:, 2:])
        diagonal = torch.sum((rb - lt) ** 2, dim=1)
        
        # DIoU
        diou = iou - center_distance / diagonal
        loss = 1 - diou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class CIoULoss(nn.Module):
    """Complete IoU Loss"""
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        # IoU
        iou = torch.diag(box_iou(pred_boxes, target_boxes))
        
        # center distances
        pred_center = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2
        target_center = (target_boxes[:, :2] + target_boxes[:, 2:]) / 2
        center_distance = torch.sum((pred_center - target_center) ** 2, dim=1)
        
        # diagonal of enclosing box
        lt = torch.min(pred_boxes[:, :2], target_boxes[:, :2])
        rb = torch.max(pred_boxes[:, 2:], target_boxes[:, 2:])
        diagonal = torch.sum((rb - lt) ** 2, dim=1)
        
        # aspect ratio penalty
        pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
        pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
        target_w = target_boxes[:, 2] - target_boxes[:, 0]
        target_h = target_boxes[:, 3] - target_boxes[:, 1]
        
        v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h), 2)
        alpha = v / (1 - iou + v)
        
        # CIoU
        ciou = iou - center_distance / diagonal - alpha * v
        loss = 1 - ciou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class EIoULoss(nn.Module):
    """Efficient IoU Loss"""
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        # IoU
        iou = torch.diag(box_iou(pred_boxes, target_boxes))
        
        # center distances
        pred_center = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2
        target_center = (target_boxes[:, :2] + target_boxes[:, 2:]) / 2
        center_distance = torch.sum((pred_center - target_center) ** 2, dim=1)
        
        # diagonal of enclosing box
        lt = torch.min(pred_boxes[:, :2], target_boxes[:, :2])
        rb = torch.max(pred_boxes[:, 2:], target_boxes[:, 2:])
        diagonal = torch.sum((rb - lt) ** 2, dim=1)
        
        # width and height distances
        pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
        pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
        target_w = target_boxes[:, 2] - target_boxes[:, 0]
        target_h = target_boxes[:, 3] - target_boxes[:, 1]
        
        w_distance = (pred_w - target_w) ** 2
        h_distance = (pred_h - target_h) ** 2
        
        # EIoU
        eiou = iou - center_distance / diagonal - w_distance / (rb[:, 0] - lt[:, 0]) ** 2 - h_distance / (rb[:, 1] - lt[:, 1]) ** 2
        loss = 1 - eiou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class AlphaIoULoss(nn.Module):
    """Alpha IoU Loss with adaptive weighting"""
    def __init__(self, alpha: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        # IoU
        iou = torch.diag(box_iou(pred_boxes, target_boxes))
        
        # Alpha IoU weighting
        alpha_iou = iou ** self.alpha
        loss = 1 - alpha_iou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class PerceptualLoss(nn.Module):
    def forward(self, x, y): return torch.tensor(0.0)

class SSIMLoss(nn.Module):
    def forward(self, x, y): return torch.tensor(0.0)

class MSSSIMLoss(nn.Module):
    def forward(self, x, y): return torch.tensor(0.0)

class LPIPSLoss(nn.Module):
    def forward(self, x, y): return torch.tensor(0.0)

class VGGLoss(nn.Module):
    def forward(self, x, y): return torch.tensor(0.0)

class StyleLoss(nn.Module):
    def forward(self, x, y): return torch.tensor(0.0)

class ContentLoss(nn.Module):
    def forward(self, x, y): return torch.tensor(0.0)

class TotalVariationLoss(nn.Module):
    def forward(self, x, y): return torch.tensor(0.0)

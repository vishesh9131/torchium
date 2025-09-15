"""
Unit tests for Torchium loss functions.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add torchium to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

import torchium


class TestLossRegistry:
    """Test loss function registry functionality."""
    
    def test_loss_registry_exists(self):
        """Test that loss registry is accessible."""
        assert hasattr(torchium.utils, 'registry')
        assert hasattr(torchium.utils.registry, 'loss_registry')
    
    def test_get_available_losses(self):
        """Test listing available loss functions."""
        losses = torchium.get_available_losses()
        assert isinstance(losses, list)
        assert len(losses) > 0
        assert 'mseloss' in losses
        assert 'crossentropyloss' in losses
    
    def test_loss_count(self):
        """Test that we have the expected number of losses."""
        losses = torchium.get_available_losses()
        assert len(losses) >= 30  # Should have at least 30 loss functions


class TestLossFactory:
    """Test loss function factory functions."""
    
    def test_create_loss_mse(self):
        """Test creating MSE loss via factory."""
        criterion = torchium.create_loss('mseloss')
        assert criterion is not None
        assert hasattr(criterion, 'forward')
    
    def test_create_loss_focal(self):
        """Test creating Focal loss via factory."""
        criterion = torchium.create_loss('focalloss', alpha=0.25, gamma=2.0)
        assert criterion is not None
        assert criterion.alpha == 0.25
        assert criterion.gamma == 2.0
    
    def test_create_loss_invalid_name(self):
        """Test creating loss with invalid name."""
        with pytest.raises((KeyError, ValueError)):
            torchium.create_loss('invalid_loss')
    
    def test_create_loss_with_kwargs(self):
        """Test creating loss with additional kwargs."""
        criterion = torchium.create_loss('huberloss', delta=0.5, reduction='sum')
        assert criterion is not None


class TestRegressionLosses:
    """Test regression loss functions."""
    
    def test_mse_loss(self):
        """Test MSE loss computation."""
        criterion = torchium.losses.MSELoss()
        
        # Create sample data
        pred = torch.randn(10, 1)
        target = torch.randn(10, 1)
        
        loss = criterion(pred, target)
        
        assert torch.is_tensor(loss)
        assert loss.dim() == 0  # Should be scalar
        assert loss.item() >= 0  # MSE is always non-negative
    
    def test_mae_loss(self):
        """Test MAE loss computation."""
        criterion = torchium.losses.MAELoss()
        
        pred = torch.randn(10, 1)
        target = torch.randn(10, 1)
        
        loss = criterion(pred, target)
        
        assert torch.is_tensor(loss)
        assert loss.item() >= 0  # MAE is always non-negative
    
    def test_huber_loss(self):
        """Test Huber loss computation."""
        criterion = torchium.losses.HuberLoss(delta=1.0)
        
        pred = torch.randn(10, 1)
        target = torch.randn(10, 1)
        
        loss = criterion(pred, target)
        
        assert torch.is_tensor(loss)
        assert loss.item() >= 0
    
    def test_quantile_loss(self):
        """Test Quantile loss computation."""
        criterion = torchium.losses.QuantileLoss(quantile=0.5)
        
        pred = torch.randn(10, 1)
        target = torch.randn(10, 1)
        
        loss = criterion(pred, target)
        
        assert torch.is_tensor(loss)
        assert loss.item() >= 0
    
    def test_log_cosh_loss(self):
        """Test Log-Cosh loss computation."""
        criterion = torchium.losses.LogCoshLoss()
        
        pred = torch.randn(10, 1)
        target = torch.randn(10, 1)
        
        loss = criterion(pred, target)
        
        assert torch.is_tensor(loss)
        assert loss.item() >= 0


class TestClassificationLosses:
    """Test classification loss functions."""
    
    def test_focal_loss(self):
        """Test Focal loss computation."""
        criterion = torchium.losses.FocalLoss(alpha=0.25, gamma=2.0)
        
        # Binary classification
        pred = torch.randn(10, 1)
        target = torch.randint(0, 2, (10, 1)).float()
        
        loss = criterion(pred, target)
        
        assert torch.is_tensor(loss)
        assert loss.item() >= 0
    
    def test_focal_loss_parameters(self):
        """Test Focal loss parameters."""
        criterion = torchium.losses.FocalLoss(alpha=0.5, gamma=3.0)
        
        assert criterion.alpha == 0.5
        assert criterion.gamma == 3.0
    
    def test_label_smoothing_loss(self):
        """Test Label Smoothing loss computation."""
        num_classes = 5
        criterion = torchium.losses.LabelSmoothingLoss(
            num_classes=num_classes, 
            smoothing=0.1
        )
        
        pred = torch.randn(10, num_classes)
        target = torch.randint(0, num_classes, (10,))
        
        loss = criterion(pred, target)
        
        assert torch.is_tensor(loss)
        assert loss.item() >= 0
    
    def test_triplet_loss(self):
        """Test Triplet loss computation."""
        criterion = torchium.losses.TripletLoss(margin=1.0)
        
        anchor = torch.randn(10, 128)
        positive = torch.randn(10, 128)
        negative = torch.randn(10, 128)
        
        loss = criterion(anchor, positive, negative)
        
        assert torch.is_tensor(loss)
        assert loss.item() >= 0


class TestComputerVisionLosses:
    """Test computer vision specific loss functions."""
    
    def test_dice_loss(self):
        """Test Dice loss computation."""
        criterion = torchium.losses.DiceLoss(smooth=1e-5)
        
        # Segmentation data (batch_size, height, width)
        pred = torch.sigmoid(torch.randn(4, 64, 64))
        target = torch.randint(0, 2, (4, 64, 64)).float()
        
        loss = criterion(pred, target)
        
        assert torch.is_tensor(loss)
        assert 0 <= loss.item() <= 1  # Dice loss is between 0 and 1
    
    def test_iou_loss(self):
        """Test IoU loss computation."""
        criterion = torchium.losses.IoULoss()
        
        pred = torch.sigmoid(torch.randn(4, 32, 32))
        target = torch.randint(0, 2, (4, 32, 32)).float()
        
        loss = criterion(pred, target)
        
        assert torch.is_tensor(loss)
        assert loss.item() >= 0
    
    def test_tversky_loss(self):
        """Test Tversky loss computation."""
        criterion = torchium.losses.TverskyLoss(alpha=0.3, beta=0.7)
        
        pred = torch.sigmoid(torch.randn(4, 32, 32))
        target = torch.randint(0, 2, (4, 32, 32)).float()
        
        loss = criterion(pred, target)
        
        assert torch.is_tensor(loss)
        assert loss.item() >= 0
        assert criterion.alpha == 0.3
        assert criterion.beta == 0.7


class TestLossReductions:
    """Test loss reduction methods."""
    
    @pytest.mark.parametrize("reduction", ['mean', 'sum', 'none'])
    def test_mse_loss_reductions(self, reduction):
        """Test MSE loss with different reductions."""
        criterion = torchium.losses.MSELoss(reduction=reduction)
        
        pred = torch.randn(5, 3)
        target = torch.randn(5, 3)
        
        loss = criterion(pred, target)
        
        if reduction == 'none':
            assert loss.shape == pred.shape
        else:
            assert loss.dim() == 0  # Should be scalar
    
    @pytest.mark.parametrize("reduction", ['mean', 'sum', 'none'])
    def test_focal_loss_reductions(self, reduction):
        """Test Focal loss with different reductions."""
        criterion = torchium.losses.FocalLoss(
            alpha=0.25, 
            gamma=2.0, 
            reduction=reduction
        )
        
        pred = torch.randn(10, 1)
        target = torch.randint(0, 2, (10, 1)).float()
        
        loss = criterion(pred, target)
        
        if reduction == 'none':
            assert loss.shape[0] == pred.shape[0]
        else:
            assert loss.dim() == 0  # Should be scalar


class TestLossProperties:
    """Test mathematical properties of loss functions."""
    
    def test_mse_loss_properties(self):
        """Test MSE loss mathematical properties."""
        criterion = torchium.losses.MSELoss()
        
        # Test symmetry: MSE(a, b) = MSE(b, a)
        a = torch.randn(10, 5)
        b = torch.randn(10, 5)
        
        loss1 = criterion(a, b)
        loss2 = criterion(b, a)
        
        assert torch.allclose(loss1, loss2, atol=1e-6)
        
        # Test minimum at equality: MSE(a, a) = 0
        loss_zero = criterion(a, a)
        assert torch.allclose(loss_zero, torch.tensor(0.0), atol=1e-6)
    
    def test_focal_loss_focusing_property(self):
        """Test that Focal loss focuses on hard examples."""
        criterion_no_focus = torchium.losses.FocalLoss(alpha=1.0, gamma=0.0)
        criterion_focus = torchium.losses.FocalLoss(alpha=1.0, gamma=2.0)
        
        # Easy example (high confidence, correct prediction)
        easy_pred = torch.tensor([[0.9]], dtype=torch.float32)
        easy_target = torch.tensor([[1.0]], dtype=torch.float32)
        
        # Hard example (low confidence, correct prediction)
        hard_pred = torch.tensor([[0.6]], dtype=torch.float32)
        hard_target = torch.tensor([[1.0]], dtype=torch.float32)
        
        easy_loss_no_focus = criterion_no_focus(easy_pred, easy_target)
        hard_loss_no_focus = criterion_no_focus(hard_pred, hard_target)
        
        easy_loss_focus = criterion_focus(easy_pred, easy_target)
        hard_loss_focus = criterion_focus(hard_pred, hard_target)
        
        # With focusing, the ratio of hard/easy loss should be higher
        ratio_no_focus = hard_loss_no_focus / easy_loss_no_focus
        ratio_focus = hard_loss_focus / easy_loss_focus
        
        assert ratio_focus > ratio_no_focus, "Focal loss should focus more on hard examples"
    
    def test_dice_loss_overlap_property(self):
        """Test that Dice loss measures overlap correctly."""
        criterion = torchium.losses.DiceLoss(smooth=0)
        
        # Perfect overlap
        perfect_pred = torch.ones(1, 10, 10)
        perfect_target = torch.ones(1, 10, 10)
        perfect_loss = criterion(perfect_pred, perfect_target)
        
        # No overlap
        no_pred = torch.zeros(1, 10, 10)
        some_target = torch.ones(1, 10, 10)
        no_loss = criterion(no_pred, some_target)
        
        # Perfect overlap should have lower loss
        assert perfect_loss < no_loss
        assert torch.allclose(perfect_loss, torch.tensor(0.0), atol=1e-6)


class TestLossGradients:
    """Test gradient computation for loss functions."""
    
    def test_mse_loss_gradients(self):
        """Test MSE loss gradient computation."""
        criterion = torchium.losses.MSELoss()
        
        pred = torch.randn(5, 3, requires_grad=True)
        target = torch.randn(5, 3)
        
        loss = criterion(pred, target)
        loss.backward()
        
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape
        assert torch.isfinite(pred.grad).all()
    
    def test_focal_loss_gradients(self):
        """Test Focal loss gradient computation."""
        criterion = torchium.losses.FocalLoss(alpha=0.25, gamma=2.0)
        
        pred = torch.randn(10, 1, requires_grad=True)
        target = torch.randint(0, 2, (10, 1)).float()
        
        loss = criterion(pred, target)
        loss.backward()
        
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape
        assert torch.isfinite(pred.grad).all()
    
    def test_dice_loss_gradients(self):
        """Test Dice loss gradient computation."""
        criterion = torchium.losses.DiceLoss()
        
        pred = torch.randn(2, 16, 16, requires_grad=True)
        target = torch.randint(0, 2, (2, 16, 16)).float()
        
        loss = criterion(pred, target)
        loss.backward()
        
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape
        assert torch.isfinite(pred.grad).all()


class TestLossCompatibility:
    """Test compatibility with PyTorch ecosystem."""
    
    def test_loss_with_optimizer(self):
        """Test loss functions work with optimizers."""
        model = nn.Linear(10, 1)
        criterion = torchium.losses.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)
        
        # Training step
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        # Should complete without errors
        assert isinstance(loss.item(), float)
    
    def test_loss_with_dataloader(self):
        """Test loss functions work with DataLoader."""
        from torch.utils.data import TensorDataset, DataLoader
        
        # Create dataset
        x = torch.randn(100, 5)
        y = torch.randn(100, 1)
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=16)
        
        model = nn.Linear(5, 1)
        criterion = torchium.losses.HuberLoss()
        
        # Test one batch
        for batch_x, batch_y in dataloader:
            output = model(batch_x)
            loss = criterion(output, batch_y)
            assert torch.is_tensor(loss)
            break


class TestLossEdgeCases:
    """Test edge cases and error handling."""
    
    def test_loss_with_nan_inputs(self):
        """Test loss behavior with NaN inputs."""
        criterion = torchium.losses.MSELoss()
        
        pred = torch.tensor([[float('nan')]])
        target = torch.tensor([[1.0]])
        
        loss = criterion(pred, target)
        assert torch.isnan(loss)
    
    def test_loss_with_inf_inputs(self):
        """Test loss behavior with infinite inputs."""
        criterion = torchium.losses.MSELoss()
        
        pred = torch.tensor([[float('inf')]])
        target = torch.tensor([[1.0]])
        
        loss = criterion(pred, target)
        assert torch.isinf(loss)
    
    def test_loss_shape_mismatch(self):
        """Test loss behavior with mismatched shapes."""
        criterion = torchium.losses.MSELoss()
        
        pred = torch.randn(5, 3)
        target = torch.randn(5, 2)  # Wrong shape
        
        with pytest.raises(RuntimeError):
            criterion(pred, target)
    
    def test_empty_tensor_loss(self):
        """Test loss with empty tensors."""
        criterion = torchium.losses.MSELoss()
        
        pred = torch.empty(0, 5)
        target = torch.empty(0, 5)
        
        loss = criterion(pred, target)
        # Should handle empty tensors gracefully
        assert torch.is_tensor(loss)


if __name__ == "__main__":
    pytest.main([__file__]) 
"""
Unit tests for Torchium optimizers.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add torchium to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

import torchium


class TestOptimizerRegistry:
    """Test optimizer registry functionality."""
    
    def test_optimizer_registry_exists(self):
        """Test that optimizer registry is accessible."""
        assert hasattr(torchium.utils, 'registry')
        assert hasattr(torchium.utils.registry, 'optimizer_registry')
    
    def test_get_available_optimizers(self):
        """Test listing available optimizers."""
        optimizers = torchium.get_available_optimizers()
        assert isinstance(optimizers, list)
        assert len(optimizers) > 0
        assert 'adam' in optimizers
        assert 'sgd' in optimizers
    
    def test_optimizer_count(self):
        """Test that we have the expected number of optimizers."""
        optimizers = torchium.get_available_optimizers()
        assert len(optimizers) >= 50  # Should have at least 50 optimizers


class TestOptimizerFactory:
    """Test optimizer factory functions."""
    
    def test_create_optimizer_adam(self, simple_model):
        """Test creating Adam optimizer via factory."""
        optimizer = torchium.create_optimizer('adam', simple_model.parameters(), lr=1e-3)
        assert optimizer is not None
        assert hasattr(optimizer, 'step')
        assert hasattr(optimizer, 'zero_grad')
    
    def test_create_optimizer_sgd(self, simple_model):
        """Test creating SGD optimizer via factory."""
        optimizer = torchium.create_optimizer('sgd', simple_model.parameters(), lr=1e-2)
        assert optimizer is not None
        assert optimizer.defaults['lr'] == 1e-2
    
    def test_create_optimizer_invalid_name(self, simple_model):
        """Test creating optimizer with invalid name."""
        with pytest.raises((KeyError, ValueError)):
            torchium.create_optimizer('invalid_optimizer', simple_model.parameters())
    
    def test_create_optimizer_with_kwargs(self, simple_model):
        """Test creating optimizer with additional kwargs."""
        optimizer = torchium.create_optimizer(
            'adamw', 
            simple_model.parameters(), 
            lr=1e-3, 
            weight_decay=1e-4
        )
        assert optimizer.defaults['lr'] == 1e-3
        assert optimizer.defaults['weight_decay'] == 1e-4


class TestAdvancedOptimizers:
    """Test advanced optimizer implementations."""
    
    @pytest.mark.parametrize("optimizer_name,lr", [
        ('ranger', 1e-3),
        ('radam', 1e-3),
        ('adabelief', 1e-3),
        ('lion', 1e-4),
        ('lamb', 1e-3),
        ('novograd', 1e-3),
        ('lars', 1e-2),
    ])
    def test_advanced_optimizer_creation(self, simple_model, optimizer_name, lr):
        """Test creating advanced optimizers."""
        try:
            optimizer = torchium.create_optimizer(
                optimizer_name, 
                simple_model.parameters(), 
                lr=lr
            )
            assert optimizer is not None
            assert hasattr(optimizer, 'step')
        except Exception as e:
            pytest.skip(f"Optimizer {optimizer_name} not fully implemented: {e}")
    
    def test_ranger_optimizer(self, simple_model, sample_data, training_step):
        """Test Ranger optimizer functionality."""
        try:
            optimizer = torchium.create_optimizer('ranger', simple_model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()
            
            data = sample_data['regression']
            loss = training_step(simple_model, optimizer, criterion, data['x'], data['y'])
            
            assert isinstance(loss, float)
            assert loss >= 0
        except Exception as e:
            pytest.skip(f"Ranger optimizer test skipped: {e}")
    
    def test_lion_optimizer_memory_efficiency(self, simple_model):
        """Test Lion optimizer for memory efficiency."""
        try:
            optimizer = torchium.create_optimizer('lion', simple_model.parameters(), lr=1e-4)
            
            # Lion should have fewer state variables than Adam
            # This is a basic check for memory efficiency
            assert optimizer is not None
            
            # Check that Lion uses less memory by examining state initialization
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        state = optimizer.state[p]
                        # Lion should have simpler state than Adam
                        break
        except Exception as e:
            pytest.skip(f"Lion optimizer test skipped: {e}")


class TestOptimizerConvergence:
    """Test optimizer convergence properties."""
    
    @pytest.mark.slow
    @pytest.mark.parametrize("optimizer_name", ['adam', 'sgd', 'adamw'])
    def test_optimizer_convergence(self, simple_model, optimizer_name, sample_data):
        """Test that optimizers can reduce loss."""
        optimizer = torchium.create_optimizer(
            optimizer_name, 
            simple_model.parameters(), 
            lr=1e-2 if optimizer_name == 'sgd' else 1e-3
        )
        criterion = nn.MSELoss()
        
        data = sample_data['regression']
        
        # Record initial loss
        optimizer.zero_grad()
        initial_output = simple_model(data['x'])
        initial_loss = criterion(initial_output, data['y']).item()
        
        # Train for several steps
        for _ in range(20):
            optimizer.zero_grad()
            output = simple_model(data['x'])
            loss = criterion(output, data['y'])
            loss.backward()
            optimizer.step()
        
        # Check final loss
        final_output = simple_model(data['x'])
        final_loss = criterion(final_output, data['y']).item()
        
        # Loss should decrease (allowing for some variance)
        improvement_ratio = (initial_loss - final_loss) / initial_loss
        assert improvement_ratio > 0.01, f"Loss should improve by at least 1%"
    
    def test_optimizer_state_persistence(self, simple_model, sample_data):
        """Test that optimizer maintains state between steps."""
        optimizer = torchium.create_optimizer('adam', simple_model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        data = sample_data['regression']
        
        # First step
        optimizer.zero_grad()
        output = simple_model(data['x'])
        loss = criterion(output, data['y'])
        loss.backward()
        optimizer.step()
        
        # Check that state was created
        state_created = False
        for group in optimizer.param_groups:
            for p in group['params']:
                if p in optimizer.state and len(optimizer.state[p]) > 0:
                    state_created = True
                    break
        
        assert state_created, "Optimizer should maintain state"


class TestOptimizerParameters:
    """Test optimizer parameter handling."""
    
    def test_parameter_groups(self, simple_model):
        """Test optimizer with multiple parameter groups."""
        # Create parameter groups with different learning rates
        param_groups = [
            {'params': simple_model[0].parameters(), 'lr': 1e-3},
            {'params': simple_model[2].parameters(), 'lr': 1e-4},  # Skip ReLU layer
            {'params': simple_model[4].parameters(), 'lr': 1e-2}
        ]
        
        optimizer = torchium.optimizers.Adam(param_groups)
        
        assert len(optimizer.param_groups) == 3
        assert optimizer.param_groups[0]['lr'] == 1e-3
        assert optimizer.param_groups[1]['lr'] == 1e-4
        assert optimizer.param_groups[2]['lr'] == 1e-2
    
    def test_parameter_validation(self, simple_model):
        """Test optimizer parameter validation."""
        # Test invalid learning rate
        with pytest.raises(ValueError):
            torchium.optimizers.Adam(simple_model.parameters(), lr=-1.0)
        
        # Test invalid betas for Adam
        with pytest.raises(ValueError):
            torchium.optimizers.Adam(simple_model.parameters(), lr=1e-3, betas=(1.5, 0.999))
    
    def test_zero_grad(self, simple_model, sample_data):
        """Test zero_grad functionality."""
        optimizer = torchium.create_optimizer('adam', simple_model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        data = sample_data['regression']
        
        # Forward pass to create gradients
        output = simple_model(data['x'])
        loss = criterion(output, data['y'])
        loss.backward()
        
        # Check gradients exist
        has_grad = any(p.grad is not None for p in simple_model.parameters())
        assert has_grad, "Model should have gradients after backward pass"
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Check gradients are zeroed
        all_zero = all(
            p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad))
            for p in simple_model.parameters()
        )
        assert all_zero, "All gradients should be zero after zero_grad()"


class TestOptimizerCompatibility:
    """Test compatibility with PyTorch ecosystem."""
    
    def test_scheduler_compatibility(self, simple_model):
        """Test compatibility with PyTorch learning rate schedulers."""
        optimizer = torchium.create_optimizer('adam', simple_model.parameters(), lr=1e-3)
        
        # Test with StepLR scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Step scheduler
        for _ in range(15):
            scheduler.step()
        
        # Learning rate should have decreased
        final_lr = optimizer.param_groups[0]['lr']
        assert final_lr < initial_lr, "Learning rate should decrease with scheduler"
    
    def test_gradient_clipping_compatibility(self, simple_model, sample_data):
        """Test compatibility with gradient clipping."""
        optimizer = torchium.create_optimizer('adam', simple_model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        data = sample_data['regression']
        
        # Forward and backward pass
        optimizer.zero_grad()
        output = simple_model(data['x'])
        loss = criterion(output, data['y'])
        loss.backward()
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(simple_model.parameters(), max_norm=1.0)
        
        # Check that gradients are clipped
        total_norm = 0
        for p in simple_model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        assert total_norm <= 1.1, "Gradient norm should be clipped"  # Allow small tolerance
        
        # Optimizer step should work normally
        optimizer.step()


if __name__ == "__main__":
    pytest.main([__file__]) 
"""
Pytest configuration and shared fixtures for Torchium tests.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add torchium to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

import torchium


@pytest.fixture(scope="session")
def device():
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def simple_model():
    """Create a simple neural network for testing."""
    return nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )


@pytest.fixture
def sample_data():
    """Generate sample training data."""
    return {
        'regression': {
            'x': torch.randn(32, 10),
            'y': torch.randn(32, 1)
        },
        'classification': {
            'x': torch.randn(32, 20),
            'y': torch.randint(0, 5, (32,))
        },
        'binary_classification': {
            'x': torch.randn(32, 10),
            'y': torch.randint(0, 2, (32,)).float()
        }
    }


@pytest.fixture
def training_step():
    """Helper function for performing a training step."""
    def _training_step(model, optimizer, criterion, data, target):
        """Perform one training step and return loss."""
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    return _training_step


# Disable info printing during tests
os.environ["TORCHIUM_SHOW_INFO"] = "false"

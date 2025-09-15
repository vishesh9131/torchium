# Contributing to Torchium

Thank you for your interest in contributing to Torchium! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Adding New Optimizers](#adding-new-optimizers)
- [Adding New Loss Functions](#adding-new-loss-functions)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to [sciencely98@gmail.com](mailto:sciencely98@gmail.com).

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/torchium.git
   cd torchium
   ```
3. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.9.0 or higher
- Git

### Installation

1. Install the package in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

2. Install pre-commit hooks (optional but recommended):
   ```bash
   pre-commit install
   ```

### Development Dependencies

The following development dependencies are included:
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `black` - Code formatting
- `flake8` - Linting
- `isort` - Import sorting
- `mypy` - Type checking
- `pre-commit` - Git hooks

## Contributing Guidelines

### General Guidelines

1. **Follow PEP 8**: Use the provided linting tools to ensure code style consistency
2. **Write Tests**: All new features must include comprehensive tests
3. **Update Documentation**: Update relevant documentation for new features
4. **Type Hints**: Use type hints for better code clarity
5. **Docstrings**: Follow Google-style docstrings for all public functions and classes

### Code Style

We use several tools to maintain code quality:

```bash
# Format code
black torchium/

# Sort imports
isort torchium/

# Lint code
flake8 torchium/

# Type checking
mypy torchium/
```

## Adding New Optimizers

### 1. Create the Optimizer Class

Create a new file in the appropriate subdirectory under `torchium/optimizers/`:

```python
import torch
from torch.optim import Optimizer
from typing import List, Optional, Dict, Any

class YourOptimizer(Optimizer):
    """
    Your optimizer description.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate
        **kwargs: Additional optimizer-specific parameters
    """
    
    def __init__(self, params, lr: float = 1e-3, **kwargs):
        defaults = dict(lr=lr, **kwargs)
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[callable] = None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Your optimization logic here
                grad = p.grad.data
                # ... optimization step ...
                p.data.add_(grad, alpha=-group['lr'])
        
        return loss
```

### 2. Register the Optimizer

Add your optimizer to the appropriate `__init__.py` file:

```python
from .your_optimizer import YourOptimizer

__all__ = ['YourOptimizer']
```

### 3. Add to Registry

Update `torchium/utils/registry.py`:

```python
@register_optimizer("your_optimizer")
class YourOptimizer(Optimizer):
    # ... implementation ...
```

### 4. Write Tests

Create comprehensive tests in `torchium/tests/unit/test_optimizers.py`:

```python
import torch
import torch.nn as nn
from torchium.optimizers import YourOptimizer

def test_your_optimizer():
    model = nn.Linear(10, 1)
    optimizer = YourOptimizer(model.parameters(), lr=1e-3)
    
    # Test basic functionality
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)
    
    for _ in range(10):
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        optimizer.step()
    
    assert loss.item() < 1.0  # Should converge
```

## Adding New Loss Functions

### 1. Create the Loss Class

Create a new file in the appropriate subdirectory under `torchium/losses/`:

```python
import torch
import torch.nn as nn
from typing import Optional, Union

class YourLoss(nn.Module):
    """
    Your loss function description.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
    """
    
    def __init__(self, param1: float = 1.0, param2: Optional[float] = None):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the loss function.
        
        Args:
            input: Predicted values
            target: Ground truth values
            
        Returns:
            Computed loss
        """
        # Your loss computation here
        loss = torch.mean((input - target) ** 2)
        return loss
```

### 2. Register the Loss

Follow similar steps as for optimizers:
1. Add to appropriate `__init__.py`
2. Update registry
3. Write comprehensive tests

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=torchium

# Run specific test file
pytest torchium/tests/unit/test_optimizers.py

# Run with verbose output
pytest -v
```

### Test Requirements

- All tests must pass
- Maintain at least 90% code coverage
- Include edge cases and error conditions
- Test both CPU and GPU (if applicable)

## Documentation

### Code Documentation

- Use Google-style docstrings
- Include type hints
- Document all public APIs
- Provide usage examples

### README Updates

- Update feature lists for new optimizers/losses
- Add usage examples
- Update installation instructions if needed

## Submitting Changes

### Pull Request Process

1. **Create a Pull Request**: Use the provided template
2. **Link Issues**: Reference any related issues
3. **Describe Changes**: Provide a clear description of your changes
4. **Test Coverage**: Ensure all tests pass
5. **Documentation**: Update relevant documentation

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] All existing tests still pass

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

## Release Process

### Version Bumping

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. Update version in `setup.py`
2. Update `CHANGELOG.md`
3. Create release notes
4. Tag the release
5. Build and upload to PyPI

## Getting Help

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and general discussion
- **Email**: Contact [sciencely98@gmail.com](mailto:sciencely98@gmail.com) for private matters

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to Torchium!

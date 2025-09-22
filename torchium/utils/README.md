# Torchium Utils

This module contains utility functions and CUDA optimizations for Torchium.

## CUDA Kernels (`cuda_kernels.py`)

The CUDA kernels module provides high-performance GPU operations for optimizers and loss functions.

### Features

- **Matrix Operations**: CUDA-optimized eigendecomposition and matrix powers
- **Gradient Operations**: Per-sample gradient computation using functorch
- **Memory Management**: Efficient memory allocation and chunked operations
- **Device Utilities**: CUDA availability checking and memory monitoring

### Usage

```python
from torchium.utils.cuda_kernels import CUDAMatrixOps, CUDAGradientOps

# Matrix operations
G_sqrt_inv = CUDAMatrixOps.matrix_sqrt_inv_eigen(G, power=-0.25)

# Per-sample gradients
per_sample_grads = CUDAGradientOps.per_sample_gradients(model, loss_fn, inputs, targets)
```

### Performance Benefits

- **Shampoo**: 2-3x faster matrix square root computation
- **KFAC**: Efficient Kronecker product operations
- **Natural Gradient**: Accurate per-sample gradient computation
- **Memory**: Automatic OOM handling with chunked operations

## Factory Functions (`factory.py`)

Factory functions for creating optimizers and loss functions by name.

## Registry (`registry.py`)

Registry system for managing available optimizers and loss functions.

## Validation (`validation.py`)

Input validation utilities for optimizers and loss functions.

## Compatibility (`compatibility.py`)

Compatibility utilities for different PyTorch versions and devices.

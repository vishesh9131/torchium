CUDA Integration and Custom Kernels
====================================

Torchium provides comprehensive CUDA integration for high-performance optimization algorithms. This guide covers how to integrate custom C++/CUDA kernels and optimize performance-critical operations.

Overview
--------

Torchium's CUDA integration includes:

- **Custom CUDA kernels** for matrix operations (e.g., Shampoo's matrix square roots)
- **Per-sample gradient computation** using functorch and custom autograd functions
- **Memory-efficient operations** for large-scale optimization
- **Automatic fallbacks** to CPU implementations when CUDA is unavailable

CUDA Kernel Architecture
------------------------

The CUDA integration is organized into several modules:

.. code-block:: python

    from torchium.utils.cuda_kernels import (
        CUDAMatrixOps,      # Matrix operations
        CUDAGradientOps,    # Gradient computations
        CUDAMemoryOps,      # Memory management
        is_cuda_available,  # Device utilities
        get_optimal_device
    )

Matrix Operations
-----------------

CUDA-optimized matrix operations are essential for second-order optimizers like Shampoo and KFAC.

Shampoo Matrix Square Roots
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shampoo requires computing matrix powers like :math:`G^{-1/4}`. The CUDA implementation uses eigendecomposition:

.. code-block:: python

    import torch
    from torchium.utils.cuda_kernels import CUDAMatrixOps
    
    # Create a symmetric matrix
    G = torch.randn(100, 100, device='cuda')
    G = G @ G.t()  # Make symmetric
    
    # Compute G^(-1/4) using CUDA optimization
    G_sqrt_inv = CUDAMatrixOps.matrix_sqrt_inv_eigen(
        G, 
        power=-0.25,  # -1/4 power
        eps=1e-8      # Numerical stability
    )

KFAC Kronecker Products
~~~~~~~~~~~~~~~~~~~~~~~

KFAC uses Kronecker products for efficient natural gradient computation:

.. code-block:: python

    from torchium.utils.cuda_kernels import CUDAMatrixOps
    
    # Input and output covariance matrices
    A = torch.randn(50, 50, device='cuda')
    G = torch.randn(100, 100, device='cuda')
    
    # Efficient Kronecker product approximation
    kron_product = CUDAMatrixOps.kronecker_product_approx(A, G)

Per-Sample Gradients
--------------------

Natural gradient methods require per-sample gradients for accurate Fisher Information Matrix estimation.

Using functorch (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    import torch.nn as nn
    from torchium.utils.cuda_kernels import CUDAGradientOps
    
    # Create model and data
    model = nn.Linear(10, 1).cuda()
    inputs = torch.randn(32, 10, device='cuda')
    targets = torch.randn(32, 1, device='cuda')
    loss_fn = nn.MSELoss()
    
    # Compute per-sample gradients
    per_sample_grads = CUDAGradientOps.per_sample_gradients(
        model, loss_fn, inputs, targets
    )

Custom Autograd Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

For more control, you can create custom autograd functions:

.. code-block:: python

    class PerSampleGradFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias):
            ctx.save_for_backward(input, weight, bias)
            return torch.nn.functional.linear(input, weight, bias)
        
        @staticmethod
        def backward(ctx, grad_output):
            input, weight, bias = ctx.saved_tensors
            
            # Compute per-sample gradients
            per_sample_grads = []
            for i in range(input.shape[0]):
                sample_input = input[i:i+1]
                sample_grad = grad_output[i:i+1]
                
                # Compute gradient for this sample
                grad_weight = sample_grad.t() @ sample_input
                per_sample_grads.append(grad_weight)
            
            return None, torch.stack(per_sample_grads), None

Memory Management
-----------------

CUDA memory management is crucial for large-scale optimization.

Memory-Efficient Matrix Multiplication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from torchium.utils.cuda_kernels import CUDAMemoryOps
    
    # Large matrices that might cause OOM
    A = torch.randn(5000, 5000, device='cuda')
    B = torch.randn(5000, 5000, device='cuda')
    
    # Memory-efficient multiplication with automatic chunking
    result = CUDAMemoryOps.memory_efficient_matmul(A, B)

Memory Information
~~~~~~~~~~~~~~~~~~

Monitor CUDA memory usage:

.. code-block:: python

    from torchium.utils.cuda_kernels import cuda_memory_info
    
    memory_info = cuda_memory_info()
    print(f"Total memory: {memory_info['total_memory'] / 1e9:.2f} GB")
    print(f"Allocated: {memory_info['allocated_memory'] / 1e9:.2f} GB")
    print(f"Free: {memory_info['free_memory'] / 1e9:.2f} GB")

Custom C++/CUDA Kernels
-----------------------

For maximum performance, you can integrate custom C++/CUDA kernels.

Setting Up Custom Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Create CUDA kernel files**:

.. code-block:: cuda

    // custom_kernels.cu
    #include <torch/extension.h>
    #include <cuda_runtime.h>
    
    __global__ void matrix_sqrt_kernel(
        const float* input,
        float* output,
        int size,
        float power
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            output[idx] = powf(input[idx], power);
        }
    }
    
    torch::Tensor matrix_sqrt_cuda(torch::Tensor input, float power) {
        auto output = torch::zeros_like(input);
        
        int threads = 256;
        int blocks = (input.numel() + threads - 1) / threads;
        
        matrix_sqrt_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            input.numel(),
            power
        );
        
        return output;
    }
    
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("matrix_sqrt_cuda", &matrix_sqrt_cuda, "Matrix square root CUDA kernel");
    }

2. **Create Python wrapper**:

.. code-block:: python

    # custom_kernels.py
    import torch
    from torch.utils.cpp_extension import load
    
    # Load the CUDA extension
    custom_kernels = load(
        name="custom_kernels",
        sources=["custom_kernels.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"]
    )
    
    def matrix_sqrt_cuda_optimized(matrix, power=-0.25):
        """CUDA-optimized matrix square root."""
        if matrix.is_cuda:
            return custom_kernels.matrix_sqrt_cuda(matrix, power)
        else:
            # Fallback to CPU
            return torch.pow(matrix, power)

3. **Integrate with Torchium**:

.. code-block:: python

    # In your optimizer
    try:
        from .custom_kernels import matrix_sqrt_cuda_optimized
        CUSTOM_KERNELS_AVAILABLE = True
    except ImportError:
        CUSTOM_KERNELS_AVAILABLE = False
    
    class OptimizedShampoo(Optimizer):
        def step(self, closure=None):
            # ... existing code ...
            
            if CUSTOM_KERNELS_AVAILABLE and G_l.is_cuda:
                G_l_sqrt_inv = matrix_sqrt_cuda_optimized(G_l, -0.25)
            else:
                # Fallback to standard implementation
                G_l_sqrt_inv = CUDAMatrixOps.matrix_sqrt_inv_eigen(G_l, -0.25)

Performance Optimization Tips
-----------------------------

1. **Use Mixed Precision Training**:

.. code-block:: python

    from torch.cuda.amp import autocast, GradScaler
    
    scaler = GradScaler()
    
    with autocast():
        output = model(input)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

2. **Optimize Memory Layout**:

.. code-block:: python

    # Use contiguous memory format
    tensor = tensor.contiguous(memory_format=torch.channels_last)
    
    # Efficient tensor creation
    tensor = CUDAMemoryOps.efficient_tensor_creation(
        shape=(1000, 1000),
        device=torch.device('cuda'),
        dtype=torch.float32
    )

3. **Batch Operations**:

.. code-block:: python

    # Batch multiple small operations
    results = CUDAMatrixOps.batch_matrix_multiply(
        A_batch,  # [batch_size, m, k]
        B_batch   # [batch_size, k, n]
    )

Troubleshooting
---------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **CUDA Out of Memory**:
   - Use gradient checkpointing
   - Reduce batch size
   - Use memory-efficient operations
   - Enable memory pooling

2. **Kernel Compilation Errors**:
   - Check CUDA version compatibility
   - Ensure proper include paths
   - Use appropriate compiler flags

3. **Performance Issues**:
   - Profile with `torch.profiler`
   - Check memory bandwidth utilization
   - Optimize kernel launch parameters

Example: Complete CUDA-Optimized Optimizer
------------------------------------------

Here's a complete example of a CUDA-optimized optimizer:

.. code-block:: python

    import torch
    import torch.nn as nn
    from torch.optim.optimizer import Optimizer
    from torchium.utils.cuda_kernels import CUDAMatrixOps, CUDAMemoryOps
    
    class CUDAShampoo(Optimizer):
        def __init__(self, params, lr=0.03, eps=1e-4, update_freq=100):
            defaults = dict(lr=lr, eps=eps, update_freq=update_freq)
            super().__init__(params, defaults)
        
        def step(self, closure=None):
            loss = None
            if closure is not None:
                loss = closure()
            
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    grad = p.grad.data
                    state = self.state[p]
                    
                    if len(state) == 0:
                        state['step'] = 0
                        if len(p.shape) >= 2:
                            # Use efficient tensor creation
                            state['G_l'] = CUDAMemoryOps.efficient_tensor_creation(
                                (p.shape[0], p.shape[0]), p.device, p.dtype
                            )
                            state['G_r'] = CUDAMemoryOps.efficient_tensor_creation(
                                (p.shape[1], p.shape[1]), p.device, p.dtype
                            )
                    
                    state['step'] += 1
                    
                    if len(p.shape) >= 2:
                        G_l, G_r = state['G_l'], state['G_r']
                        
                        # Update preconditioners
                        G_l.add_(torch.mm(grad, grad.t()))
                        G_r.add_(torch.mm(grad.t(), grad))
                        
                        if state['step'] % group['update_freq'] == 0:
                            # CUDA-optimized matrix operations
                            G_l_sqrt_inv = CUDAMatrixOps.matrix_sqrt_inv_eigen(
                                G_l, power=-0.25, eps=group['eps']
                            )
                            G_r_sqrt_inv = CUDAMatrixOps.matrix_sqrt_inv_eigen(
                                G_r, power=-0.25, eps=group['eps']
                            )
                            
                            # Memory-efficient matrix multiplication
                            search_direction = CUDAMemoryOps.memory_efficient_matmul(
                                CUDAMemoryOps.memory_efficient_matmul(G_l_sqrt_inv, grad),
                                G_r_sqrt_inv
                            )
                        else:
                            search_direction = grad
                    
                    p.data.add_(search_direction, alpha=-group['lr'])
            
            return loss

This example demonstrates how to integrate CUDA optimizations into a complete optimizer implementation.

Best Practices
--------------

1. **Always provide CPU fallbacks** for maximum compatibility
2. **Use proper error handling** for CUDA operations
3. **Profile performance** to identify bottlenecks
4. **Test on multiple GPU architectures** for portability
5. **Document memory requirements** and performance characteristics

For more advanced CUDA integration examples, see the `examples/` directory in the Torchium repository.

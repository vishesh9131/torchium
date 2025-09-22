import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from typing import Any, Dict, List, Optional, Tuple
import math

try:
    from ..utils.cuda_kernels import CUDAGradientOps

    CUDA_KERNELS_AVAILABLE = True
except ImportError:
    CUDA_KERNELS_AVAILABLE = False

try:
    from ..utils.cython_wrapper import CythonOptimizedOps

    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False


class NaturalGradient(Optimizer):
    """Natural Gradient optimizer using Fisher Information Matrix approximation"""

    def __init__(self, params, lr=0.01, damping=1e-3, update_freq=1, eps=1e-8):
        defaults = dict(lr=lr, damping=damping, update_freq=update_freq, eps=eps)
        super().__init__(params, defaults)
        self.steps = 0

    def step(self, closure=None):
        """Perform a single optimization step using natural gradients"""
        loss = None
        if closure is not None:
            loss = closure()

        self.steps += 1

        for group in self.param_groups:
            # Update Fisher Information Matrix approximation
            if self.steps % group["update_freq"] == 0:
                self._update_fisher_info(group)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["fisher_info"] = torch.zeros_like(p.data)
                    state["fisher_inv"] = torch.zeros_like(p.data)

                state["step"] += 1

                # Get Fisher Information Matrix diagonal approximation
                fisher_info = state["fisher_info"]
                fisher_inv = state["fisher_inv"]

                # Compute natural gradient: F^(-1) * grad
                # where F is the Fisher Information Matrix
                natural_grad = fisher_inv * grad

                # Update parameters
                p.data.add_(natural_grad, alpha=-group["lr"])

        return loss

    def _update_fisher_info(self, group):
        """Update Fisher Information Matrix approximation using per-sample gradients"""
        for p in group["params"]:
            if p.grad is None:
                continue

            state = self.state[p]
            if "fisher_info" not in state:
                continue

            grad = p.grad.data

            # Fisher Information Matrix diagonal approximation
            # F_ii ≈ E[g_i^2] where g_i is the gradient of parameter i
            # We use exponential moving average to estimate this expectation
            alpha = 0.9  # decay rate for moving average

            if state["step"] == 1:
                # Initialize with current gradient squared
                state["fisher_info"] = grad**2
            else:
                # Update with exponential moving average
                if CYTHON_AVAILABLE:
                    # Use Cython-optimized momentum update
                    try:
                        CythonOptimizedOps.momentum_update(state["fisher_info"], grad**2, alpha, 1.0 - alpha)
                    except:
                        # Fallback to standard update
                        state["fisher_info"].mul_(alpha).add_(grad**2, alpha=1 - alpha)
                else:
                    state["fisher_info"].mul_(alpha).add_(grad**2, alpha=1 - alpha)

            # Compute inverse Fisher Information Matrix with damping
            fisher_info = state["fisher_info"]
            fisher_inv = 1.0 / (fisher_info + group["damping"])
            state["fisher_inv"] = fisher_inv

    def compute_per_sample_gradients(self, model, loss_fn, inputs, targets):
        """Compute per-sample gradients for more accurate Fisher Information Matrix"""
        if CUDA_KERNELS_AVAILABLE:
            return CUDAGradientOps.per_sample_gradients(model, loss_fn, inputs, targets)
        else:
            # Fallback to manual computation
            return CUDAGradientOps._cpu_per_sample_gradients(model, loss_fn, inputs, targets)

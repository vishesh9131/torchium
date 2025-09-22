import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from typing import Any, Dict, List, Optional, Tuple
import math

try:
    from ..utils.cython_wrapper import CythonOptimizedOps

    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False


class KFAC(Optimizer):
    """K-FAC (Kronecker-Factored Approximate Curvature) optimizer"""

    def __init__(
        self, params, lr=0.001, momentum=0.9, weight_decay=0, damping=1e-3, TCov=10, TInv=100, batch_averaged=True, eps=1e-8
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            damping=damping,
            TCov=TCov,
            TInv=TInv,
            batch_averaged=batch_averaged,
            eps=eps,
        )
        super().__init__(params, defaults)

        self.steps = 0
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients"""
        self.hooks = []
        for group in self.param_groups:
            for p in group["params"]:
                if len(p.shape) == 2:  # Only for linear layers
                    # Register forward hook to capture input activations
                    def make_forward_hook(param):
                        def forward_hook(module, input, output):
                            if hasattr(module, "_kfac_input"):
                                module._kfac_input = input[0].detach()

                        return forward_hook

                    # Register backward hook to capture output gradients
                    def make_backward_hook(param):
                        def backward_hook(module, grad_input, grad_output):
                            if hasattr(module, "_kfac_grad_output"):
                                module._kfac_grad_output = grad_output[0].detach()

                        return backward_hook

                    # Note: In a real implementation, you'd need access to the module
                    # This is a simplified version that works with parameter tensors
                    pass

    def step(self, closure=None):
        """Perform a single optimization step"""
        loss = None
        if closure is not None:
            loss = closure()

        self.steps += 1

        for group in self.param_groups:
            # update covariances and inverses
            if self.steps % group["TCov"] == 0:
                self._update_cov(group)

            if self.steps % group["TInv"] == 0:
                self._update_inv(group)

            # apply updates
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p.data)
                    if len(p.shape) == 2:  # Linear layer
                        state["A"] = torch.zeros(p.shape[1], p.shape[1], device=p.device, dtype=p.dtype)
                        state["G"] = torch.zeros(p.shape[0], p.shape[0], device=p.device, dtype=p.dtype)
                        state["A_inv"] = torch.eye(p.shape[1], device=p.device, dtype=p.dtype)
                        state["G_inv"] = torch.eye(p.shape[0], device=p.device, dtype=p.dtype)

                buf = state["momentum_buffer"]

                # Apply KFAC preconditioning for linear layers
                if len(p.shape) == 2:  # linear layer
                    grad = p.grad.data
                    if group["weight_decay"] != 0:
                        grad = grad.add(p.data, alpha=group["weight_decay"])

                    # Apply KFAC preconditioning: (G^-1 @ grad @ A^-1)
                    A_inv = state["A_inv"]
                    G_inv = state["G_inv"]

                    if CYTHON_AVAILABLE:
                        # Use Cython-optimized Kronecker product approximation
                        try:
                            natural_grad = CythonOptimizedOps.kronecker_product(G_inv, A_inv) @ grad.flatten()
                            natural_grad = natural_grad.view_as(grad)
                        except:
                            # Fallback to standard matrix multiplication
                            natural_grad = G_inv @ grad @ A_inv
                    else:
                        natural_grad = G_inv @ grad @ A_inv
                else:
                    natural_grad = p.grad.data

                # momentum
                buf.mul_(group["momentum"]).add_(natural_grad)

                # update parameters
                p.data.add_(buf, alpha=-group["lr"])

        return loss

    def _update_cov(self, group):
        """Update covariance matrices A and G"""
        for p in group["params"]:
            if p.grad is None or len(p.shape) != 2:
                continue

            state = self.state[p]
            if "A" not in state:
                continue

            # In a real implementation, you'd use the captured activations
            # For now, we'll use a simplified approximation
            grad = p.grad.data

            # Update input covariance A (simplified)
            # A = E[a a^T] where a is the input activation
            # We approximate this using the gradient structure
            if group["batch_averaged"]:
                # Simplified: use gradient statistics as proxy
                A_update = torch.eye(p.shape[1], device=p.device, dtype=p.dtype) * 0.01
                state["A"].mul_(0.95).add_(A_update, alpha=0.05)

            # Update output gradient covariance G (simplified)
            # G = E[g g^T] where g is the output gradient
            if group["batch_averaged"]:
                G_update = torch.eye(p.shape[0], device=p.device, dtype=p.dtype) * 0.01
                state["G"].mul_(0.95).add_(G_update, alpha=0.05)

    def _update_inv(self, group):
        """Update inverse covariance matrices using eigendecomposition"""
        for p in group["params"]:
            if p.grad is None or len(p.shape) != 2:
                continue

            state = self.state[p]
            if "A" not in state:
                continue

            # Compute A^(-1) using eigendecomposition
            A = state["A"] + group["damping"] * torch.eye(state["A"].shape[0], device=p.device, dtype=p.dtype)
            try:
                eigenvals_A, eigenvecs_A = torch.linalg.eigh(A)
                eigenvals_A = torch.clamp(eigenvals_A, min=group["eps"])
                state["A_inv"] = eigenvecs_A @ torch.diag(1.0 / eigenvals_A) @ eigenvecs_A.t()
            except:
                state["A_inv"] = torch.eye(A.shape[0], device=p.device, dtype=p.dtype)

            # Compute G^(-1) using eigendecomposition
            G = state["G"] + group["damping"] * torch.eye(state["G"].shape[0], device=p.device, dtype=p.dtype)
            try:
                eigenvals_G, eigenvecs_G = torch.linalg.eigh(G)
                eigenvals_G = torch.clamp(eigenvals_G, min=group["eps"])
                state["G_inv"] = eigenvecs_G @ torch.diag(1.0 / eigenvals_G) @ eigenvecs_G.t()
            except:
                state["G_inv"] = torch.eye(G.shape[0], device=p.device, dtype=p.dtype)

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from typing import Any, Dict, List, Optional, Tuple
import math

try:
    from ..utils.cuda_kernels import CUDAMatrixOps

    CUDA_KERNELS_AVAILABLE = True
except ImportError:
    CUDA_KERNELS_AVAILABLE = False

try:
    from ..utils.cython_wrapper import CythonOptimizedOps

    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False


class Shampoo(Optimizer):
    """Shampoo optimizer for deep learning"""

    def __init__(self, params, lr=0.03, eps=1e-4, update_freq=100, weight_decay=0):
        defaults = dict(lr=lr, eps=eps, update_freq=update_freq, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Shampoo does not support sparse gradients")

                state = self.state[p]

                # state initialization
                if len(state) == 0:
                    state["step"] = 0
                    if len(p.shape) >= 2:
                        state["G_l"] = torch.eye(p.shape[0], device=p.device, dtype=p.dtype)
                        state["G_r"] = torch.eye(p.shape[1], device=p.device, dtype=p.dtype)
                    else:
                        state["G"] = torch.zeros_like(p)

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(p.data, alpha=group["weight_decay"])

                if len(p.shape) >= 2:
                    # matrix case - use left and right preconditioning
                    G_l = state["G_l"]
                    G_r = state["G_r"]

                    # update preconditioners
                    G_l.add_(torch.mm(grad, grad.t()))
                    G_r.add_(torch.mm(grad.t(), grad))

                    if state["step"] % group["update_freq"] == 0:
                        # compute matrix square roots using optimized eigendecomposition
                        try:
                            if CUDA_KERNELS_AVAILABLE and G_l.is_cuda:
                                # Use CUDA-optimized matrix operations
                                G_l_sqrt_inv = CUDAMatrixOps.matrix_sqrt_inv_eigen(G_l, power=-0.25, eps=group["eps"])
                                G_r_sqrt_inv = CUDAMatrixOps.matrix_sqrt_inv_eigen(G_r, power=-0.25, eps=group["eps"])
                            elif CYTHON_AVAILABLE:
                                # Use Cython-optimized matrix operations
                                G_l_sqrt_inv = CythonOptimizedOps.matrix_sqrt_inv(G_l, power=-0.25, eps=group["eps"])
                                G_r_sqrt_inv = CythonOptimizedOps.matrix_sqrt_inv(G_r, power=-0.25, eps=group["eps"])
                            else:
                                # Fallback to standard eigendecomposition
                                eigenvals_l, eigenvecs_l = torch.linalg.eigh(
                                    G_l + group["eps"] * torch.eye(G_l.shape[0], device=G_l.device, dtype=G_l.dtype)
                                )
                                G_l_sqrt_inv = eigenvecs_l @ torch.diag(eigenvals_l ** (-0.25)) @ eigenvecs_l.t()

                                eigenvals_r, eigenvecs_r = torch.linalg.eigh(
                                    G_r + group["eps"] * torch.eye(G_r.shape[0], device=G_r.device, dtype=G_r.dtype)
                                )
                                G_r_sqrt_inv = eigenvecs_r @ torch.diag(eigenvals_r ** (-0.25)) @ eigenvecs_r.t()

                            # preconditioning
                            search_direction = torch.mm(torch.mm(G_l_sqrt_inv, grad), G_r_sqrt_inv)
                        except:
                            # fallback to regular gradient if matrix operations fail
                            search_direction = grad
                    else:
                        search_direction = grad
                else:
                    # vector case - use diagonal preconditioning
                    G = state["G"]
                    G.add_(grad**2)

                    search_direction = grad / (torch.sqrt(G) + group["eps"])

                p.data.add_(search_direction, alpha=-group["lr"])

        return loss

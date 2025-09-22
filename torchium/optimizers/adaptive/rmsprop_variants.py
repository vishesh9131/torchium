"""
RMSprop and its variants implementation.
"""

import torch
import torch.optim as optim
import math
from typing import Optional, Callable, Union, List, Dict, Any
from ...utils.registry import register_optimizer


@register_optimizer("rmsprop")
class RMSprop(optim.RMSprop):
    """RMSprop optimizer with enhanced features."""

    def __init__(
        self,
        params: Union[List[torch.Tensor], Dict[str, Any]],
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum: float = 0,
        centered: bool = False,
        **kwargs,
    ):
        super().__init__(params, lr, alpha, eps, weight_decay, momentum, centered)


@register_optimizer("yogi")
class Yogi(optim.Optimizer):
    """
    Yogi: Adaptive Methods for Nonconvex Optimization.

    Reference: https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization
    """

    def __init__(
        self,
        params: Union[List[torch.Tensor], Dict[str, Any]],
        lr: float = 1e-2,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-3,
        initial_accumulator: float = 1e-6,
        weight_decay: float = 0,
        **kwargs,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, initial_accumulator=initial_accumulator, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Yogi does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.full_like(p.data, group["initial_accumulator"])

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Decay the first moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Yogi update rule for second moment
                grad_squared = grad.pow(2)
                exp_avg_sq.addcmul_(torch.sign(grad_squared - exp_avg_sq), grad_squared - exp_avg_sq, value=-(1 - beta2))

                # Compute bias-corrected second moment estimate
                denom = (exp_avg_sq / bias_correction2).sqrt().add_(group["eps"])

                # Apply weight decay
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-group["lr"] / bias_correction1)

        return loss

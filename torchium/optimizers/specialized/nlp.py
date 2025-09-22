"""
Optimizers specialized for NLP tasks.
"""

import torch
import torch.optim as optim
from typing import Optional, Callable, Union, List, Dict, Any
from ...utils.registry import register_optimizer

# Import AdaFactor from adaptive module
from ..adaptive.adagrad_variants import AdaFactor


@register_optimizer("lamb")
class LAMB(optim.Optimizer):
    """
    LAMB: Large Batch Optimization for Deep Learning.

    Reference: https://arxiv.org/abs/1904.00962
    """

    def __init__(
        self,
        params: Union[List[torch.Tensor], Dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.01,
        clamp_value: float = 10.0,
        **kwargs,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, clamp_value=clamp_value)
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
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Exponential moving average of squared gradient values
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Update biased first moment estimate
                scaled_exp_avg = exp_avg / bias_correction1

                # Update biased second raw moment estimate
                scaled_exp_avg_sq = exp_avg_sq / bias_correction2

                # Add weight decay
                if group["weight_decay"] != 0:
                    scaled_exp_avg.add_(p.data, alpha=group["weight_decay"])

                # Compute update
                update = scaled_exp_avg / (scaled_exp_avg_sq.sqrt() + group["eps"])

                # Layer-wise adaptation
                update_norm = update.norm()
                weight_norm = p.data.norm()

                if weight_norm > 0 and update_norm > 0:
                    # Compute the local learning rate
                    trust_ratio = min(weight_norm / update_norm, group["clamp_value"])
                    local_lr = group["lr"] * trust_ratio
                else:
                    local_lr = group["lr"]

                # Apply update
                p.data.add_(update, alpha=-local_lr)

        return loss


@register_optimizer("novograd")
class NovoGrad(optim.Optimizer):
    """
    NovoGrad: Stochastic Gradient Methods with Layer-wise Adaptive Moments.

    Reference: https://arxiv.org/abs/1905.11286
    """

    def __init__(
        self,
        params: Union[List[torch.Tensor], Dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple = (0.95, 0.98),
        eps: float = 1e-8,
        weight_decay: float = 0,
        grad_averaging: bool = True,
        amsgrad: bool = False,
        **kwargs,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, grad_averaging=grad_averaging, amsgrad=amsgrad)
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
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros([])
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = torch.zeros([])

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Add weight decay
                if group["weight_decay"] != 0:
                    grad = grad.add(p.data, alpha=group["weight_decay"])

                # Exponential moving average of gradient values
                if group["grad_averaging"]:
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                else:
                    exp_avg.copy_(grad)

                # Exponential moving average of squared gradient norm
                grad_norm_sq = grad.norm().pow(2)
                exp_avg_sq.mul_(beta2).add_(grad_norm_sq, alpha=1 - beta2)

                if group["amsgrad"]:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom_sq = max_exp_avg_sq
                else:
                    denom_sq = exp_avg_sq

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Update parameters
                denom = (denom_sq / bias_correction2).sqrt().add_(group["eps"])
                step_size = group["lr"] / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

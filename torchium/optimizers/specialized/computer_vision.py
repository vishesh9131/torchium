"""
Optimizers specialized for computer vision tasks.
"""

import torch
import torch.optim as optim
import math
from typing import Optional, Callable, Union, List, Dict, Any
from ...utils.registry import register_optimizer


@register_optimizer("ranger")
class Ranger(optim.Optimizer):
    """
    Ranger: A synergistic optimizer combining RAdam and LookAhead.

    Reference: https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
    """

    def __init__(
        self,
        params: Union[List[torch.Tensor], Dict[str, Any]],
        lr: float = 1e-3,
        alpha: float = 0.5,
        k: int = 6,
        n_sma_threshhold: int = 5,
        betas: tuple = (0.95, 0.999),
        eps: float = 1e-5,
        weight_decay: float = 0,
        **kwargs,
    ):
        # Initialize with RAdam defaults
        defaults = dict(
            lr=lr, alpha=alpha, k=k, n_sma_threshhold=n_sma_threshhold, betas=betas, eps=eps, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

        # Initialize LookAhead parameters
        self.alpha = alpha
        self.k = k
        for group in self.param_groups:
            group["step_counter"] = 0

        self.slow_weights = [[p.clone().detach() for p in group["params"]] for group in self.param_groups]

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
                    raise RuntimeError("Ranger does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Apply weight decay
                if group["weight_decay"] != 0:
                    grad = grad.add(p.data, alpha=group["weight_decay"])

                # Exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Exponential moving average of squared gradient values
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                buffered = exp_avg_sq.clone()
                buffered.sqrt_().add_(group["eps"])

                # RAdam variance rectification term
                sma_inf = 2.0 / (1.0 - beta2) - 1.0
                sma_t = sma_inf - 2.0 * state["step"] * (beta2 ** state["step"]) / (1.0 - beta2 ** state["step"])

                if sma_t >= group["n_sma_threshhold"]:
                    # Variance rectification term
                    r_t = math.sqrt((sma_t - 4.0) * (sma_t - 2.0) * sma_inf / ((sma_inf - 4.0) * (sma_inf - 2.0) * sma_t))

                    # Apply bias correction
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]

                    step_size = group["lr"] * r_t / bias_correction1
                    bias_correction2_sqrt = math.sqrt(bias_correction2)

                    p.data.addcdiv_(exp_avg, buffered / bias_correction2_sqrt, value=-step_size)
                else:
                    # Use unbiased estimate
                    bias_correction1 = 1 - beta1 ** state["step"]
                    step_size = group["lr"] / bias_correction1

                    p.data.addcdiv_(exp_avg, buffered, value=-step_size)

        # LookAhead mechanism
        for group in self.param_groups:
            group["step_counter"] += 1

            if group["step_counter"] % self.k == 0:
                for i, p in enumerate(group["params"]):
                    slow_p = self.slow_weights[self.param_groups.index(group)][i]
                    slow_p.add_(p.data - slow_p, alpha=self.alpha)
                    p.data.copy_(slow_p)

        return loss


@register_optimizer("ranger21")
class Ranger21(optim.Optimizer):
    """
    Ranger21: An advanced synergistic optimizer.

    Reference: https://github.com/lessw2020/Ranger21
    """

    def __init__(
        self,
        params: Union[List[torch.Tensor], Dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        use_madgrad: bool = True,
        use_gc: bool = True,
        warmup_pct: float = 0.22,
        **kwargs,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            use_madgrad=use_madgrad,
            use_gc=use_gc,
            warmup_pct=warmup_pct,
        )
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

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    if group["use_madgrad"]:
                        state["s"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Gradient centralization
                if group["use_gc"] and len(grad.shape) > 1:
                    grad = grad - grad.mean(dim=tuple(range(1, len(grad.shape))), keepdim=True)

                # Weight decay
                if group["weight_decay"] != 0:
                    grad = grad.add(p.data, alpha=group["weight_decay"])

                if group["use_madgrad"]:
                    # MADGRAD update
                    s = state["s"]

                    # Bias correction
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]

                    # Update biased first moment estimate
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                    # Update biased second raw moment estimate
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # Update the sum for MADGRAD
                    s.add_(grad, alpha=group["lr"])

                    # Compute the update
                    denom = (exp_avg_sq / bias_correction2).sqrt().add_(group["eps"])
                    update = (exp_avg / bias_correction1) / denom

                    p.data.add_(update, alpha=-group["lr"])
                else:
                    # Standard Adam update
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]

                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    denom = (exp_avg_sq / bias_correction2).sqrt().add_(group["eps"])

                    p.data.addcdiv_(exp_avg, denom, value=-group["lr"] / bias_correction1)

        return loss


@register_optimizer("ranger25")
class Ranger25(optim.Optimizer):
    """Ranger25: Latest version of Ranger with additional improvements."""

    def __init__(
        self,
        params: Union[List[torch.Tensor], Dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        use_softplus: bool = False,
        **kwargs,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, use_softplus=use_softplus)
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
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Softplus transformation
                if group["use_softplus"]:
                    grad = torch.nn.functional.softplus(grad, beta=1)

                if group["weight_decay"] != 0:
                    grad = grad.add(p.data, alpha=group["weight_decay"])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq / bias_correction2).sqrt().add_(group["eps"])

                p.data.addcdiv_(exp_avg, denom, value=-group["lr"] / bias_correction1)

        return loss


# AdamP is already implemented in adaptive/adam_variants.py, so we'll just import it
from ..adaptive.adam_variants import AdamP

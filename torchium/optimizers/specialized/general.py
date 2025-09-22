"""
General specialized optimizers.
"""

import torch
import torch.optim as optim
from typing import Optional, Callable, Union, List, Dict, Any
from ...utils.registry import register_optimizer


@register_optimizer("lion")
class Lion(optim.Optimizer):
    """
    Lion: Symbolic Discovery of Optimization Algorithms.

    Reference: https://arxiv.org/abs/2302.06675
    """

    def __init__(
        self,
        params: Union[List[torch.Tensor], Dict[str, Any]],
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.99),
        weight_decay: float = 0.0,
        **kwargs,
    ):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
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
                    state["exp_avg"] = torch.zeros_like(p.data)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Weight decay
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                # Update rule: sign(lerp(grad, momentum, beta1))
                lerp_result = grad.lerp(exp_avg, beta1)
                update = torch.sign(lerp_result)

                # Update parameters
                p.data.add_(update, alpha=-group["lr"])

                # Update momentum
                exp_avg.lerp_(grad, 1 - beta2)

        return loss


@register_optimizer("madgrad")
class MADGRAD(optim.Optimizer):
    """
    MADGRAD: A Momentumized, Adaptive, Dual Averaged Gradient Method.

    Reference: https://arxiv.org/abs/2101.11075
    """

    def __init__(
        self,
        params: Union[List[torch.Tensor], Dict[str, Any]],
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 0,
        eps: float = 1e-6,
        **kwargs,
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eps=eps)
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
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    state["s"] = torch.zeros_like(p.data)
                    if group["momentum"] > 0:
                        state["x0"] = p.data.clone()

                exp_avg_sq = state["exp_avg_sq"]
                s = state["s"]

                state["step"] += 1

                # Weight decay
                if group["weight_decay"] != 0:
                    grad = grad.add(p.data, alpha=group["weight_decay"])

                # Update squared gradient accumulator
                exp_avg_sq.addcmul_(grad, grad, value=1)

                # Compute RMS
                rms = exp_avg_sq.div(state["step"]).sqrt_().add_(group["eps"])

                # Update s
                s.addcdiv_(grad, rms, value=group["lr"])

                if group["momentum"] == 0:
                    # No momentum case
                    p.data.copy_(s.neg())
                else:
                    # With momentum
                    x0 = state["x0"]
                    z = x0 - s / (1 - group["momentum"])
                    p.data.mul_(group["momentum"]).add_(z, alpha=1 - group["momentum"])

        return loss


# Simplified implementations for remaining optimizers
@register_optimizer("apollo")
class Apollo(optim.Optimizer):
    """Apollo: An Adaptive Parameter-wise Diagonal Quasi-Newton Method."""

    def __init__(self, params, lr=1e-2, beta=0.9, eps=1e-4, init_lr=0.01, warmup=0, **kwargs):
        defaults = dict(lr=lr, beta=beta, eps=eps, init_lr=init_lr, warmup=warmup)
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
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg_sq = state["exp_avg_sq"]
                beta = group["beta"]

                state["step"] += 1

                exp_avg_sq.mul_(beta).addcmul_(grad, grad, value=1 - beta)

                bias_correction = 1 - beta ** state["step"]
                denominator = (exp_avg_sq / bias_correction).sqrt().add_(group["eps"])

                p.data.addcdiv_(grad, denominator, value=-group["lr"])

        return loss


@register_optimizer("a2grad")
class A2Grad(optim.Optimizer):
    """A2Grad: Optimal Adaptive and Accelerated Stochastic Gradient Descent."""

    def __init__(self, params, lr=1e-3, beta=0.7, lips=10.0, rho=0.5, **kwargs):
        defaults = dict(lr=lr, beta=beta, lips=lips, rho=rho)
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
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(p.data)

                buf = state["momentum_buffer"]
                state["step"] += 1

                buf.mul_(group["beta"]).add_(grad)

                p.data.add_(buf, alpha=-group["lr"])

        return loss


@register_optimizer("accsgd")
class AccSGD(optim.Optimizer):
    """AccSGD: Accelerating Stochastic Gradient Descent."""

    def __init__(self, params, lr=1e-3, kappa=1000.0, xi=10.0, **kwargs):
        defaults = dict(lr=lr, kappa=kappa, xi=xi)
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
                state = self.state[p]

                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p.data)

                buf = state["momentum_buffer"]

                # Simple momentum update
                buf.mul_(0.9).add_(grad)
                p.data.add_(buf, alpha=-group["lr"])

        return loss


@register_optimizer("asgd")
class ASGD(optim.ASGD):
    """ASGD optimizer with enhanced features."""

    pass


@register_optimizer("sgdw")
class SGDW(optim.Optimizer):
    """SGDW: SGD with decoupled Weight Decay."""

    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=1e-4, **kwargs):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
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
                state = self.state[p]

                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p.data)

                buf = state["momentum_buffer"]

                # Decoupled weight decay
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                buf.mul_(group["momentum"]).add_(grad)
                p.data.add_(buf, alpha=-group["lr"])

        return loss

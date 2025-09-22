"""
Adam and its variants implementation.
"""

import torch
import torch.optim as optim
import math
from typing import Optional, Callable, Union, List, Dict, Any
from ...utils.registry import register_optimizer


class Adam(optim.Adam):
    """Adam optimizer with enhanced features."""

    def __init__(
        self,
        params: Union[List[torch.Tensor], Dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        **kwargs,
    ):
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)


@register_optimizer("adamw")
class AdamW(optim.AdamW):
    """AdamW optimizer with decoupled weight decay."""

    def __init__(
        self,
        params: Union[List[torch.Tensor], Dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        **kwargs,
    ):
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)


@register_optimizer("radam")
class RAdam(optim.Optimizer):
    """
    RAdam: Rectified Adam optimizer.

    Reference: https://arxiv.org/abs/1908.03265
    """

    def __init__(
        self,
        params: Union[List[torch.Tensor], Dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        **kwargs,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
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
                    raise RuntimeError("RAdam does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected first moment estimate
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Compute length of approximated SMA
                sma_inf = 2.0 / (1 - beta2) - 1
                sma_t = sma_inf - 2 * state["step"] * (beta2 ** state["step"]) / bias_correction2

                # Compute the length of the approximated SMA
                if sma_t >= 5:
                    # Compute bias-corrected running average
                    exp_avg_sq_sqrt = exp_avg_sq.sqrt()
                    r_t = math.sqrt((sma_t - 4) * (sma_t - 2) * sma_inf / ((sma_inf - 4) * (sma_inf - 2) * sma_t))

                    # Update parameters
                    denom = exp_avg_sq_sqrt / math.sqrt(bias_correction2) + group["eps"]
                    p.data.addcdiv_(exp_avg, denom, value=-group["lr"] * r_t / bias_correction1)
                else:
                    # Use unrectified update
                    p.data.addcdiv_(
                        exp_avg,
                        exp_avg_sq.sqrt() / math.sqrt(bias_correction2) + group["eps"],
                        value=-group["lr"] / bias_correction1,
                    )

                # Apply weight decay
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

        return loss


@register_optimizer("adabelief")
class AdaBelief(optim.Optimizer):
    """
    AdaBelief: Adapting Step-sizes by the Belief in Observed Gradients.

    Reference: https://arxiv.org/abs/2010.07468
    """

    def __init__(
        self,
        params: Union[List[torch.Tensor], Dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        **kwargs,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
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
                    raise RuntimeError("AdaBelief does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Compute the belief (variance of gradients)
                grad_residual = grad - exp_avg
                exp_avg_sq.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

                if group["amsgrad"]:
                    # Maintains the maximum of all 2nd moment running avg. of squared gradients
                    torch.max(state["max_exp_avg_sq"], exp_avg_sq, out=state["max_exp_avg_sq"])
                    # Use the max. for normalizing running avg. of gradient
                    denom = (state["max_exp_avg_sq"] / bias_correction2).sqrt().add_(group["eps"])
                else:
                    denom = (exp_avg_sq / bias_correction2).sqrt().add_(group["eps"])

                # Apply weight decay
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-group["lr"] / bias_correction1)

        return loss


@register_optimizer("adabound")
class AdaBound(optim.Optimizer):
    """
    AdaBound: Adaptive Gradient Methods with Dynamic Bound of Learning Rate.

    Reference: https://openreview.net/forum?id=Bkg3g2R9FX
    """

    def __init__(
        self,
        params: Union[List[torch.Tensor], Dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        final_lr: float = 0.1,
        gamma: float = 1e-3,
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsbound: bool = False,
        **kwargs,
    ):
        defaults = dict(
            lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps, weight_decay=weight_decay, amsbound=amsbound
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
                if grad.is_sparse:
                    raise RuntimeError("AdaBound does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    if group["amsbound"]:
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if group["amsbound"]:
                    # Maintains the maximum of all 2nd moment running avg. of squared gradients
                    torch.max(state["max_exp_avg_sq"], exp_avg_sq, out=state["max_exp_avg_sq"])
                    # Use the max. for normalizing running avg. of gradient
                    denom = (state["max_exp_avg_sq"] / bias_correction2).sqrt().add_(group["eps"])
                else:
                    denom = (exp_avg_sq / bias_correction2).sqrt().add_(group["eps"])

                # Compute the bounds
                step_size = group["lr"] / bias_correction1
                final_lr = group["final_lr"] * group["lr"] / group["lr"]
                lower_bound = final_lr * (1.0 - 1.0 / (group["gamma"] * state["step"] + 1))
                upper_bound = final_lr * (1.0 + 1.0 / (group["gamma"] * state["step"]))

                # Clamp the step size
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(denom)

                # Apply weight decay
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


@register_optimizer("adahessian")
class AdaHessian(optim.Optimizer):
    """
    AdaHessian: An Adaptive Second Order Optimizer for Machine Learning.

    Reference: https://arxiv.org/abs/2006.00719
    """

    def __init__(
        self,
        params: Union[List[torch.Tensor], Dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        hessian_power: float = 1.0,
        **kwargs,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, hessian_power=hessian_power)
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
                    raise RuntimeError("AdaHessian does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    state["hessian_diag"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq, hessian_diag = state["exp_avg"], state["exp_avg_sq"], state["hessian_diag"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Estimate diagonal Hessian
                hessian_diag.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute the denominator
                denom = (hessian_diag / bias_correction2).pow_(group["hessian_power"]).add_(group["eps"])

                # Apply weight decay
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-group["lr"] / bias_correction1)

        return loss


@register_optimizer("adamp")
class AdamP(optim.Optimizer):
    """
    AdamP: Slowing Down the Slowdown for Momentum Optimizers on Scale-invariant Weights.

    Reference: https://arxiv.org/abs/2006.08217
    """

    def __init__(
        self,
        params: Union[List[torch.Tensor], Dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        delta: float = 0.1,
        wd_ratio: float = 0.1,
        nesterov: bool = False,
        **kwargs,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, delta=delta, wd_ratio=wd_ratio, nesterov=nesterov
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
                if grad.is_sparse:
                    raise RuntimeError("AdamP does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute the denominator
                denom = (exp_avg_sq / bias_correction2).sqrt().add_(group["eps"])

                # Compute the cosine similarity
                cosine_sim = torch.sum(exp_avg * p.data) / (torch.norm(exp_avg) * torch.norm(p.data) + 1e-8)

                # Compute the projection
                if cosine_sim < group["delta"]:
                    # Project the gradient
                    grad_proj = grad - (torch.sum(grad * p.data) / (torch.norm(p.data) ** 2 + 1e-8)) * p.data
                    exp_avg_proj = exp_avg - (torch.sum(exp_avg * p.data) / (torch.norm(p.data) ** 2 + 1e-8)) * p.data
                else:
                    grad_proj = grad
                    exp_avg_proj = exp_avg

                # Update parameters
                p.data.addcdiv_(exp_avg_proj, denom, value=-group["lr"] / bias_correction1)

                # Apply weight decay
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

        return loss


# Additional Adam variants (simplified implementations)
@register_optimizer("adams")
class AdamS(optim.Optimizer):
    """AdamS: Adam with stable weight decay."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, **kwargs):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
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
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq / bias_correction2).sqrt().add_(group["eps"])

                # Stable weight decay
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                p.data.addcdiv_(exp_avg, denom, value=-group["lr"] / bias_correction1)

        return loss


@register_optimizer("adamd")
class AdamD(optim.Optimizer):
    """AdamD: Improved bias-correction in Adam."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, **kwargs):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
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
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Improved bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Enhanced bias correction
                bias_correction1 = bias_correction1**0.5
                bias_correction2 = bias_correction2**0.5

                denom = (exp_avg_sq / bias_correction2).sqrt().add_(group["eps"])

                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                p.data.addcdiv_(exp_avg, denom, value=-group["lr"] / bias_correction1)

        return loss


@register_optimizer("adamax")
class Adamax(optim.Adamax):
    """Adamax optimizer."""

    pass


@register_optimizer("adashift")
class AdaShift(optim.Optimizer):
    """AdaShift: Decentralized Adaptive Stochastic Optimization."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, **kwargs):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
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
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # AdaShift update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Shift the second moment
                shifted_exp_avg_sq = exp_avg_sq.clone()
                shifted_exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (shifted_exp_avg_sq / bias_correction2).sqrt().add_(group["eps"])

                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                p.data.addcdiv_(exp_avg, denom, value=-group["lr"] / bias_correction1)

        return loss


@register_optimizer("adasmooth")
class AdaSmooth(optim.Optimizer):
    """AdaSmooth: Adaptive Smoothing for Non-Convex Optimization."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, **kwargs):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
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
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Smooth the gradients
                smooth_grad = grad * (1 - beta1) + exp_avg * beta1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(smooth_grad, smooth_grad, value=1 - beta2)

                denom = (exp_avg_sq / bias_correction2).sqrt().add_(group["eps"])

                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                p.data.addcdiv_(exp_avg, denom, value=-group["lr"] / bias_correction1)

        return loss

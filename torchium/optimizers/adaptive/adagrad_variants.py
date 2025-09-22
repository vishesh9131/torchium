"""
Adagrad and its variants implementation.
"""

import torch
import torch.optim as optim
import math
from typing import Optional, Callable, Union, List, Dict, Any
from ...utils.registry import register_optimizer


@register_optimizer("adagrad")
class Adagrad(optim.Adagrad):
    """Adagrad optimizer with enhanced features."""

    def __init__(
        self,
        params: Union[List[torch.Tensor], Dict[str, Any]],
        lr: float = 1e-2,
        lr_decay: float = 0,
        weight_decay: float = 0,
        initial_accumulator_value: float = 0,
        eps: float = 1e-10,
        **kwargs,
    ):
        super().__init__(params, lr, lr_decay, weight_decay, initial_accumulator_value, eps)


@register_optimizer("adadelta")
class Adadelta(optim.Adadelta):
    """Adadelta optimizer with enhanced features."""

    def __init__(
        self,
        params: Union[List[torch.Tensor], Dict[str, Any]],
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0,
        **kwargs,
    ):
        super().__init__(params, lr, rho, eps, weight_decay)


@register_optimizer("adafactor")
class AdaFactor(optim.Optimizer):
    """
    AdaFactor: Adaptive Learning Rates with Sublinear Memory Cost.

    Reference: https://arxiv.org/abs/1804.04235
    """

    def __init__(
        self,
        params: Union[List[torch.Tensor], Dict[str, Any]],
        lr: Optional[float] = None,
        eps2: float = 1e-30,
        cliping_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta1: Optional[float] = None,
        weight_decay: float = 0.0,
        scale_parameter: bool = True,
        relative_step: bool = True,
        **kwargs,
    ):
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual lr and relative_step options")
        if not relative_step and lr is None:
            raise ValueError("Must set lr if relative_step is False")

        defaults = dict(
            lr=lr,
            eps2=eps2,
            cliping_threshold=cliping_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
        )
        super().__init__(params, defaults)

    def _get_lr(self, param_group, param_state):
        """Compute learning rate for each parameter tensor."""
        min_step = 1e-6 * param_state["step"] if param_group["scale_parameter"] else 1e-2
        rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps2"], param_state["RMS"])
        return param_scale * rel_step_sz

    def _get_options(self, param_group, param_shape):
        """Determine factorization options."""
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment

    def _rms(self, tensor):
        """Root mean square."""
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col):
        """Approximation of exponential moving average of square of gradient."""
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1).clamp_(0, math.inf)
        c_factor = (exp_avg_sq_col.rsqrt()).unsqueeze(0).clamp_(0, math.inf)
        return torch.mul(r_factor, c_factor)

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
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]
                grad_shape = grad.shape

                factored, use_first_moment = self._get_options(group, grad_shape)

                # State Initialization
                if len(state) == 0:
                    state["step"] = 0

                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(grad).float()
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).float()
                        state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).float()
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad).float()

                    state["RMS"] = 0

                p_data_fp32 = p.data.float() if p.data.dtype in {torch.float16, torch.bfloat16} else p.data

                state["step"] += 1
                state["RMS"] = self._rms(p_data_fp32)

                lr = group["lr"]
                if group["lr"] is None:
                    lr = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                update = grad**2 + group["eps2"]

                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=1.0 - beta2t)
                    exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=1.0 - beta2t)
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2t).add_(update, alpha=1.0 - beta2t)
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_((self._rms(update) / group["cliping_threshold"]).clamp_(min=1.0))

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(update, alpha=1 - group["beta1"])
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p_data_fp32.mul_(1 - group["weight_decay"] * lr)

                p_data_fp32.add_(update, alpha=-lr)

                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)

        return loss


@register_optimizer("adagc")
class AdaGC(optim.Optimizer):
    """AdaGC: Adaptive Gradient Compression."""

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

                # Gradient compression
                grad_norm = torch.norm(grad)
                if grad_norm > 1.0:
                    grad = grad / grad_norm

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq / bias_correction2).sqrt().add_(group["eps"])

                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                p.data.addcdiv_(exp_avg, denom, value=-group["lr"] / bias_correction1)

        return loss


@register_optimizer("adago")
class AdaGO(optim.Optimizer):
    """AdaGO: Adaptive Gradient with Orthogonalization."""

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
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    state["prev_grad"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq, prev_grad = state["exp_avg"], state["exp_avg_sq"], state["prev_grad"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Orthogonalize gradient
                dot_product = torch.sum(grad * prev_grad)
                prev_norm_sq = torch.sum(prev_grad * prev_grad)
                if prev_norm_sq > 0:
                    grad = grad - (dot_product / prev_norm_sq) * prev_grad

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq / bias_correction2).sqrt().add_(group["eps"])

                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                p.data.addcdiv_(exp_avg, denom, value=-group["lr"] / bias_correction1)

                # Update previous gradient
                prev_grad.copy_(grad)

        return loss


@register_optimizer("adalomo")
class AdaLOMO(optim.Optimizer):
    """AdaLOMO: Low-Memory Optimization for Large Models."""

    def __init__(self, model: torch.nn.Module, lr: float = 1e-3, eps: float = 1e-8, clip_threshold: float = 1.0, **kwargs):
        self.model = model
        params = list(model.parameters())
        defaults = dict(lr=lr, eps=eps, clip_threshold=clip_threshold)
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

                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1

                # Update second moment
                exp_avg_sq.mul_(0.999).addcmul_(grad, grad, value=0.001)

                # Compute update
                bias_correction = 1 - 0.999 ** state["step"]
                denom = (exp_avg_sq / bias_correction).sqrt().add_(group["eps"])

                # Clip gradient
                grad_norm = torch.norm(grad)
                if grad_norm > group["clip_threshold"]:
                    grad = grad * (group["clip_threshold"] / grad_norm)

                p.data.addcdiv_(grad, denom, value=-group["lr"])

        return loss


@register_optimizer("adai")
class Adai(optim.Optimizer):
    """Adai: Disentangling the Effects of Adaptive Learning Rate and Momentum."""

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

                # Adaptive inertia
                inertia = beta1 * (1 - beta1 ** (state["step"] - 1)) / bias_correction1

                exp_avg.mul_(inertia).add_(grad, alpha=1 - inertia)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq / bias_correction2).sqrt().add_(group["eps"])

                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                p.data.addcdiv_(exp_avg, denom, value=-group["lr"])

        return loss


# Simplified implementations for remaining variants
@register_optimizer("adalite")
class Adalite(optim.Optimizer):
    """Adalite: Lightweight Adaptive Optimizer."""

    def __init__(self, params, lr=1e-3, eps=1e-8, weight_decay=0, **kwargs):
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
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
                    state["sum_sq"] = torch.zeros_like(p.data)

                sum_sq = state["sum_sq"]
                sum_sq.addcmul_(grad, grad)

                std = sum_sq.sqrt().add_(group["eps"])

                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                p.data.addcdiv_(grad, std, value=-group["lr"])

        return loss


@register_optimizer("adammini")
class AdamMini(optim.Optimizer):
    """AdamMini: Minimal memory footprint Adam."""

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
                    state["v"] = torch.zeros_like(p.data)

                v = state["v"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction = 1 - beta2 ** state["step"]

                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (v / bias_correction).sqrt().add_(group["eps"])

                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                p.data.addcdiv_(grad, denom, value=-group["lr"])

        return loss


@register_optimizer("adamod")
class AdaMod(optim.Optimizer):
    """AdaMod: An Adaptive and Momental Bound Method."""

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

                # Momental bound
                denom = (exp_avg_sq / bias_correction2).sqrt().add_(group["eps"])
                step_size = group["lr"] / bias_correction1

                # Adaptive bound
                bound = step_size * torch.ones_like(denom)
                bound = torch.min(bound, 1.0 / denom)

                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                p.data.add_(exp_avg, alpha=-bound)

        return loss


@register_optimizer("adanorm")
class AdaNorm(optim.Optimizer):
    """AdaNorm: Adaptive Gradient Norm Correction."""

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

                # Normalize gradient
                grad_norm = torch.norm(grad)
                if grad_norm > 0:
                    grad = grad / grad_norm

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq / bias_correction2).sqrt().add_(group["eps"])

                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                p.data.addcdiv_(exp_avg, denom, value=-group["lr"] / bias_correction1)

        return loss


@register_optimizer("adapnm")
class AdaPNM(optim.Optimizer):
    """AdaPNM: Adaptive Positive-Negative Momentum."""

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
                    state["exp_avg_pos"] = torch.zeros_like(p.data)
                    state["exp_avg_neg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg_pos, exp_avg_neg, exp_avg_sq = (state["exp_avg_pos"], state["exp_avg_neg"], state["exp_avg_sq"])
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Separate positive and negative gradients
                grad_pos = torch.clamp(grad, min=0)
                grad_neg = torch.clamp(grad, max=0)

                exp_avg_pos.mul_(beta1).add_(grad_pos, alpha=1 - beta1)
                exp_avg_neg.mul_(beta1).add_(grad_neg, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Combine positive and negative momentum
                exp_avg = exp_avg_pos + exp_avg_neg

                denom = (exp_avg_sq / bias_correction2).sqrt().add_(group["eps"])

                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                p.data.addcdiv_(exp_avg, denom, value=-group["lr"] / bias_correction1)

        return loss

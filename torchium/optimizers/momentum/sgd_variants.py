"""
SGD and its variants implementation.
"""

import torch
import torch.optim as optim
import math
from typing import Optional, Callable, Union, List, Dict, Any
from ...utils.registry import register_optimizer


@register_optimizer("sgd")
class SGD(optim.SGD):
    """SGD optimizer with enhanced features."""

    def __init__(
        self,
        params: Union[List[torch.Tensor], Dict[str, Any]],
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        **kwargs,
    ):
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)


@register_optimizer("nesterovsgd")
class NesterovSGD(optim.SGD):
    """Nesterov SGD optimizer."""

    def __init__(
        self,
        params: Union[List[torch.Tensor], Dict[str, Any]],
        lr: float = 1e-3,
        momentum: float = 0.9,
        dampening: float = 0,
        weight_decay: float = 0,
        **kwargs,
    ):
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov=True)


@register_optimizer("qhm")
class QHM(optim.Optimizer):
    """
    QHM: Quasi-Hyperbolic Momentum optimizer.

    Reference: https://arxiv.org/abs/1810.06801
    """

    def __init__(
        self,
        params: Union[List[torch.Tensor], Dict[str, Any]],
        lr: float = 1e-3,
        momentum: float = 0.999,
        nu: float = 0.7,
        weight_decay: float = 0,
        weight_decouple: bool = True,
        **kwargs,
    ):
        defaults = dict(lr=lr, momentum=momentum, nu=nu, weight_decay=weight_decay, weight_decouple=weight_decouple)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                d_p = p.grad.data

                if len(param_state) == 0:
                    param_state["momentum_buffer"] = torch.zeros_like(p.data)

                buf = param_state["momentum_buffer"]

                if weight_decay != 0:
                    if group["weight_decouple"]:
                        # Decoupled weight decay (L2 regularization)
                        p.data.mul_(1.0 - group["lr"] * weight_decay)
                    else:
                        # L2 weight decay
                        d_p = d_p.add(p.data, alpha=weight_decay)

                # Update biased first moment estimate
                buf.mul_(momentum).add_(d_p)

                # Update parameters
                p.data.add_(d_p, alpha=-group["lr"] * group["nu"])
                p.data.add_(buf, alpha=-group["lr"] * (1.0 - group["nu"]))

        return loss


@register_optimizer("aggmo")
class AggMo(optim.Optimizer):
    """
    AggMo: Aggregated Momentum optimizer.

    Reference: https://arxiv.org/abs/1804.00325
    """

    def __init__(
        self,
        params: Union[List[torch.Tensor], Dict[str, Any]],
        lr: float = 1e-3,
        betas: List[float] = [0.0, 0.9, 0.99],
        weight_decay: float = 0,
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
            weight_decay = group["weight_decay"]
            betas = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)

                if len(param_state) == 0:
                    param_state["momentum_buffers"] = [torch.zeros_like(p.data) for _ in betas]

                buffers = param_state["momentum_buffers"]

                # Update momentum buffers
                for i, beta in enumerate(betas):
                    buffers[i].mul_(beta).add_(d_p)

                # Aggregate momentum
                avg_momentum = sum(buffers) / len(buffers)

                # Update parameters
                p.data.add_(avg_momentum, alpha=-group["lr"])

        return loss


@register_optimizer("swats")
class SWATS(optim.Optimizer):
    """
    SWATS: Switching from Adam to SGD optimizer.

    Reference: https://arxiv.org/abs/1712.07628
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
                    raise RuntimeError("SWATS does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data)
                    state["switched"] = False

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if group["amsgrad"]:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if group["amsgrad"]:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq / bias_correction2).sqrt().add_(group["eps"])
                else:
                    denom = (exp_avg_sq / bias_correction2).sqrt().add_(group["eps"])

                # Check switching condition
                if state["step"] > 100 and not state["switched"]:
                    # Compute projection
                    proj = torch.sum(exp_avg * grad) / (torch.norm(exp_avg) * torch.norm(grad) + 1e-8)
                    if proj.abs() > 0.9:  # High correlation, switch to SGD
                        state["switched"] = True

                # Apply weight decay
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                # Update parameters
                if state["switched"]:
                    # Use SGD update
                    p.data.add_(grad, alpha=-group["lr"])
                else:
                    # Use Adam update
                    p.data.addcdiv_(exp_avg, denom, value=-group["lr"] / bias_correction1)

        return loss


@register_optimizer("sgdp")
class SGDP(optim.Optimizer):
    """
    SGDP: SGD with Projection optimizer.

    Reference: https://arxiv.org/abs/2006.08217
    """

    def __init__(
        self,
        params: Union[List[torch.Tensor], Dict[str, Any]],
        lr: float = 1e-3,
        momentum: float = 0,
        eps: float = 1e-8,
        delta: float = 0.1,
        wd_ratio: float = 0.1,
        nesterov: bool = False,
        weight_decay: float = 0,
        **kwargs,
    ):
        defaults = dict(
            lr=lr, momentum=momentum, eps=eps, delta=delta, wd_ratio=wd_ratio, nesterov=nesterov, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)

                if momentum != 0:
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1)

                    if group["nesterov"]:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                # Projection
                wd_ratio = 1.0
                if len(p.shape) > 1:
                    cosine_sim = torch.sum(d_p * p.data) / (torch.norm(d_p) * torch.norm(p.data) + group["eps"])

                    if cosine_sim.abs() < group["delta"]:
                        # Project gradient
                        proj_norm = torch.sum(d_p * p.data) / (torch.norm(p.data) ** 2 + group["eps"])
                        d_p = d_p - proj_norm * p.data
                        wd_ratio = group["wd_ratio"]

                # Update parameters
                p.data.add_(d_p, alpha=-group["lr"])

                # Apply projected weight decay
                if weight_decay != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"] * wd_ratio)

        return loss


@register_optimizer("sgdsai")
class SGDSaI(optim.Optimizer):
    """SGDSaI: SGD with Scale-adaptive and Inertial momentum."""

    def __init__(
        self,
        params: Union[List[torch.Tensor], Dict[str, Any]],
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0,
        **kwargs,
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)

                if len(param_state) == 0:
                    param_state["momentum_buffer"] = torch.zeros_like(p.data)
                    param_state["step"] = 0

                buf = param_state["momentum_buffer"]
                param_state["step"] += 1

                # Scale-adaptive inertial momentum
                scale = 1.0 / (1.0 + param_state["step"] * 0.001)
                adaptive_momentum = momentum * scale

                buf.mul_(adaptive_momentum).add_(d_p)

                # Update parameters
                p.data.add_(buf, alpha=-group["lr"])

        return loss


@register_optimizer("signsgd")
class SignSGD(optim.Optimizer):
    """
    SignSGD: Sign SGD optimizer.

    Reference: https://arxiv.org/abs/1802.04434
    """

    def __init__(
        self,
        params: Union[List[torch.Tensor], Dict[str, Any]],
        lr: float = 1e-3,
        momentum: float = 0,
        weight_decay: float = 0,
        **kwargs,
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)

                # Sign of gradient
                d_p = torch.sign(d_p)

                if momentum != 0:
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p)

                    d_p = buf

                # Update parameters
                p.data.add_(d_p, alpha=-group["lr"])

        return loss

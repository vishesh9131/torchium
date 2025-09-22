"""
Classic momentum methods implementation.
"""

import torch
import torch.optim as optim
from typing import Optional, Callable, Union, List, Dict, Any
from ...utils.registry import register_optimizer


@register_optimizer("heavyball")
class HeavyBall(optim.Optimizer):
    """
    Heavy Ball momentum optimizer.

    Reference: Polyak, B. T. (1964). Some methods of speeding up the convergence of iteration methods.
    """

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

                buf = param_state["momentum_buffer"]

                # Heavy ball update
                buf.mul_(momentum).add_(d_p, alpha=-group["lr"])

                # Update parameters
                p.data.add_(buf)

        return loss


@register_optimizer("nag")
class NAG(optim.Optimizer):
    """
    Nesterov Accelerated Gradient optimizer.

    Reference: Nesterov, Y. (1983). A method of solving a convex programming problem
               with convergence rate O(1/k^2).
    """

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

                buf = param_state["momentum_buffer"]

                # Nesterov momentum update
                buf.mul_(momentum).add_(d_p)

                # Update parameters with lookahead
                p.data.add_(d_p, alpha=-group["lr"])
                p.data.add_(buf, alpha=-group["lr"] * momentum)

        return loss

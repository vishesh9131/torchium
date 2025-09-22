"""
Optimizers specialized for distributed training.
"""

import torch
import torch.optim as optim
from typing import Optional, Callable, Union, List, Dict, Any
from ...utils.registry import register_optimizer


@register_optimizer("lars")
class LARS(optim.Optimizer):
    """
    LARS: Layer-wise Adaptive Rate Scaling.

    Reference: https://arxiv.org/abs/1708.03888
    """

    def __init__(
        self,
        params: Union[List[torch.Tensor], Dict[str, Any]],
        lr: float = 1.0,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        trust_coefficient: float = 1e-3,
        eps: float = 1e-8,
        **kwargs,
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, trust_coefficient=trust_coefficient, eps=eps)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            trust_coeff = group["trust_coefficient"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)

                weight_norm = torch.norm(p.data)
                grad_norm = torch.norm(d_p)

                # Compute local learning rate
                if weight_norm > 0 and grad_norm > 0:
                    local_lr = trust_coeff * weight_norm / (grad_norm + group["eps"])
                else:
                    local_lr = 1.0

                local_lr = min(local_lr, 1.0)  # Clip to maximum of 1.0

                if momentum != 0:
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p)

                    d_p = buf

                # Apply update with local learning rate
                p.data.add_(d_p, alpha=-group["lr"] * local_lr)

        return loss

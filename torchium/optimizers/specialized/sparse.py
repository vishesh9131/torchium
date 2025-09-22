"""
Optimizers specialized for sparse data.
"""

import torch
import torch.optim as optim
from typing import Optional, Callable, Union, List, Dict, Any
from ...utils.registry import register_optimizer


@register_optimizer("sparseadam")
class SparseAdam(optim.SparseAdam):
    """SparseAdam optimizer with enhanced features."""

    def __init__(
        self,
        params: Union[List[torch.Tensor], Dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(params, lr, betas, eps)


@register_optimizer("sm3")
class SM3(optim.Optimizer):
    """
    SM3: Memory-Efficient Adaptive Optimization.

    Reference: https://arxiv.org/abs/1901.11150
    """

    def __init__(
        self,
        params: Union[List[torch.Tensor], Dict[str, Any]],
        lr: float = 1e-3,
        momentum: float = 0.0,
        eps: float = 1e-8,
        **kwargs,
    ):
        defaults = dict(lr=lr, momentum=momentum, eps=eps)
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
                    state["momentum_buffer"] = torch.zeros_like(p.data)
                    # Initialize accumulator
                    state["accumulator"] = torch.zeros_like(p.data)

                momentum_buf = state["momentum_buffer"]
                accumulator = state["accumulator"]

                state["step"] += 1

                # Update accumulator
                accumulator.addcmul_(grad, grad)

                # Compute update
                update = grad / (accumulator.sqrt() + group["eps"])

                # Apply momentum
                momentum_buf.mul_(group["momentum"]).add_(update)

                # Update parameters
                p.data.add_(momentum_buf, alpha=-group["lr"])

        return loss


@register_optimizer("ftrl")
class FTRL(optim.Optimizer):
    """
    FTRL: Follow The Regularized Leader optimizer.

    Reference: https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    """

    def __init__(
        self,
        params: Union[List[torch.Tensor], Dict[str, Any]],
        lr: float = 1.0,
        lr_power: float = -0.5,
        l1_regularization_strength: float = 0.0,
        l2_regularization_strength: float = 0.0,
        initial_accumulator_value: float = 0.1,
        **kwargs,
    ):
        defaults = dict(
            lr=lr,
            lr_power=lr_power,
            l1_regularization_strength=l1_regularization_strength,
            l2_regularization_strength=l2_regularization_strength,
            initial_accumulator_value=initial_accumulator_value,
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

                if len(state) == 0:
                    state["accumulator"] = torch.full_like(p.data, group["initial_accumulator_value"])
                    state["z"] = torch.zeros_like(p.data)

                accumulator = state["accumulator"]
                z = state["z"]

                # Update accumulator
                accumulator.addcmul_(grad, grad)

                # Compute sigma
                sigma = (accumulator.pow(group["lr_power"]) - accumulator.pow(group["lr_power"])) / group["lr"]

                # Update z
                z.add_(grad - sigma * p.data)

                # Apply L1 and L2 regularization
                l1_reg = group["l1_regularization_strength"]
                l2_reg = group["l2_regularization_strength"]

                # Compute new weights
                new_weights = torch.where(
                    z.abs() <= l1_reg,
                    torch.zeros_like(z),
                    -(z - l1_reg * z.sign()) / (l2_reg + (accumulator.pow(-group["lr_power"]) / group["lr"])),
                )

                p.data.copy_(new_weights)

        return loss

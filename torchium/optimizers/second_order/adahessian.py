import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from typing import Any, Dict, List, Optional, Tuple
import math


class AdaHessian(Optimizer):
    """AdaHessian optimizer using second-order information"""

    def __init__(
        self,
        params,
        lr=0.15,
        betas=(0.9, 0.999),
        eps=1e-4,
        weight_decay=0,
        hessian_power=1,
        update_each=1,
        n_samples=1,
        avg_conv_kernel=False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            hessian_power=hessian_power,
            update_each=update_each,
            n_samples=n_samples,
            avg_conv_kernel=avg_conv_kernel,
        )
        super().__init__(params, defaults)

    def get_trace(self, gradsH):
        """Compute trace of Hessian"""
        trace = 0.0
        for grad, Hv in gradsH:
            trace += torch.sum(grad * Hv).cpu().item()
        return trace

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        params = []
        grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    params.append(p)
                    grads.append(p.grad)

        # compute Hessian diagonal approximation
        device = grads[0].device
        zv = [torch.randint_like(p, high=2, device=device) for p in params]
        # rademacher random variables
        for z in zv:
            z[z == 0] = -1

        h_zv = torch.autograd.grad(grads, params, grad_outputs=zv, only_inputs=True, retain_graph=False)

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdaHessian does not support sparse gradients")

                state = self.state[p]

                # state initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_hessian_diag_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_hessian_diag_sq = state["exp_avg"], state["exp_hessian_diag_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                # add weight decay
                if group["weight_decay"] != 0:
                    grad = grad.add(p.data, alpha=group["weight_decay"])

                # update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # get Hessian diagonal element
                hut_trace = h_zv[i] * zv[i]

                # update biased second raw moment estimate
                exp_hessian_diag_sq.mul_(beta2).addcmul_(hut_trace, hut_trace, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # scale learning rate
                scaled_lr = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                # apply Hessian power
                hessian_powered = exp_hessian_diag_sq ** group["hessian_power"]

                p.data.addcdiv_(exp_avg, hessian_powered.sqrt() + group["eps"], value=-scaled_lr)

        return loss

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from typing import Any, Dict, List, Optional, Tuple, Callable
import math
from collections import defaultdict
import random


class SAM(Optimizer):
    """Sharpness-Aware Minimization optimizer"""

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """First step: compute adversarial parameters"""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Second step: update parameters using base optimizer"""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """Combined step function"""
        assert closure is not None, "SAM requires a closure, but it was not provided"

        # first forward-backward pass
        self.first_step(zero_grad=True)

        # second forward-backward pass
        with torch.enable_grad():
            closure()
        self.second_step()

    def _grad_norm(self):
        """Compute the norm of gradients"""
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(dtype=torch.float32).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            dtype=torch.float32,
        )
        return norm


class GSAM(Optimizer):
    """Gradient-based Sharpness-Aware Minimization"""

    def __init__(self, params, base_optimizer, rho=0.05, alpha=0.4, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        assert alpha >= 0.0, f"Invalid alpha, should be non-negative: {alpha}"

        defaults = dict(rho=rho, alpha=alpha, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                self.state[p]["old_g"] = p.grad.clone()

                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # gradient decomposition
                old_g = self.state[p]["old_g"]
                new_g = p.grad

                inner_prod = torch.sum(old_g * new_g)
                old_g_norm_sq = torch.sum(old_g * old_g)

                # cosine similarity based decomposition
                if old_g_norm_sq > 0:
                    cos_sim = inner_prod / old_g_norm_sq
                    cos_sim = torch.clamp(cos_sim, 0, 1)  # ensure non-negative

                    # surrogate gap and gradient
                    surrogate_g = cos_sim * old_g + group["alpha"] * (new_g - cos_sim * old_g)
                    p.grad = surrogate_g

                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        assert closure is not None, "GSAM requires a closure"

        self.first_step(zero_grad=True)

        with torch.enable_grad():
            closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(dtype=torch.float32).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            dtype=torch.float32,
        )
        return norm


class ASAM(Optimizer):
    """Adaptive Sharpness-Aware Minimization"""

    def __init__(self, params, base_optimizer, rho=0.5, eta=0.01, adaptive=True, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, eta=eta, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                self.state[p]["old_p"] = p.data.clone()

                if group["adaptive"]:
                    # adaptive scaling based on parameter magnitude
                    abs_p = torch.abs(p.data)
                    k = torch.clamp(abs_p, min=group["eta"])
                    e_w = p.grad * scale.to(p) * k
                else:
                    e_w = p.grad * scale.to(p)

                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        assert closure is not None, "ASAM requires a closure"

        self.first_step(zero_grad=True)

        with torch.enable_grad():
            closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack(
                [
                    p.grad.norm(dtype=torch.float32).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            dtype=torch.float32,
        )
        return norm


class LookSAM(Optimizer):
    """Look-ahead SAM optimizer"""

    def __init__(self, params, base_optimizer, k=5, alpha=0.5, rho=0.05, **kwargs):
        assert 0 <= alpha <= 1, f"Invalid alpha, should be in [0, 1]: {alpha}"
        assert k >= 1, f"Invalid k, should be >= 1: {k}"

        defaults = dict(k=k, alpha=alpha, rho=rho, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

        self.step_count = 0

    @torch.no_grad()
    def lookahead_step(self):
        """Perform lookahead update"""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                if "slow_buffer" not in param_state:
                    param_state["slow_buffer"] = torch.zeros_like(p.data)
                    param_state["slow_buffer"].copy_(p.data)

                slow = param_state["slow_buffer"]
                slow.add_(p.data - slow, alpha=group["alpha"])
                p.data.copy_(slow)

    def step(self, closure=None):
        assert closure is not None, "LookSAM requires a closure"

        # SAM step
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                if "old_p" not in self.state[p]:
                    self.state[p]["old_p"] = torch.zeros_like(p.data)
                self.state[p]["old_p"].copy_(p.data)
                p.add_(p.grad * scale.to(p))

        self.zero_grad()

        with torch.enable_grad():
            closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.copy_(self.state[p]["old_p"])

        self.base_optimizer.step()

        self.step_count += 1

        # Lookahead step
        if self.step_count % self.param_groups[0]["k"] == 0:
            self.lookahead_step()

        self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack(
                [
                    p.grad.norm(dtype=torch.float32).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            dtype=torch.float32,
        )
        return norm


class WSAM(Optimizer):
    """Weighted Sharpness-Aware Minimization"""

    def __init__(self, params, base_optimizer, rho=0.05, tau=1.0, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, tau=tau, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                self.state[p]["old_p"] = p.data.clone()

                # compute importance weights
                grad_norm_p = p.grad.norm()
                weight = torch.exp(group["tau"] * grad_norm_p)

                e_w = weight * p.grad * scale.to(p)
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        assert closure is not None, "WSAM requires a closure"

        self.first_step(zero_grad=True)

        with torch.enable_grad():
            closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack(
                [
                    p.grad.norm(dtype=torch.float32).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            dtype=torch.float32,
        )
        return norm


class GradientCentralization(Optimizer):
    """Gradient Centralization wrapper"""

    def __init__(self, params, base_optimizer, use_gc=True, gc_conv_only=False, **kwargs):
        defaults = dict(use_gc=use_gc, gc_conv_only=gc_conv_only, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    def centralize_gradient(self, p):
        """Apply gradient centralization"""
        if len(p.shape) > 1:  # only for conv and linear layers
            p.grad.data = p.grad.data - p.grad.data.mean(dim=tuple(range(1, len(p.shape))), keepdim=True)

    def step(self, closure=None):
        # Apply gradient centralization
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                if group["use_gc"]:
                    if group["gc_conv_only"]:
                        # only apply to conv layers (4D tensors)
                        if len(p.shape) == 4:
                            self.centralize_gradient(p)
                    else:
                        # apply to all multi-dimensional parameters
                        if len(p.shape) > 1:
                            self.centralize_gradient(p)

        # Call base optimizer
        return self.base_optimizer.step(closure)


class PCGrad(Optimizer):
    """Projecting Conflicting Gradients"""

    def __init__(self, params, base_optimizer, num_tasks, **kwargs):
        defaults = dict(num_tasks=num_tasks, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.num_tasks = num_tasks

    def project_conflicting_gradients(self, grads):
        """Project conflicting gradients"""
        pc_grads = []

        for i in range(len(grads)):
            grad_i = grads[i]
            pc_grad = grad_i.clone()

            for j in range(len(grads)):
                if i != j:
                    grad_j = grads[j]

                    # compute inner product
                    inner_prod = torch.sum(grad_i * grad_j)

                    # if negative inner product (conflicting), project
                    if inner_prod < 0:
                        grad_j_norm_sq = torch.sum(grad_j * grad_j)
                        if grad_j_norm_sq > 0:
                            pc_grad = pc_grad - (inner_prod / grad_j_norm_sq) * grad_j

            pc_grads.append(pc_grad)

        return pc_grads

    def step(self, closure=None, task_losses=None):
        """
        Args:
            closure: closure function
            task_losses: list of task-specific loss functions
        """
        if task_losses is not None:
            # compute gradients for each task
            task_grads = []
            for task_loss in task_losses:
                self.zero_grad()
                task_loss().backward(retain_graph=True)

                # collect gradients
                grads = []
                for group in self.param_groups:
                    for p in group["params"]:
                        if p.grad is not None:
                            grads.append(p.grad.clone().flatten())

                if grads:
                    task_grads.append(torch.cat(grads))

            # project conflicting gradients
            if task_grads:
                pc_grads = self.project_conflicting_gradients(task_grads)
                avg_grad = sum(pc_grads) / len(pc_grads)

                # assign back to parameters
                idx = 0
                for group in self.param_groups:
                    for p in group["params"]:
                        if p.grad is not None:
                            param_size = p.numel()
                            p.grad.data = avg_grad[idx : idx + param_size].reshape(p.shape)
                            idx += param_size

        return self.base_optimizer.step(closure)


class GradNorm(Optimizer):
    """Gradient Normalization for multi-task learning"""

    def __init__(self, params, base_optimizer, num_tasks, alpha=1.5, **kwargs):
        defaults = dict(num_tasks=num_tasks, alpha=alpha, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.num_tasks = num_tasks
        self.task_weights = nn.Parameter(torch.ones(num_tasks))

    def step(self, closure=None, task_losses=None, task_grads=None):
        """
        Args:
            closure: closure function
            task_losses: tensor of task losses
            task_grads: list of task gradient norms
        """
        if task_losses is not None and task_grads is not None:
            # compute relative task losses
            loss_ratios = task_losses / task_losses[0]  # relative to first task

            # compute gradient norm ratios
            grad_norm_avg = sum(task_grads) / len(task_grads)
            grad_ratios = [g / grad_norm_avg for g in task_grads]

            # compute targets
            targets = []
            for i in range(self.num_tasks):
                if i == 0:
                    targets.append(torch.tensor(1.0))
                else:
                    target = loss_ratios[i] ** self.param_groups[0]["alpha"]
                    targets.append(target)

            # update task weights
            for i in range(self.num_tasks):
                target = targets[i]
                grad_ratio = grad_ratios[i]

                # gradnorm loss
                gradnorm_loss = torch.abs(grad_ratio - target.detach())

                # update weight
                weight_grad = torch.autograd.grad(gradnorm_loss, self.task_weights[i], retain_graph=True)[0]
                self.task_weights.data[i] -= 0.025 * weight_grad  # small lr for weights

            # renormalize weights
            self.task_weights.data = self.num_tasks * self.task_weights.data / self.task_weights.data.sum()

        return self.base_optimizer.step(closure)


__all__ = ["SAM", "GSAM", "ASAM", "LookSAM", "WSAM", "GradientCentralization", "PCGrad", "GradNorm"]

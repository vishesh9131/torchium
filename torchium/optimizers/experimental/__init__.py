import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from typing import Any, Dict, List, Optional, Tuple, Callable
import math
import random
import numpy as np
from collections import defaultdict


class CMAES(Optimizer):
    """Covariance Matrix Adaptation Evolution Strategy"""

    def __init__(self, params, sigma=0.1, popsize=None, seed=None):
        if popsize is None:
            popsize = 4 + int(3 * math.log(sum(p.numel() for group in params for p in group["params"])))

        defaults = dict(sigma=sigma, popsize=popsize, seed=seed)
        super().__init__(params, defaults)

        # flatten all parameters
        self.param_shapes = []
        self.param_sizes = []
        for group in self.param_groups:
            for p in group["params"]:
                self.param_shapes.append(p.shape)
                self.param_sizes.append(p.numel())

        self.dim = sum(self.param_sizes)
        self.generation = 0

        # CMA-ES parameters
        self.popsize = popsize
        self.mu = popsize // 2
        self.weights = torch.log(torch.tensor(self.mu + 0.5)) - torch.log(torch.arange(1, self.mu + 1, dtype=torch.float))
        self.weights /= self.weights.sum()
        self.mueff = 1.0 / (self.weights**2).sum()

        # adaptation parameters
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.dim + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, math.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs

        # initialize state
        self.mean = torch.zeros(self.dim)
        self.sigma = sigma
        self.C = torch.eye(self.dim)
        self.pc = torch.zeros(self.dim)
        self.ps = torch.zeros(self.dim)
        self.B = torch.eye(self.dim)
        self.D = torch.ones(self.dim)
        self.BD = self.B * self.D
        self.chiN = math.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))

        if seed is not None:
            torch.manual_seed(seed)

    def _flatten_params(self):
        """Flatten all parameters into a single vector"""
        flat_params = []
        for group in self.param_groups:
            for p in group["params"]:
                flat_params.append(p.data.flatten())
        return torch.cat(flat_params)

    def _unflatten_params(self, flat_params):
        """Unflatten vector back to parameter tensors"""
        offset = 0
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                size = self.param_sizes[offset // p.numel() if offset < len(self.param_sizes) else -1]
                p.data = flat_params[offset : offset + size].reshape(p.shape)
                offset += size

    def step(self, closure):
        """Perform one generation of CMA-ES"""
        if closure is None:
            raise ValueError("CMA-ES requires a closure function")

        # sample population
        population = []
        fitness = []

        for _ in range(self.popsize):
            # sample from multivariate normal
            z = torch.randn(self.dim)
            y = self.BD @ z
            x = self.mean + self.sigma * y

            # set parameters and evaluate
            self._unflatten_params(x)
            with torch.enable_grad():
                f = float(closure())
            fitness.append(f)
            population.append(x)

        # sort by fitness
        fitness = torch.tensor(fitness)
        sorted_indices = torch.argsort(fitness)

        # update mean
        old_mean = self.mean.clone()
        self.mean = sum(self.weights[i] * population[sorted_indices[i]] for i in range(self.mu))

        # update evolution paths
        y_mean = (self.mean - old_mean) / self.sigma
        z_mean = self.B.T @ y_mean / self.D

        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * (self.B @ z_mean)

        hsig = (self.ps.norm() / math.sqrt(1 - (1 - self.cs) ** (2 * self.generation + 1)) / self.chiN) < (
            1.4 + 2 / (self.dim + 1)
        )

        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_mean

        # update covariance matrix
        artmp = torch.stack([(population[sorted_indices[i]] - old_mean) / self.sigma for i in range(self.mu)])
        self.C = (
            (1 - self.c1 - self.cmu) * self.C
            + self.c1 * (self.pc.outer(self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
            + self.cmu * sum(self.weights[i] * artmp[i].outer(artmp[i]) for i in range(self.mu))
        )

        # update step size
        self.sigma *= math.exp((self.cs / self.damps) * (self.ps.norm() / self.chiN - 1))

        # eigendecomposition
        if self.generation % (self.dim / (self.c1 + self.cmu) / self.dim / 10) < 1:
            eigvals, eigvecs = torch.linalg.eigh(self.C)
            self.D = torch.sqrt(torch.clamp(eigvals, min=1e-14))
            self.B = eigvecs
            self.BD = self.B * self.D

        self.generation += 1

        # set best parameters
        best_idx = sorted_indices[0]
        self._unflatten_params(population[best_idx])

        return fitness[best_idx]


class DifferentialEvolution(Optimizer):
    """Differential Evolution optimizer"""

    def __init__(self, params, popsize=50, mutation=0.8, recombination=0.7, seed=None, bounds=None):
        defaults = dict(popsize=popsize, mutation=mutation, recombination=recombination, seed=seed, bounds=bounds)
        super().__init__(params, defaults)

        self.param_shapes = []
        self.param_sizes = []
        for group in self.param_groups:
            for p in group["params"]:
                self.param_shapes.append(p.shape)
                self.param_sizes.append(p.numel())

        self.dim = sum(self.param_sizes)
        self.generation = 0

        if seed is not None:
            torch.manual_seed(seed)

        # initialize population
        self.population = []
        for _ in range(popsize):
            individual = torch.randn(self.dim)
            if bounds is not None:
                individual = torch.clamp(individual, bounds[0], bounds[1])
            self.population.append(individual)

    def _flatten_params(self):
        flat_params = []
        for group in self.param_groups:
            for p in group["params"]:
                flat_params.append(p.data.flatten())
        return torch.cat(flat_params)

    def _unflatten_params(self, flat_params):
        offset = 0
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                size = self.param_sizes[idx]
                p.data = flat_params[offset : offset + size].reshape(p.shape)
                offset += size
                idx += 1

    def step(self, closure):
        if closure is None:
            raise ValueError("DE requires a closure function")

        group = self.param_groups[0]
        popsize = group["popsize"]
        mutation = group["mutation"]
        recombination = group["recombination"]
        bounds = group["bounds"]

        fitness = []

        # evaluate current population
        for individual in self.population:
            self._unflatten_params(individual)
            with torch.enable_grad():
                f = float(closure())
            fitness.append(f)

        new_population = []

        for i in range(popsize):
            # select three random individuals (different from current)
            candidates = list(range(popsize))
            candidates.remove(i)
            a, b, c = random.sample(candidates, 3)

            # mutation
            mutant = self.population[a] + mutation * (self.population[b] - self.population[c])

            if bounds is not None:
                mutant = torch.clamp(mutant, bounds[0], bounds[1])

            # crossover
            trial = self.population[i].clone()
            crossover_mask = torch.rand(self.dim) < recombination
            # ensure at least one dimension is from mutant
            if not crossover_mask.any():
                crossover_mask[random.randint(0, self.dim - 1)] = True
            trial[crossover_mask] = mutant[crossover_mask]

            # selection
            self._unflatten_params(trial)
            with torch.enable_grad():
                trial_fitness = float(closure())

            if trial_fitness < fitness[i]:
                new_population.append(trial)
                fitness[i] = trial_fitness
            else:
                new_population.append(self.population[i])

        self.population = new_population
        self.generation += 1

        # set best individual
        best_idx = fitness.index(min(fitness))
        self._unflatten_params(self.population[best_idx])

        return min(fitness)


class ParticleSwarmOptimization(Optimizer):
    """Particle Swarm Optimization"""

    def __init__(self, params, popsize=50, inertia=0.9, cognitive=2.0, social=2.0, seed=None, bounds=None):
        defaults = dict(popsize=popsize, inertia=inertia, cognitive=cognitive, social=social, seed=seed, bounds=bounds)
        super().__init__(params, defaults)

        self.param_shapes = []
        self.param_sizes = []
        for group in self.param_groups:
            for p in group["params"]:
                self.param_shapes.append(p.shape)
                self.param_sizes.append(p.numel())

        self.dim = sum(self.param_sizes)

        if seed is not None:
            torch.manual_seed(seed)

        # initialize swarm
        self.positions = []
        self.velocities = []
        self.personal_best_positions = []
        self.personal_best_fitness = []
        self.global_best_position = None
        self.global_best_fitness = float("inf")

        for _ in range(popsize):
            pos = torch.randn(self.dim)
            vel = torch.zeros(self.dim)

            if bounds is not None:
                pos = torch.clamp(pos, bounds[0], bounds[1])

            self.positions.append(pos)
            self.velocities.append(vel)
            self.personal_best_positions.append(pos.clone())
            self.personal_best_fitness.append(float("inf"))

    def _unflatten_params(self, flat_params):
        offset = 0
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                size = self.param_sizes[idx]
                p.data = flat_params[offset : offset + size].reshape(p.shape)
                offset += size
                idx += 1

    def step(self, closure):
        if closure is None:
            raise ValueError("PSO requires a closure function")

        group = self.param_groups[0]
        popsize = group["popsize"]
        inertia = group["inertia"]
        cognitive = group["cognitive"]
        social = group["social"]
        bounds = group["bounds"]

        # evaluate particles
        for i in range(popsize):
            self._unflatten_params(self.positions[i])
            with torch.enable_grad():
                fitness = float(closure())

            # update personal best
            if fitness < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = fitness
                self.personal_best_positions[i] = self.positions[i].clone()

            # update global best
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = self.positions[i].clone()

        # update velocities and positions
        for i in range(popsize):
            r1, r2 = torch.rand(2)

            cognitive_component = cognitive * r1 * (self.personal_best_positions[i] - self.positions[i])
            social_component = social * r2 * (self.global_best_position - self.positions[i])

            self.velocities[i] = inertia * self.velocities[i] + cognitive_component + social_component

            self.positions[i] += self.velocities[i]

            if bounds is not None:
                self.positions[i] = torch.clamp(self.positions[i], bounds[0], bounds[1])

        # set best position
        if self.global_best_position is not None:
            self._unflatten_params(self.global_best_position)

        return self.global_best_fitness


class QuantumAnnealing(Optimizer):
    """Simplified Quantum Annealing optimizer"""

    def __init__(self, params, temperature=1.0, cooling_rate=0.99, min_temperature=0.01, seed=None):
        defaults = dict(temperature=temperature, cooling_rate=cooling_rate, min_temperature=min_temperature, seed=seed)
        super().__init__(params, defaults)

        self.param_shapes = []
        self.param_sizes = []
        for group in self.param_groups:
            for p in group["params"]:
                self.param_shapes.append(p.shape)
                self.param_sizes.append(p.numel())

        self.dim = sum(self.param_sizes)
        self.current_temperature = temperature

        if seed is not None:
            torch.manual_seed(seed)

    def _flatten_params(self):
        flat_params = []
        for group in self.param_groups:
            for p in group["params"]:
                flat_params.append(p.data.flatten())
        return torch.cat(flat_params)

    def _unflatten_params(self, flat_params):
        offset = 0
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                size = self.param_sizes[idx]
                p.data = flat_params[offset : offset + size].reshape(p.shape)
                offset += size
                idx += 1

    def step(self, closure):
        if closure is None:
            raise ValueError("QA requires a closure function")

        group = self.param_groups[0]
        cooling_rate = group["cooling_rate"]
        min_temperature = group["min_temperature"]

        # get current state and energy
        current_state = self._flatten_params()
        current_energy = float(closure())

        # generate neighbor state (quantum tunneling effect)
        perturbation_strength = self.current_temperature * 0.1
        neighbor_state = current_state + perturbation_strength * torch.randn_like(current_state)

        # evaluate neighbor
        self._unflatten_params(neighbor_state)
        with torch.enable_grad():
            neighbor_energy = float(closure())

        # acceptance probability (quantum annealing)
        delta_energy = neighbor_energy - current_energy

        if delta_energy < 0:
            # accept better solution
            accepted = True
        else:
            # accept worse solution with quantum probability
            if self.current_temperature > min_temperature:
                prob = math.exp(-delta_energy / self.current_temperature)
                accepted = random.random() < prob
            else:
                accepted = False

        if not accepted:
            # revert to current state
            self._unflatten_params(current_state)
            return current_energy

        # cool down temperature
        self.current_temperature = max(self.current_temperature * cooling_rate, min_temperature)

        return neighbor_energy


class GeneticAlgorithm(Optimizer):
    """Genetic Algorithm optimizer"""

    def __init__(self, params, popsize=50, mutation_rate=0.1, crossover_rate=0.8, elite_ratio=0.1, seed=None, bounds=None):
        defaults = dict(
            popsize=popsize,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            elite_ratio=elite_ratio,
            seed=seed,
            bounds=bounds,
        )
        super().__init__(params, defaults)

        self.param_shapes = []
        self.param_sizes = []
        for group in self.param_groups:
            for p in group["params"]:
                self.param_shapes.append(p.shape)
                self.param_sizes.append(p.numel())

        self.dim = sum(self.param_sizes)
        self.generation = 0

        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)

        # initialize population
        self.population = []
        for _ in range(popsize):
            individual = torch.randn(self.dim)
            if bounds is not None:
                individual = torch.clamp(individual, bounds[0], bounds[1])
            self.population.append(individual)

    def _unflatten_params(self, flat_params):
        offset = 0
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                size = self.param_sizes[idx]
                p.data = flat_params[offset : offset + size].reshape(p.shape)
                offset += size
                idx += 1

    def _tournament_selection(self, fitness, tournament_size=3):
        """Tournament selection"""
        selected_indices = []
        popsize = len(self.population)

        for _ in range(popsize):
            tournament_indices = random.sample(range(popsize), min(tournament_size, popsize))
            tournament_fitness = [fitness[i] for i in tournament_indices]
            winner_idx = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
            selected_indices.append(winner_idx)

        return selected_indices

    def _crossover(self, parent1, parent2):
        """Single-point crossover"""
        if random.random() > self.param_groups[0]["crossover_rate"]:
            return parent1.clone(), parent2.clone()

        crossover_point = random.randint(1, self.dim - 1)

        child1 = torch.cat([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = torch.cat([parent2[:crossover_point], parent1[crossover_point:]])

        return child1, child2

    def _mutate(self, individual):
        """Gaussian mutation"""
        mutation_rate = self.param_groups[0]["mutation_rate"]
        mutation_mask = torch.rand(self.dim) < mutation_rate

        if mutation_mask.any():
            individual[mutation_mask] += torch.randn(mutation_mask.sum()) * 0.1

        bounds = self.param_groups[0]["bounds"]
        if bounds is not None:
            individual = torch.clamp(individual, bounds[0], bounds[1])

        return individual

    def step(self, closure):
        if closure is None:
            raise ValueError("GA requires a closure function")

        group = self.param_groups[0]
        popsize = group["popsize"]
        elite_ratio = group["elite_ratio"]

        # evaluate population
        fitness = []
        for individual in self.population:
            self._unflatten_params(individual)
            with torch.enable_grad():
                f = float(closure())
            fitness.append(f)

        # sort by fitness
        sorted_indices = sorted(range(popsize), key=lambda i: fitness[i])

        # elitism - keep best individuals
        num_elites = int(popsize * elite_ratio)
        new_population = [self.population[i].clone() for i in sorted_indices[:num_elites]]

        # selection for breeding
        selected_indices = self._tournament_selection(fitness)

        # generate offspring
        while len(new_population) < popsize:
            parent1_idx = random.choice(selected_indices)
            parent2_idx = random.choice(selected_indices)

            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]

            child1, child2 = self._crossover(parent1, parent2)

            child1 = self._mutate(child1)
            child2 = self._mutate(child2)

            new_population.extend([child1, child2])

        # trim to population size
        self.population = new_population[:popsize]
        self.generation += 1

        # set best individual
        best_fitness = min(fitness)
        best_idx = sorted_indices[0]
        self._unflatten_params(self.population[0])  # elite is now at index 0

        return best_fitness


__all__ = ["CMAES", "DifferentialEvolution", "ParticleSwarmOptimization", "QuantumAnnealing", "GeneticAlgorithm"]

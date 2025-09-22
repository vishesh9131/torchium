"""
Optimizers module for Torchium.
"""

from .adaptive import *
from .momentum import *
from .second_order import *
from .specialized import *
from .meta_optimizers import *
from .experimental import *

__all__ = [
    # Adaptive optimizers
    "Adam",
    "AdamW",
    "RAdam",
    "AdaBelief",
    "AdaBound",
    "AdaHessian",
    "AdamP",
    "AdamS",
    "AdamD",
    "Adamax",
    "AdaShift",
    "AdaSmooth",
    "Adagrad",
    "Adadelta",
    "AdaFactor",
    "AdaGC",
    "AdaGO",
    "AdaLOMO",
    "Adai",
    "Adalite",
    "AdamMini",
    "AdaMod",
    "AdaNorm",
    "AdaPNM",
    "RMSprop",
    "Yogi",
    # PyTorch optimizers we're including for completeness
    "NAdam",
    "Rprop",
    # Momentum optimizers
    "SGD",
    "NesterovSGD",
    "QHM",
    "AggMo",
    "SWATS",
    "SGDP",
    "SGDSaI",
    "SignSGD",
    "HeavyBall",
    "NAG",
    # Second-order optimizers
    "LBFGS",
    "Shampoo",
    "AdaHessian",
    "KFAC",
    "NaturalGradient",
    # Specialized optimizers
    "Ranger",
    "Ranger21",
    "Ranger25",
    "Lookahead",
    "LAMB",
    "NovoGrad",
    "LARS",
    "Lion",
    "MADGRAD",
    "SM3",
    "Apollo",
    "A2Grad",
    "AccSGD",
    "ASGD",
    "SGDW",
    "SparseAdam",
    "FTRL",
    # Meta-optimizers
    "SAM",
    "GSAM",
    "ASAM",
    "LookSAM",
    "WSAM",
    "GradientCentralization",
    "PCGrad",
    "GradNorm",
    # Experimental optimizers
    "CMAES",
    "DifferentialEvolution",
    "ParticleSwarmOptimization",
    "QuantumAnnealing",
    "GeneticAlgorithm",
]

# Import PyTorch optimizers for completeness
import torch.optim as optim

# Add PyTorch optimizers that we're not overriding
NAdam = optim.NAdam
Rprop = optim.Rprop

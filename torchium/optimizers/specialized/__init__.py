"""
Specialized optimizers module.
"""

from .computer_vision import Ranger, Ranger21, Ranger25, AdamP
from .nlp import LAMB, NovoGrad, AdaFactor
from .sparse import SparseAdam, SM3, FTRL
from .distributed import LARS
from .general import Lion, MADGRAD, Apollo, A2Grad, AccSGD, ASGD, SGDW

__all__ = [
    # Computer Vision
    "Ranger",
    "Ranger21",
    "Ranger25",
    "AdamP",
    # NLP
    "LAMB",
    "NovoGrad",
    "AdaFactor",
    # Sparse
    "SparseAdam",
    "SM3",
    "FTRL",
    # Distributed
    "LARS",
    # General
    "Lion",
    "MADGRAD",
    "Apollo",
    "A2Grad",
    "AccSGD",
    "ASGD",
    "SGDW",
]

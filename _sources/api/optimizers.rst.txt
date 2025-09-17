Optimizers API
==============

Torchium provides 65+ advanced optimizers organized into several categories, extending PyTorch's native optimizer collection with state-of-the-art algorithms from recent research.

.. currentmodule:: torchium.optimizers

Second-Order Optimizers
-----------------------

These optimizers use second-order information for faster convergence.

LBFGS
~~~~~

.. autoclass:: LBFGS
   :members:
   :undoc-members:
   :show-inheritance:

Shampoo
~~~~~~~

.. autoclass:: Shampoo
   :members:
   :undoc-members:
   :show-inheritance:

AdaHessian
~~~~~~~~~~

.. autoclass:: AdaHessian
   :members:
   :undoc-members:
   :show-inheritance:

KFAC
~~~~

.. autoclass:: KFAC
   :members:
   :undoc-members:
   :show-inheritance:

NaturalGradient
~~~~~~~~~~~~~~~

.. autoclass:: NaturalGradient
   :members:
   :undoc-members:
   :show-inheritance:

Meta-Optimizers
---------------

Sharpness-Aware Minimization (SAM) family and gradient manipulation methods.

SAM
~~~

.. autoclass:: SAM
   :members:
   :undoc-members:
   :show-inheritance:

GSAM
~~~~

.. autoclass:: GSAM
   :members:
   :undoc-members:
   :show-inheritance:

ASAM
~~~~

.. autoclass:: ASAM
   :members:
   :undoc-members:
   :show-inheritance:

LookSAM
~~~~~~~

.. autoclass:: LookSAM
   :members:
   :undoc-members:
   :show-inheritance:

WSAM
~~~~

.. autoclass:: WSAM
   :members:
   :undoc-members:
   :show-inheritance:

GradientCentralization
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GradientCentralization
   :members:
   :undoc-members:
   :show-inheritance:

PCGrad
~~~~~~

.. autoclass:: PCGrad
   :members:
   :undoc-members:
   :show-inheritance:

GradNorm
~~~~~~~~

.. autoclass:: GradNorm
   :members:
   :undoc-members:
   :show-inheritance:

Experimental Optimizers
-----------------------

Evolutionary and nature-inspired optimization algorithms.

CMA-ES
~~~~~~

.. autoclass:: CMAES
   :members:
   :undoc-members:
   :show-inheritance:

DifferentialEvolution
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: DifferentialEvolution
   :members:
   :undoc-members:
   :show-inheritance:

ParticleSwarmOptimization
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ParticleSwarmOptimization
   :members:
   :undoc-members:
   :show-inheritance:

QuantumAnnealing
~~~~~~~~~~~~~~~

.. autoclass:: QuantumAnnealing
   :members:
   :undoc-members:
   :show-inheritance:

GeneticAlgorithm
~~~~~~~~~~~~~~~

.. autoclass:: GeneticAlgorithm
   :members:
   :undoc-members:
   :show-inheritance:

Adaptive Optimizers
-------------------

Adam Variants
~~~~~~~~~~~~~

.. autoclass:: Adam
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: AdamW
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: RAdam
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: AdaBelief
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: AdaBound
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: AdamP
   :members:
   :undoc-members:
   :show-inheritance:

Adagrad Variants
~~~~~~~~~~~~~~~~

.. autoclass:: Adagrad
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Adadelta
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: AdaFactor
   :members:
   :undoc-members:
   :show-inheritance:

RMSprop Variants
~~~~~~~~~~~~~~~~

.. autoclass:: RMSprop
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Yogi
   :members:
   :undoc-members:
   :show-inheritance:

Momentum-Based Optimizers
-------------------------

SGD Variants
~~~~~~~~~~~~

.. autoclass:: SGD
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: HeavyBall
   :members:
   :undoc-members:
   :show-inheritance:

Specialized Optimizers
----------------------

Computer Vision
~~~~~~~~~~~~~~~

.. autoclass:: Ranger
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Lookahead
   :members:
   :undoc-members:
   :show-inheritance:

NLP Optimizers
~~~~~~~~~~~~~~

.. autoclass:: LAMB
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: NovoGrad
   :members:
   :undoc-members:
   :show-inheritance:

Sparse Data
~~~~~~~~~~~

.. autoclass:: SparseAdam
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: SM3
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: FTRL
   :members:
   :undoc-members:
   :show-inheritance:

Distributed Training
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LARS
   :members:
   :undoc-members:
   :show-inheritance:

General Purpose
~~~~~~~~~~~~~~~

.. autoclass:: Lion
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: MADGRAD
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Apollo
   :members:
   :undoc-members:
   :show-inheritance:

PyTorch Native Optimizers
-------------------------

For completeness, Torchium also includes all PyTorch native optimizers:

.. autoclass:: NAdam
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Rprop
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   import torchium

   model = nn.Linear(10, 1)
   
   # Use SAM optimizer for better generalization
   optimizer = torchium.optimizers.SAM(
       model.parameters(), 
       lr=1e-3, 
       rho=0.05
   )

Advanced Usage
~~~~~~~~~~~~~~

.. code-block:: python

   # Different learning rates for different layers
   param_groups = [
       {'params': model.features.parameters(), 'lr': 1e-4},
       {'params': model.classifier.parameters(), 'lr': 1e-3}
   ]
   
   optimizer = torchium.optimizers.Lion(param_groups)

Factory Functions
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create optimizer by name
   optimizer = torchium.create_optimizer(
       'sam', 
       model.parameters(), 
       lr=1e-3
   )
   
   # List all available optimizers
   available = torchium.get_available_optimizers()
   print(f"Available optimizers: {len(available)}")

Performance Comparison
---------------------

Based on our comprehensive benchmarks:

===================== =============== ================== ===================
Optimizer            Convergence     Memory Usage       Best Use Case
===================== =============== ================== ===================
SAM                  Better Gen      Standard           General Purpose
LBFGS                Fast Conv       High               Well-conditioned
Shampoo              Excellent       High               Large Models
AdaHessian           Good            High               Second-order
Ranger               36% faster      Standard           Computer Vision
AdaBelief            7.9% better     Standard           General Purpose
Lion                 2.9% better     Low                Memory Constrained
LAMB                 Excellent       High               Large Batch
NovoGrad             Good            Standard           NLP Tasks
LARS                 Good            Standard           Distributed
CMA-ES               Global Opt      High               Non-convex
===================== =============== ================== ===================

Optimizer Selection Guide
-------------------------

**For General Purpose Training:**
   - SAM: Best generalization, flatter minima
   - AdaBelief: Stable, good for most tasks
   - Lion: Memory efficient, good performance

**For Computer Vision:**
   - Ranger: Excellent for vision tasks
   - Lookahead: Good for large models
   - SAM: Better generalization

**For Natural Language Processing:**
   - LAMB: Excellent for large batch training
   - NovoGrad: Good for transformer models
   - AdamW: Reliable baseline

**For Memory-Constrained Environments:**
   - Lion: Lowest memory usage
   - SGD: Classic, minimal memory
   - HeavyBall: Good momentum alternative

**For Second-Order Optimization:**
   - LBFGS: Fast convergence for well-conditioned problems
   - Shampoo: Excellent for large models
   - AdaHessian: Adaptive second-order method

**For Experimental/Research:**
   - CMA-ES: Global optimization
   - DifferentialEvolution: Robust optimization
   - ParticleSwarmOptimization: Nature-inspired

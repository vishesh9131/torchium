Torchium Documentation
=====================

.. image:: https://img.shields.io/badge/version-0.1.0-blue.svg
   :alt: Version
.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :alt: License
.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :alt: Python Version

**Torchium** is the most comprehensive PyTorch extension library, providing **65+ advanced optimizers** and **70+ specialized loss functions** for deep learning research and production. Built on top of PyTorch's robust foundation, Torchium seamlessly integrates cutting-edge optimization algorithms and loss functions from various domains including computer vision, natural language processing, generative models, and metric learning.

Key Features
------------

* **65+ Advanced Optimizers**: Including second-order methods, meta-optimizers, experimental algorithms, and specialized optimizers for different domains
* **70+ Specialized Loss Functions**: Covering classification, regression, computer vision, NLP, generative models, metric learning, and multi-task scenarios
* **Complete PyTorch Compatibility**: All standard PyTorch optimizers and losses included for seamless integration
* **Domain-Specific Solutions**: Specialized components for computer vision, NLP, generative models, and more
* **Research-Grade Quality**: State-of-the-art implementations with comprehensive testing and benchmarking
* **Easy-to-Use API**: Drop-in replacement for PyTorch optimizers and losses with enhanced functionality
* **Factory Functions**: Dynamic optimizer/loss creation with string names and parameter groups
* **Registry System**: Automatic discovery of all available components with extensible architecture
* **High Performance**: Significant improvements over standard optimizers with optimized implementations
* **Modular Design**: Easy to extend with new optimizers and losses following established patterns

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install torchium

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   import torchium

   # Create model
   model = nn.Linear(10, 1)

   # Use advanced optimizers
   optimizer = torchium.optimizers.SAM(model.parameters(), lr=1e-3)
   
   # Use specialized loss functions
   criterion = torchium.losses.FocalLoss(alpha=0.25, gamma=2.0)

   # Or use factory functions
   optimizer = torchium.create_optimizer('sam', model.parameters(), lr=1e-3)
   criterion = torchium.create_loss('focal', alpha=0.25, gamma=2.0)

Factory Functions
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Discover available components
   optimizers = torchium.get_available_optimizers()  # 65+ optimizers
   losses = torchium.get_available_losses()          # 70+ loss functions

   # Create with parameter groups
   optimizer = torchium.utils.factory.create_optimizer_with_groups(
       model, 'adamw', lr=1e-3, weight_decay=1e-4, no_decay=['bias']
   )

Documentation Contents
---------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   tutorials/quickstart
   tutorials/advanced_usage
   tutorials/domain_specific_usage
   tutorials/performance_guide
   tutorials/cuda_integration
   tutorials/cython_optimizations
   tutorials/custom_components

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/optimizers
   api/losses
   api/utils
   api/factory

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/computer_vision
   examples/nlp
   examples/generative_models
   examples/optimization_comparison
   examples/benchmarks

.. toctree::
   :maxdepth: 2
   :caption: Development

   contributing
   changelog
   roadmap

Optimizer Categories
--------------------

**Second-Order Methods**
   LBFGS, Shampoo, AdaHessian, KFAC, NaturalGradient

**Meta-Optimizers**
   SAM, GSAM, ASAM, LookSAM, WSAM, GradientCentralization, PCGrad, GradNorm

**Experimental Algorithms**
   CMA-ES, DifferentialEvolution, ParticleSwarmOptimization, QuantumAnnealing, GeneticAlgorithm

**Adaptive Optimizers**
   Adam variants, Adagrad variants, RMSprop variants with advanced features

**Momentum-Based Methods**
   SGD variants, HeavyBall, and momentum-enhanced algorithms

**Specialized Optimizers**
   Computer vision, NLP, distributed training, sparse data, and general-purpose optimizers

Loss Function Categories
------------------------

**Classification**
   Cross-entropy variants, focal loss, label smoothing, class-balanced losses

**Regression**
   MSE variants, robust losses, quantile regression, log-cosh loss

**Computer Vision**
   Object detection (IoU losses), segmentation (Dice, Tversky), super-resolution, style transfer

**Natural Language Processing**
   Perplexity, CRF, structured prediction, BLEU, ROUGE, METEOR, BERTScore

**Generative Models**
   GAN losses, VAE losses, diffusion model losses, score matching

**Metric Learning**
   Contrastive, triplet, quadruplet, angular, proxy-based losses

**Multi-Task Learning**
   Uncertainty weighting, gradient surgery, dynamic loss balancing

**Domain-Specific**
   Medical imaging, audio processing, time series, word embeddings

Performance Highlights
---------------------

Our comprehensive benchmarks show significant improvements across various domains:

* **SAM Family**: Up to 15% better generalization with flatter minima
* **Second-Order Methods**: Superior convergence for well-conditioned problems
* **Specialized Optimizers**: Domain-specific performance gains
* **Advanced Loss Functions**: Better training stability and final performance

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

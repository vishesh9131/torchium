Quickstart Guide
================

This guide will get you started with Torchium in 5 minutes. Torchium provides 65+ advanced optimizers and 70+ specialized loss functions, extending PyTorch with state-of-the-art algorithms from recent research.

Installation
------------

.. code-block:: bash

   pip install torchium

Basic Usage
-----------

Simple Optimizer Usage
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   import torchium

   # Create your model
   model = nn.Sequential(
       nn.Linear(10, 64),
       nn.ReLU(),
       nn.Linear(64, 1)
   )

   # Use advanced optimizers - SAM for better generalization
   optimizer = torchium.optimizers.SAM(model.parameters(), lr=1e-3, rho=0.05)

   # Or use the factory function
   optimizer = torchium.create_optimizer('sam', model.parameters(), lr=1e-3)

Simple Loss Function Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use specialized loss functions - Focal loss for class imbalance
   criterion = torchium.losses.FocalLoss(alpha=0.25, gamma=2.0)

   # Or use the factory function
   criterion = torchium.create_loss('focal', alpha=0.25, gamma=2.0)

Complete Training Example
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   import torchium

   # Data
   x = torch.randn(100, 10)
   y = torch.randn(100, 1)

   # Model
   model = nn.Linear(10, 1)

   # Advanced optimizer and loss
   optimizer = torchium.optimizers.SAM(model.parameters(), lr=1e-3, rho=0.05)
   criterion = torchium.losses.HuberLoss(delta=1.0)

   # Training loop
   for epoch in range(100):
       optimizer.zero_grad()
       output = model(x)
       loss = criterion(output, y)
       loss.backward()
       optimizer.step()
       
       if epoch % 10 == 0:
           print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

Factory Functions
-----------------

Torchium provides convenient factory functions for dynamic creation:

.. code-block:: python

   # List available optimizers
   optimizers = torchium.get_available_optimizers()
   print(f"Available optimizers: {len(optimizers)}")

   # List available losses
   losses = torchium.get_available_losses()
   print(f"Available losses: {len(losses)}")

   # Create optimizer by name
   optimizer = torchium.create_optimizer('adabelief', model.parameters(), lr=1e-3)

   # Create loss by name
   criterion = torchium.create_loss('dice', smooth=1e-5)

Advanced Features
-----------------

Parameter Groups
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Different learning rates for different layers
   param_groups = [
       {'params': model.features.parameters(), 'lr': 1e-4},
       {'params': model.classifier.parameters(), 'lr': 1e-3}
   ]
   
   optimizer = torchium.optimizers.Lion(param_groups)

Specialized Optimizers by Domain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For computer vision - Ranger optimizer
   optimizer = torchium.optimizers.Ranger(model.parameters(), lr=1e-3)

   # For NLP - LAMB optimizer
   optimizer = torchium.optimizers.LAMB(model.parameters(), lr=1e-3)

   # For memory efficiency - Lion optimizer
   optimizer = torchium.optimizers.Lion(model.parameters(), lr=1e-4)

   # For second-order optimization - LBFGS
   optimizer = torchium.optimizers.LBFGS(model.parameters(), lr=1.0)

   # For meta-optimization - SAM family
   optimizer = torchium.optimizers.SAM(model.parameters(), lr=1e-3, rho=0.05)

Specialized Loss Functions by Domain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For segmentation - Dice loss
   criterion = torchium.losses.DiceLoss(smooth=1e-5)

   # For classification with imbalanced data - Focal loss
   criterion = torchium.losses.FocalLoss(alpha=0.25, gamma=2.0)

   # For regression with outliers - Huber loss
   criterion = torchium.losses.HuberLoss(delta=1.0)

   # For object detection - GIoU loss
   criterion = torchium.losses.GIoULoss()

   # For generative models - GAN loss
   criterion = torchium.losses.GANLoss()

   # For metric learning - Triplet loss
   criterion = torchium.losses.TripletMetricLoss(margin=0.3)

   # For multi-task learning - Uncertainty weighting
   criterion = torchium.losses.UncertaintyWeightingLoss(num_tasks=3)

Computer Vision Example
-----------------------

Object Detection with Advanced Losses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   import torchium

   # Model for object detection
   class DetectionModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.backbone = nn.Sequential(
               nn.Conv2d(3, 64, 3, padding=1),
               nn.ReLU(),
               nn.Conv2d(64, 128, 3, padding=1),
               nn.ReLU()
           )
           self.classifier = nn.Linear(128, 10)  # 10 classes
           self.regressor = nn.Linear(128, 4)    # 4 bbox coordinates

   model = DetectionModel()

   # Use Ranger optimizer for computer vision
   optimizer = torchium.optimizers.Ranger(model.parameters(), lr=1e-3)

   # Combined loss for detection
   class DetectionLoss(nn.Module):
       def __init__(self):
           super().__init__()
           self.cls_loss = torchium.losses.FocalLoss(alpha=0.25, gamma=2.0)
           self.reg_loss = torchium.losses.GIoULoss()

       def forward(self, cls_pred, reg_pred, cls_target, reg_target):
           cls_loss = self.cls_loss(cls_pred, cls_target)
           reg_loss = self.reg_loss(reg_pred, reg_target)
           return cls_loss + reg_loss

   criterion = DetectionLoss()

Segmentation Example
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Segmentation model
   class SegmentationModel(nn.Module):
       def __init__(self, num_classes=21):
           super().__init__()
           self.encoder = nn.Sequential(
               nn.Conv2d(3, 64, 3, padding=1),
               nn.ReLU(),
               nn.Conv2d(64, 128, 3, padding=1),
               nn.ReLU()
           )
           self.decoder = nn.Conv2d(128, num_classes, 1)

   model = SegmentationModel()

   # Use SAM for better generalization
   optimizer = torchium.optimizers.SAM(model.parameters(), lr=1e-3, rho=0.05)

   # Combined segmentation loss
   class SegmentationLoss(nn.Module):
       def __init__(self):
           super().__init__()
           self.dice = torchium.losses.DiceLoss(smooth=1e-5)
           self.focal = torchium.losses.FocalLoss(alpha=0.25, gamma=2.0)

       def forward(self, pred, target):
           dice_loss = self.dice(pred, target)
           focal_loss = self.focal(pred, target)
           return 0.6 * dice_loss + 0.4 * focal_loss

   criterion = SegmentationLoss()

NLP Example
-----------

Transformer Training
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   import torchium

   # Simple transformer-like model
   class TransformerModel(nn.Module):
       def __init__(self, vocab_size=10000, d_model=512):
           super().__init__()
           self.embedding = nn.Embedding(vocab_size, d_model)
           self.transformer = nn.TransformerEncoder(
               nn.TransformerEncoderLayer(d_model, nhead=8),
               num_layers=6
           )
           self.classifier = nn.Linear(d_model, vocab_size)

   model = TransformerModel()

   # Use LAMB optimizer for NLP
   optimizer = torchium.optimizers.LAMB(model.parameters(), lr=1e-3)

   # Use label smoothing for better generalization
   criterion = torchium.losses.LabelSmoothingLoss(
       num_classes=10000,
       smoothing=0.1
   )

Generative Models Example
-------------------------

GAN Training
~~~~~~~~~~~~

.. code-block:: python

   # Generator and Discriminator
   class Generator(nn.Module):
       def __init__(self, latent_dim=100, output_dim=784):
           super().__init__()
           self.net = nn.Sequential(
               nn.Linear(latent_dim, 256),
               nn.ReLU(),
               nn.Linear(256, 512),
               nn.ReLU(),
               nn.Linear(512, output_dim),
               nn.Tanh()
           )

   class Discriminator(nn.Module):
       def __init__(self, input_dim=784):
           super().__init__()
           self.net = nn.Sequential(
               nn.Linear(input_dim, 512),
               nn.LeakyReLU(0.2),
               nn.Linear(512, 256),
               nn.LeakyReLU(0.2),
               nn.Linear(256, 1),
               nn.Sigmoid()
           )

   generator = Generator()
   discriminator = Discriminator()

   # Use different optimizers for G and D
   g_optimizer = torchium.optimizers.Adam(generator.parameters(), lr=2e-4)
   d_optimizer = torchium.optimizers.Adam(discriminator.parameters(), lr=2e-4)

   # Use GAN loss
   criterion = torchium.losses.GANLoss()

VAE Training
~~~~~~~~~~~~

.. code-block:: python

   class VAE(nn.Module):
       def __init__(self, input_dim=784, latent_dim=20):
           super().__init__()
           self.encoder = nn.Sequential(
               nn.Linear(input_dim, 400),
               nn.ReLU(),
               nn.Linear(400, latent_dim * 2)  # mean and log_var
           )
           self.decoder = nn.Sequential(
               nn.Linear(latent_dim, 400),
               nn.ReLU(),
               nn.Linear(400, input_dim),
               nn.Sigmoid()
           )

   model = VAE()

   # Use AdaBelief for stable training
   optimizer = torchium.optimizers.AdaBelief(model.parameters(), lr=1e-3)

   # Use ELBO loss for VAE
   criterion = torchium.losses.ELBOLoss(kl_weight=0.1)

Metric Learning Example
-----------------------

Face Recognition
~~~~~~~~~~~~~~~~

.. code-block:: python

   class FaceRecognitionModel(nn.Module):
       def __init__(self, embedding_dim=512, num_classes=1000):
           super().__init__()
           self.backbone = nn.Sequential(
               nn.Conv2d(3, 64, 3, padding=1),
               nn.ReLU(),
               nn.AdaptiveAvgPool2d(1)
           )
           self.embedding = nn.Linear(64, embedding_dim)
           self.classifier = nn.Linear(embedding_dim, num_classes)

   model = FaceRecognitionModel()

   # Use Lion for memory efficiency
   optimizer = torchium.optimizers.Lion(model.parameters(), lr=1e-4)

   # Use ArcFace loss for face recognition
   criterion = torchium.losses.ArcFaceMetricLoss(
       num_classes=1000,
       embedding_size=512,
       margin=0.5,
       scale=64
   )

Multi-Task Learning Example
---------------------------

Uncertainty Weighting
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class MultiTaskModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.shared = nn.Sequential(
               nn.Linear(10, 64),
               nn.ReLU()
           )
           self.task1 = nn.Linear(64, 1)  # Regression
           self.task2 = nn.Linear(64, 5)  # Classification

   model = MultiTaskModel()

   # Use SAM for better generalization
   optimizer = torchium.optimizers.SAM(model.parameters(), lr=1e-3, rho=0.05)

   # Use uncertainty weighting for multi-task
   criterion = torchium.losses.UncertaintyWeightingLoss(num_tasks=2)

Quick Benchmark
---------------

Run a quick benchmark to see which optimizer works best for your task:

.. code-block:: python

   from torchium.benchmarks import QuickBenchmark

   benchmark = QuickBenchmark()
   results = benchmark.simple_regression_benchmark()

   # Compare different optimizers
   optimizers_to_test = ['adam', 'adamw', 'sam', 'ranger', 'lion']
   results = benchmark.compare_optimizers(optimizers_to_test)

Next Steps
----------

- Check out the :doc:`/api/optimizers` for all 65+ available optimizers
- Explore :doc:`/api/losses` for 70+ specialized loss functions
- Run :doc:`/examples/benchmarks` to find the best optimizer for your task
- Learn about :doc:`advanced_usage` for expert features
- Explore :doc:`domain_specific_usage` for specialized applications

Optimizer Selection Quick Guide
------------------------------

**For General Purpose:**
   - SAM: Best generalization, flatter minima
   - AdaBelief: Stable, good for most tasks
   - Lion: Memory efficient, good performance

**For Computer Vision:**
   - Ranger: Excellent for vision tasks
   - Lookahead: Good for large models
   - SAM: Better generalization

**For NLP:**
   - LAMB: Excellent for large batch training
   - NovoGrad: Good for transformer models
   - AdamW: Reliable baseline

**For Memory-Constrained:**
   - Lion: Lowest memory usage
   - SGD: Classic, minimal memory
   - HeavyBall: Good momentum alternative

Loss Function Selection Quick Guide
----------------------------------

**For Classification:**
   - FocalLoss: Class imbalance
   - LabelSmoothingLoss: Overconfidence
   - ClassBalancedLoss: Imbalanced datasets

**For Computer Vision:**
   - DiceLoss: Segmentation
   - GIoULoss: Object detection
   - PerceptualLoss: Super resolution

**For Generative Models:**
   - GANLoss: GANs
   - ELBOLoss: VAEs
   - DDPMLoss: Diffusion models

**For Metric Learning:**
   - TripletMetricLoss: Distance learning
   - ArcFaceMetricLoss: Face recognition
   - ContrastiveMetricLoss: Contrastive learning

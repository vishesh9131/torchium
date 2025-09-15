Advanced Usage Guide
====================

This guide covers advanced features and usage patterns in Torchium, including meta-optimizers, experimental algorithms, and sophisticated loss combinations.

Meta-Optimizers
---------------

Sharpness-Aware Minimization (SAM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SAM finds flatter minima for better generalization:

.. code-block:: python

   import torch
   import torch.nn as nn
   import torchium

   model = nn.Linear(10, 1)
   
   # Basic SAM
   optimizer = torchium.optimizers.SAM(
       model.parameters(),
       lr=1e-3,
       rho=0.05,  # Perturbation radius
       adaptive=False
   )

   # Training loop with SAM
   for epoch in range(100):
       # First forward pass
       output = model(x)
       loss = criterion(output, y)
       loss.backward()
       
       # SAM perturbation step
       optimizer.first_step(zero_grad=True)
       
       # Second forward pass
       output = model(x)
       loss = criterion(output, y)
       loss.backward()
       
       # SAM update step
       optimizer.second_step(zero_grad=True)

Adaptive SAM (ASAM)
~~~~~~~~~~~~~~~~~~~

ASAM adapts the perturbation radius:

.. code-block:: python

   optimizer = torchium.optimizers.ASAM(
       model.parameters(),
       lr=1e-3,
       rho=0.5,  # Initial perturbation radius
       eta=0.01  # Adaptation rate
   )

Gradient Surgery Methods
~~~~~~~~~~~~~~~~~~~~~~~~

PCGrad for Multi-Task Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class MultiTaskModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.shared = nn.Linear(10, 64)
           self.task1 = nn.Linear(64, 1)
           self.task2 = nn.Linear(64, 5)

   model = MultiTaskModel()

   # Use PCGrad for gradient surgery
   optimizer = torchium.optimizers.PCGrad(
       model.parameters(),
       lr=1e-3
   )

   # Training with multiple tasks
   for epoch in range(100):
       optimizer.zero_grad()
       
       # Forward passes for different tasks
       shared_features = model.shared(x)
       task1_output = model.task1(shared_features)
       task2_output = model.task2(shared_features)
       
       # Compute losses
       loss1 = criterion1(task1_output, y1)
       loss2 = criterion2(task2_output, y2)
       
       # PCGrad handles gradient conflicts
       optimizer.step([loss1, loss2])

GradNorm for Dynamic Loss Balancing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   optimizer = torchium.optimizers.GradNorm(
       model.parameters(),
       lr=1e-3,
       alpha=1.5  # Restoring force hyperparameter
   )

Second-Order Optimizers
-----------------------

LBFGS for Well-Conditioned Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # LBFGS works best with full batch or large batches
   optimizer = torchium.optimizers.LBFGS(
       model.parameters(),
       lr=1.0,
       max_iter=20,
       max_eval=None,
       tolerance_grad=1e-7,
       tolerance_change=1e-9,
       history_size=100,
       line_search_fn="strong_wolfe"
   )

   # Training loop for LBFGS
   def closure():
       optimizer.zero_grad()
       output = model(x)
       loss = criterion(output, y)
       loss.backward()
       return loss

   for epoch in range(100):
       optimizer.step(closure)

Shampoo for Large Models
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   optimizer = torchium.optimizers.Shampoo(
       model.parameters(),
       lr=1e-3,
       momentum=0.9,
       weight_decay=1e-4,
       epsilon=1e-4,
       update_freq=1
   )

Experimental Optimizers
-----------------------

CMA-ES for Global Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # CMA-ES for non-convex optimization
   optimizer = torchium.optimizers.CMAES(
       model.parameters(),
       population_size=20,
       sigma=0.1,
       max_generations=1000
   )

   # Training loop for CMA-ES
   for generation in range(1000):
       optimizer.step()
       if optimizer.should_stop():
           break

Differential Evolution
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   optimizer = torchium.optimizers.DifferentialEvolution(
       model.parameters(),
       population_size=30,
       mutation_factor=0.8,
       crossover_probability=0.9,
       max_generations=1000
   )

Particle Swarm Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   optimizer = torchium.optimizers.ParticleSwarmOptimization(
       model.parameters(),
       swarm_size=20,
       inertia_weight=0.9,
       cognitive_weight=2.0,
       social_weight=2.0,
       max_iterations=1000
   )

Advanced Loss Combinations
-------------------------

Multi-Task Learning with Uncertainty Weighting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class MultiTaskLoss(nn.Module):
       def __init__(self, num_tasks):
           super().__init__()
           self.uncertainty_loss = torchium.losses.UncertaintyWeightingLoss(num_tasks)
           self.task_losses = [
               torchium.losses.MSELoss(),
               torchium.losses.CrossEntropyLoss(),
               torchium.losses.DiceLoss()
           ]

       def forward(self, predictions, targets):
           losses = []
           for i, (pred, target) in enumerate(zip(predictions, targets)):
               loss = self.task_losses[i](pred, target)
               losses.append(loss)
           
           return self.uncertainty_loss(losses)

   criterion = MultiTaskLoss(num_tasks=3)

Combined Segmentation Loss
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class CombinedSegmentationLoss(nn.Module):
       def __init__(self):
           super().__init__()
           self.dice = torchium.losses.DiceLoss(smooth=1e-5)
           self.focal = torchium.losses.FocalLoss(alpha=0.25, gamma=2.0)
           self.tversky = torchium.losses.TverskyLoss(alpha=0.3, beta=0.7)
           self.lovasz = torchium.losses.LovaszLoss()

       def forward(self, pred, target):
           dice_loss = self.dice(pred, target)
           focal_loss = self.focal(pred, target)
           tversky_loss = self.tversky(pred, target)
           lovasz_loss = self.lovasz(pred, target)
           
           # Weighted combination
           total_loss = (0.4 * dice_loss + 
                        0.3 * focal_loss + 
                        0.2 * tversky_loss + 
                        0.1 * lovasz_loss)
           
           return total_loss

   criterion = CombinedSegmentationLoss()

Generative Model Loss Combinations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class GANLossCombination(nn.Module):
       def __init__(self):
           super().__init__()
           self.gan_loss = torchium.losses.GANLoss()
           self.perceptual_loss = torchium.losses.PerceptualLoss()
           self.feature_matching_loss = torchium.losses.FeatureMatchingLoss()

       def forward(self, fake_pred, real_pred, fake_features, real_features):
           gan_loss = self.gan_loss(fake_pred, real_pred)
           perceptual_loss = self.perceptual_loss(fake_features, real_features)
           feature_matching_loss = self.feature_matching_loss(fake_features, real_features)
           
           return gan_loss + 0.1 * perceptual_loss + 0.1 * feature_matching_loss

Custom Parameter Groups
-----------------------

Advanced Parameter Grouping
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Different optimizers for different parts
   param_groups = [
       {
           'params': model.backbone.parameters(),
           'lr': 1e-4,
           'weight_decay': 1e-4
       },
       {
           'params': model.classifier.parameters(),
           'lr': 1e-3,
           'weight_decay': 1e-5
       },
       {
           'params': model.bn.parameters(),
           'lr': 1e-3,
           'weight_decay': 0  # No weight decay for batch norm
       }
   ]

   optimizer = torchium.optimizers.AdamW(param_groups)

   # Or use factory function for complex grouping
   optimizer = torchium.utils.factory.create_optimizer_with_groups(
       model,
       'adamw',
       lr=1e-3,
       weight_decay=1e-4,
       no_decay=['bias', 'bn', 'ln']  # Exclude these from weight decay
   )

Learning Rate Scheduling
------------------------

Custom Learning Rate Schedules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Warmup + cosine annealing
   def get_lr_scheduler(optimizer, warmup_epochs, total_epochs):
       def lr_lambda(epoch):
           if epoch < warmup_epochs:
               return epoch / warmup_epochs
           else:
               return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
       
       return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

   scheduler = get_lr_scheduler(optimizer, warmup_epochs=10, total_epochs=100)

   # Training loop with scheduler
   for epoch in range(100):
       # Training step
       train_one_epoch(model, optimizer, criterion, dataloader)
       
       # Update learning rate
       scheduler.step()

Gradient Clipping
-----------------

Advanced Gradient Clipping
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Gradient clipping with different methods
   def train_with_clipping(model, optimizer, criterion, dataloader, max_norm=1.0):
       for batch in dataloader:
           optimizer.zero_grad()
           output = model(batch.input)
           loss = criterion(output, batch.target)
           loss.backward()
           
           # Gradient clipping
           torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
           
           optimizer.step()

   # Or use built-in clipping for some optimizers
   optimizer = torchium.optimizers.AdamW(
       model.parameters(),
       lr=1e-3,
       max_grad_norm=1.0  # Built-in gradient clipping
   )

Mixed Precision Training
------------------------

Automatic Mixed Precision
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from torch.cuda.amp import autocast, GradScaler

   scaler = GradScaler()

   for epoch in range(100):
       for batch in dataloader:
           optimizer.zero_grad()
           
           with autocast():
               output = model(batch.input)
               loss = criterion(output, batch.target)
           
           scaler.scale(loss).backward()
           scaler.step(optimizer)
           scaler.update()

Distributed Training
--------------------

Multi-GPU Training
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch.distributed as dist
   from torch.nn.parallel import DistributedDataParallel as DDP

   # Initialize distributed training
   dist.init_process_group(backend='nccl')
   
   # Wrap model with DDP
   model = DDP(model)
   
   # Use LARS for distributed training
   optimizer = torchium.optimizers.LARS(
       model.parameters(),
       lr=1e-3,
       momentum=0.9,
       weight_decay=1e-4
   )

Performance Optimization
------------------------

Memory Optimization
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use Lion for memory efficiency
   optimizer = torchium.optimizers.Lion(
       model.parameters(),
       lr=1e-4,
       betas=(0.9, 0.99),
       weight_decay=1e-2
   )

   # Gradient checkpointing for large models
   from torch.utils.checkpoint import checkpoint

   class CheckpointedModel(nn.Module):
       def forward(self, x):
           return checkpoint(self._forward, x)

       def _forward(self, x):
           # Your model forward pass
           return self.layers(x)

Profiling and Debugging
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Profile optimizer performance
   import torch.profiler

   with torch.profiler.profile(
       activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
       schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
       on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler')
   ) as prof:
       for step, batch in enumerate(dataloader):
           optimizer.zero_grad()
           output = model(batch.input)
           loss = criterion(output, batch.target)
           loss.backward()
           optimizer.step()
           prof.step()

Best Practices
--------------

1. **Choose the Right Optimizer:**
   - SAM for better generalization
   - Lion for memory efficiency
   - LBFGS for well-conditioned problems
   - CMA-ES for global optimization

2. **Combine Losses Wisely:**
   - Use uncertainty weighting for multi-task learning
   - Combine complementary losses (e.g., Dice + Focal)
   - Balance loss weights carefully

3. **Parameter Grouping:**
   - Different learning rates for different layers
   - Exclude batch norm from weight decay
   - Use appropriate weight decay values

4. **Learning Rate Scheduling:**
   - Use warmup for stable training
   - Cosine annealing for better convergence
   - Monitor learning rate during training

5. **Gradient Management:**
   - Use gradient clipping for stability
   - Monitor gradient norms
   - Use gradient surgery for multi-task learning

6. **Memory Management:**
   - Use Lion for memory efficiency
   - Gradient checkpointing for large models
   - Mixed precision training when possible

Performance Guide
=================

This guide covers performance optimization techniques, benchmarking methodologies, and best practices for getting the most out of Torchium's optimizers and loss functions.

Benchmarking Methodology
------------------------

Quick Benchmarking
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from torchium.benchmarks import QuickBenchmark

   # Initialize benchmark
   benchmark = QuickBenchmark()

   # Run simple regression benchmark
   results = benchmark.simple_regression_benchmark()

   # Compare specific optimizers
   optimizers_to_test = ['adam', 'adamw', 'sam', 'ranger', 'lion', 'adabelief']
   results = benchmark.compare_optimizers(optimizers_to_test)

   # Print results
   for optimizer_name, metrics in results.items():
       print(f"{optimizer_name}:")
       print(f"  Final Loss: {metrics['final_loss']:.6f}")
       print(f"  Convergence Time: {metrics['convergence_time']:.2f}s")
       print(f"  Memory Usage: {metrics['memory_usage']:.2f}MB")

Comprehensive Benchmarking
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from torchium.benchmarks import OptimizerBenchmark

   # Initialize comprehensive benchmark
   benchmark = OptimizerBenchmark()

   # Test on different tasks
   tasks = ['regression', 'classification', 'computer_vision', 'nlp']
   
   for task in tasks:
       print(f"\nBenchmarking {task}...")
       results = benchmark.benchmark_task(task)
       
       # Analyze results
       best_optimizer = min(results.items(), key=lambda x: x[1]['final_loss'])
       print(f"Best optimizer for {task}: {best_optimizer[0]}")
       print(f"Final loss: {best_optimizer[1]['final_loss']:.6f}")

Performance Metrics
-------------------

Convergence Speed
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   import torch
   import torchium

   def benchmark_convergence(model, optimizer, criterion, dataloader, target_loss=0.01):
       start_time = time.time()
       epoch = 0
       
       while True:
           total_loss = 0
           for batch in dataloader:
               optimizer.zero_grad()
               output = model(batch.input)
               loss = criterion(output, batch.target)
               loss.backward()
               optimizer.step()
               total_loss += loss.item()
           
           avg_loss = total_loss / len(dataloader)
           epoch += 1
           
           if avg_loss < target_loss or epoch > 1000:
               break
       
       convergence_time = time.time() - start_time
       return convergence_time, avg_loss, epoch

   # Test different optimizers
   optimizers = {
       'adam': torchium.optimizers.Adam(model.parameters(), lr=1e-3),
       'sam': torchium.optimizers.SAM(model.parameters(), lr=1e-3, rho=0.05),
       'ranger': torchium.optimizers.Ranger(model.parameters(), lr=1e-3),
       'lion': torchium.optimizers.Lion(model.parameters(), lr=1e-4)
   }

   for name, optimizer in optimizers.items():
       time_taken, final_loss, epochs = benchmark_convergence(
           model, optimizer, criterion, dataloader
       )
       print(f"{name}: {time_taken:.2f}s, {final_loss:.6f}, {epochs} epochs")

Memory Usage
~~~~~~~~~~~~

.. code-block:: python

   import psutil
   import torch

   def benchmark_memory(model, optimizer, criterion, dataloader, num_batches=100):
       process = psutil.Process()
       initial_memory = process.memory_info().rss / 1024 / 1024  # MB
       
       max_memory = initial_memory
       
       for i, batch in enumerate(dataloader):
           if i >= num_batches:
               break
               
           optimizer.zero_grad()
           output = model(batch.input)
           loss = criterion(output, batch.target)
           loss.backward()
           optimizer.step()
           
           current_memory = process.memory_info().rss / 1024 / 1024  # MB
           max_memory = max(max_memory, current_memory)
       
       return max_memory - initial_memory

   # Test memory usage
   for name, optimizer in optimizers.items():
       memory_usage = benchmark_memory(model, optimizer, criterion, dataloader)
       print(f"{name}: {memory_usage:.2f}MB peak memory usage")

Optimization Strategies
-----------------------

Learning Rate Scheduling
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import math

   # Warmup + cosine annealing
   def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs):
       def lr_lambda(epoch):
           if epoch < warmup_epochs:
               return epoch / warmup_epochs
           else:
               return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
       
       return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

   # One-cycle learning rate
   def get_one_cycle_scheduler(optimizer, max_lr, total_epochs):
       return torch.optim.lr_scheduler.OneCycleLR(
           optimizer, max_lr=max_lr, total_steps=total_epochs
       )

   # Apply scheduling
   optimizer = torchium.optimizers.SAM(model.parameters(), lr=1e-3, rho=0.05)
   scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs=10, total_epochs=100)

   for epoch in range(100):
       train_one_epoch(model, optimizer, criterion, dataloader)
       scheduler.step()

Gradient Clipping
~~~~~~~~~~~~~~~~~

.. code-block:: python

   def train_with_gradient_clipping(model, optimizer, criterion, dataloader, max_norm=1.0):
       for batch in dataloader:
           optimizer.zero_grad()
           output = model(batch.input)
           loss = criterion(output, batch.target)
           loss.backward()
           
           # Gradient clipping
           torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
           
           optimizer.step()

   # Or use built-in clipping
   optimizer = torchium.optimizers.AdamW(
       model.parameters(),
       lr=1e-3,
       max_grad_norm=1.0
   )

Mixed Precision Training
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from torch.cuda.amp import autocast, GradScaler

   scaler = GradScaler()

   def train_with_mixed_precision(model, optimizer, criterion, dataloader):
       for batch in dataloader:
           optimizer.zero_grad()
           
           with autocast():
               output = model(batch.input)
               loss = criterion(output, batch.target)
           
           scaler.scale(loss).backward()
           scaler.step(optimizer)
           scaler.update()

   # Use Lion for memory efficiency with mixed precision
   optimizer = torchium.optimizers.Lion(model.parameters(), lr=1e-4)

Domain-Specific Performance
---------------------------

Computer Vision
~~~~~~~~~~~~~~~

.. code-block:: python

   # Computer vision specific benchmarking
   def benchmark_vision_optimizers(model, dataloader):
       optimizers = {
           'ranger': torchium.optimizers.Ranger(model.parameters(), lr=1e-3),
           'lookahead': torchium.optimizers.Lookahead(model.parameters(), lr=1e-3),
           'sam': torchium.optimizers.SAM(model.parameters(), lr=1e-3, rho=0.05),
           'adamw': torchium.optimizers.AdamW(model.parameters(), lr=1e-3)
       }
       
       results = {}
       for name, optimizer in optimizers.items():
           start_time = time.time()
           final_loss = train_until_convergence(model, optimizer, dataloader)
           training_time = time.time() - start_time
           
           results[name] = {
               'final_loss': final_loss,
               'training_time': training_time
           }
       
       return results

   # Vision-specific loss combinations
   class OptimizedVisionLoss(nn.Module):
       def __init__(self):
           super().__init__()
           self.dice = torchium.losses.DiceLoss(smooth=1e-5)
           self.focal = torchium.losses.FocalLoss(alpha=0.25, gamma=2.0)
           self.giou = torchium.losses.GIoULoss()

       def forward(self, pred, target, task_type='segmentation'):
           if task_type == 'segmentation':
               return 0.6 * self.dice(pred, target) + 0.4 * self.focal(pred, target)
           elif task_type == 'detection':
               return self.giou(pred, target)
           else:
               return self.focal(pred, target)

Natural Language Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # NLP specific benchmarking
   def benchmark_nlp_optimizers(model, dataloader):
       optimizers = {
           'lamb': torchium.optimizers.LAMB(model.parameters(), lr=1e-3),
           'novograd': torchium.optimizers.NovoGrad(model.parameters(), lr=1e-3),
           'adamw': torchium.optimizers.AdamW(model.parameters(), lr=1e-3),
           'sam': torchium.optimizers.SAM(model.parameters(), lr=1e-3, rho=0.05)
       }
       
       results = {}
       for name, optimizer in optimizers.items():
           start_time = time.time()
           final_loss = train_until_convergence(model, optimizer, dataloader)
           training_time = time.time() - start_time
           
           results[name] = {
               'final_loss': final_loss,
               'training_time': training_time
           }
       
       return results

   # NLP-specific loss with label smoothing
   criterion = torchium.losses.LabelSmoothingLoss(
       num_classes=50000,
       smoothing=0.1
   )

Generative Models
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # GAN specific benchmarking
   def benchmark_gan_optimizers(generator, discriminator, dataloader):
       g_optimizers = {
           'adam': torchium.optimizers.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999)),
           'rmsprop': torchium.optimizers.RMSprop(generator.parameters(), lr=2e-4),
           'lion': torchium.optimizers.Lion(generator.parameters(), lr=1e-4)
       }
       
       d_optimizers = {
           'adam': torchium.optimizers.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999)),
           'rmsprop': torchium.optimizers.RMSprop(discriminator.parameters(), lr=2e-4),
           'lion': torchium.optimizers.Lion(discriminator.parameters(), lr=1e-4)
       }
       
       results = {}
       for g_name, g_opt in g_optimizers.items():
           for d_name, d_opt in d_optimizers.items():
               combo_name = f"G_{g_name}_D_{d_name}"
               start_time = time.time()
               final_loss = train_gan_until_convergence(generator, discriminator, g_opt, d_opt, dataloader)
               training_time = time.time() - start_time
               
               results[combo_name] = {
                   'final_loss': final_loss,
                   'training_time': training_time
               }
       
       return results

   # GAN-specific loss
   criterion = torchium.losses.GANLoss()

Memory Optimization
-------------------

Memory-Efficient Training
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use Lion for memory efficiency
   optimizer = torchium.optimizers.Lion(
       model.parameters(),
       lr=1e-4,
       betas=(0.9, 0.99),
       weight_decay=1e-2
   )

   # Gradient checkpointing
   from torch.utils.checkpoint import checkpoint

   class CheckpointedModel(nn.Module):
       def forward(self, x):
           return checkpoint(self._forward, x)

       def _forward(self, x):
           return self.layers(x)

   # Memory-efficient loss computation
   class MemoryEfficientLoss(nn.Module):
       def __init__(self):
           super().__init__()
           self.dice = torchium.losses.DiceLoss(smooth=1e-5)
           self.focal = torchium.losses.FocalLoss(alpha=0.25, gamma=2.0)

       def forward(self, pred, target):
           # Compute losses separately to save memory
           dice_loss = self.dice(pred, target)
           focal_loss = self.focal(pred, target)
           return 0.6 * dice_loss + 0.4 * focal_loss

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

   # Distributed training loop
   for epoch in range(100):
       for batch in dataloader:
           optimizer.zero_grad()
           output = model(batch.input)
           loss = criterion(output, batch.target)
           loss.backward()
           optimizer.step()

Performance Monitoring
----------------------

Training Monitoring
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import wandb
   import time

   def monitor_training(model, optimizer, criterion, dataloader, num_epochs=100):
       wandb.init(project="torchium-performance")
       
       for epoch in range(num_epochs):
           epoch_start = time.time()
           total_loss = 0
           
           for batch in dataloader:
               optimizer.zero_grad()
               output = model(batch.input)
               loss = criterion(output, batch.target)
               loss.backward()
               optimizer.step()
               total_loss += loss.item()
           
           epoch_time = time.time() - epoch_start
           avg_loss = total_loss / len(dataloader)
           
           # Log metrics
           wandb.log({
               'epoch': epoch,
               'loss': avg_loss,
               'epoch_time': epoch_time,
               'learning_rate': optimizer.param_groups[0]['lr']
           })

   # Monitor different optimizers
   optimizers = {
       'sam': torchium.optimizers.SAM(model.parameters(), lr=1e-3, rho=0.05),
       'ranger': torchium.optimizers.Ranger(model.parameters(), lr=1e-3),
       'lion': torchium.optimizers.Lion(model.parameters(), lr=1e-4)
   }

   for name, optimizer in optimizers.items():
       print(f"Monitoring {name}...")
       monitor_training(model, optimizer, criterion, dataloader)

Profiling
~~~~~~~~~

.. code-block:: python

   import torch.profiler

   def profile_optimizer(optimizer, model, criterion, dataloader):
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

   # Profile different optimizers
   for name, optimizer in optimizers.items():
       print(f"Profiling {name}...")
       profile_optimizer(optimizer, model, criterion, dataloader)

Best Practices
--------------

1. **Optimizer Selection:**
   - Use SAM for better generalization
   - Use Lion for memory efficiency
   - Use LAMB for large batch training
   - Use Ranger for computer vision

2. **Learning Rate Scheduling:**
   - Use warmup for stable training
   - Apply cosine annealing for better convergence
   - Monitor learning rate during training

3. **Gradient Management:**
   - Use gradient clipping for stability
   - Monitor gradient norms
   - Apply gradient surgery for multi-task learning

4. **Memory Management:**
   - Use Lion for memory efficiency
   - Apply gradient checkpointing for large models
   - Use mixed precision training when possible

5. **Performance Monitoring:**
   - Monitor training metrics
   - Profile optimizer performance
   - Use distributed training for large models

6. **Domain-Specific Optimization:**
   - Use appropriate optimizers for each domain
   - Apply domain-specific loss combinations
   - Consider task-specific hyperparameters

Performance Comparison Table
----------------------------

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

This performance guide provides comprehensive strategies for optimizing your training with Torchium's advanced optimizers and loss functions. Choose the right combination based on your specific requirements and domain.

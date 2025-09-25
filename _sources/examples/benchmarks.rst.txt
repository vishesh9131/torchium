Benchmarks Examples
===================

This section provides comprehensive examples for running benchmarks and performance tests using Torchium's benchmarking tools.

Quick Benchmarks
----------------

Simple Regression Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   import torchium
   from torchium.benchmarks import QuickBenchmark

   # Initialize benchmark
   benchmark = QuickBenchmark()

   # Run simple regression benchmark
   results = benchmark.simple_regression_benchmark()

   # Print results
   print("Simple Regression Benchmark Results:")
   for optimizer_name, metrics in results.items():
       print(f"{optimizer_name}:")
       print(f"  Final Loss: {metrics['final_loss']:.6f}")
       print(f"  Convergence Time: {metrics['convergence_time']:.2f}s")
       print(f"  Memory Usage: {metrics['memory_usage']:.2f}MB")

   # Compare specific optimizers
   optimizers_to_test = ['adam', 'adamw', 'sam', 'ranger', 'lion', 'adabelief']
   results = benchmark.compare_optimizers(optimizers_to_test)

   # Print comparison results
   print("\nOptimizer Comparison Results:")
   for optimizer_name, metrics in results.items():
       print(f"{optimizer_name}:")
       print(f"  Final Loss: {metrics['final_loss']:.6f}")
       print(f"  Convergence Time: {metrics['convergence_time']:.2f}s")
       print(f"  Memory Usage: {metrics['memory_usage']:.2f}MB")

Classification Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Run classification benchmark
   results = benchmark.classification_benchmark()

   # Print results
   print("Classification Benchmark Results:")
   for optimizer_name, metrics in results.items():
       print(f"{optimizer_name}:")
       print(f"  Final Accuracy: {metrics['final_accuracy']:.2f}%")
       print(f"  Final Loss: {metrics['final_loss']:.6f}")
       print(f"  Convergence Time: {metrics['convergence_time']:.2f}s")

   # Compare specific optimizers for classification
   optimizers_to_test = ['adam', 'adamw', 'sam', 'ranger', 'lion', 'adabelief']
   results = benchmark.compare_optimizers(optimizers_to_test, task='classification')

   # Print comparison results
   print("\nClassification Optimizer Comparison Results:")
   for optimizer_name, metrics in results.items():
       print(f"{optimizer_name}:")
       print(f"  Final Accuracy: {metrics['final_accuracy']:.2f}%")
       print(f"  Final Loss: {metrics['final_loss']:.6f}")
       print(f"  Convergence Time: {metrics['convergence_time']:.2f}s")

Comprehensive Benchmarks
------------------------

Multi-Task Benchmarking
~~~~~~~~~~~~~~~~~~~~~~~

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

   # Compare all optimizers across all tasks
   all_results = benchmark.benchmark_all_tasks()
   
   # Print summary
   print("\nOverall Results Summary:")
   for task, task_results in all_results.items():
       print(f"\n{task.upper()}:")
       sorted_results = sorted(task_results.items(), key=lambda x: x[1]['final_loss'])
       for i, (optimizer, metrics) in enumerate(sorted_results[:5]):  # Top 5
           print(f"  {i+1}. {optimizer}: {metrics['final_loss']:.6f}")

Domain-Specific Benchmarks
--------------------------

Computer Vision Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def computer_vision_benchmark():
       # Test optimizers on computer vision tasks
       optimizers = {
           'ranger': torchium.optimizers.Ranger,
           'lookahead': torchium.optimizers.Lookahead,
           'sam': torchium.optimizers.SAM,
           'adamw': torchium.optimizers.AdamW
       }
       
       results = {}
       
       for optimizer_name, optimizer_class in optimizers.items():
           print(f"Benchmarking {optimizer_name} on computer vision...")
           
           # Create model
           model = nn.Sequential(
               nn.Conv2d(3, 64, 3, padding=1),
               nn.ReLU(),
               nn.AdaptiveAvgPool2d(1),
               nn.Flatten(),
               nn.Linear(64, 10)
           )
           
           # Create optimizer
           optimizer = optimizer_class(model.parameters(), lr=1e-3)
           criterion = nn.CrossEntropyLoss()
           
           # Training loop
           start_time = time.time()
           model.train()
           
           for epoch in range(100):
               total_loss = 0
               correct = 0
               total = 0
               
               for batch in dataloader:
                   optimizer.zero_grad()
                   output = model(batch.images)
                   loss = criterion(output, batch.labels)
                   loss.backward()
                   optimizer.step()
                   
                   total_loss += loss.item()
                   _, predicted = torch.max(output.data, 1)
                   total += batch.labels.size(0)
                   correct += (predicted == batch.labels).sum().item()
               
               if epoch % 20 == 0:
                   accuracy = 100 * correct / total
                   print(f"  Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%")
           
           training_time = time.time() - start_time
           final_accuracy = 100 * correct / total
           
           results[optimizer_name] = {
               'final_accuracy': final_accuracy,
               'final_loss': total_loss / len(dataloader),
               'training_time': training_time
           }
       
       return results

   # Run computer vision benchmark
   results = computer_vision_benchmark()

   # Print results
   print("\nComputer Vision Benchmark Results:")
   sorted_results = sorted(results.items(), key=lambda x: x[1]['final_accuracy'], reverse=True)
   for i, (optimizer, metrics) in enumerate(sorted_results):
       print(f"{i+1}. {optimizer}:")
       print(f"   Final Accuracy: {metrics['final_accuracy']:.2f}%")
       print(f"   Final Loss: {metrics['final_loss']:.6f}")
       print(f"   Training Time: {metrics['training_time']:.2f}s")

NLP Benchmark
~~~~~~~~~~~~~

.. code-block:: python

   def nlp_benchmark():
       # Test optimizers on NLP tasks
       optimizers = {
           'lamb': torchium.optimizers.LAMB,
           'novograd': torchium.optimizers.NovoGrad,
           'adamw': torchium.optimizers.AdamW,
           'sam': torchium.optimizers.SAM
       }
       
       results = {}
       
       for optimizer_name, optimizer_class in optimizers.items():
           print(f"Benchmarking {optimizer_name} on NLP...")
           
           # Create model
           model = nn.Sequential(
               nn.Embedding(10000, 128),
               nn.LSTM(128, 64, batch_first=True),
               nn.Linear(64, 2)
           )
           
           # Create optimizer
           optimizer = optimizer_class(model.parameters(), lr=1e-3)
           criterion = nn.CrossEntropyLoss()
           
           # Training loop
           start_time = time.time()
           model.train()
           
           for epoch in range(100):
               total_loss = 0
               correct = 0
               total = 0
               
               for batch in dataloader:
                   optimizer.zero_grad()
                   output, _ = model(batch.input_ids)
                   output = output[:, -1, :]  # Use last hidden state
                   loss = criterion(output, batch.labels)
                   loss.backward()
                   
                   # Gradient clipping
                   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                   
                   optimizer.step()
                   
                   total_loss += loss.item()
                   _, predicted = torch.max(output.data, 1)
                   total += batch.labels.size(0)
                   correct += (predicted == batch.labels).sum().item()
               
               if epoch % 20 == 0:
                   accuracy = 100 * correct / total
                   print(f"  Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%")
           
           training_time = time.time() - start_time
           final_accuracy = 100 * correct / total
           
           results[optimizer_name] = {
               'final_accuracy': final_accuracy,
               'final_loss': total_loss / len(dataloader),
               'training_time': training_time
           }
       
       return results

   # Run NLP benchmark
   results = nlp_benchmark()

   # Print results
   print("\nNLP Benchmark Results:")
   sorted_results = sorted(results.items(), key=lambda x: x[1]['final_accuracy'], reverse=True)
   for i, (optimizer, metrics) in enumerate(sorted_results):
       print(f"{i+1}. {optimizer}:")
       print(f"   Final Accuracy: {metrics['final_accuracy']:.2f}%")
       print(f"   Final Loss: {metrics['final_loss']:.6f}")
       print(f"   Training Time: {metrics['training_time']:.2f}s")

Generative Models Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def generative_models_benchmark():
       # Test optimizers on generative models
       optimizers = {
           'adam': torchium.optimizers.Adam,
           'rmsprop': torchium.optimizers.RMSprop,
           'lion': torchium.optimizers.Lion
       }
       
       results = {}
       
       for optimizer_name, optimizer_class in optimizers.items():
           print(f"Benchmarking {optimizer_name} on generative models...")
           
           # Create generator
           generator = nn.Sequential(
               nn.Linear(100, 256),
               nn.ReLU(),
               nn.Linear(256, 512),
               nn.ReLU(),
               nn.Linear(512, 784),
               nn.Tanh()
           )
           
           # Create discriminator
           discriminator = nn.Sequential(
               nn.Linear(784, 512),
               nn.LeakyReLU(0.2),
               nn.Linear(512, 256),
               nn.LeakyReLU(0.2),
               nn.Linear(256, 1),
               nn.Sigmoid()
           )
           
           # Create optimizers
           g_optimizer = optimizer_class(generator.parameters(), lr=2e-4)
           d_optimizer = optimizer_class(discriminator.parameters(), lr=2e-4)
           
           # Training loop
           start_time = time.time()
           generator.train()
           discriminator.train()
           
           for epoch in range(100):
               g_loss_total = 0
               d_loss_total = 0
               
               for batch in dataloader:
                   real_images = batch.images
                   batch_size = real_images.size(0)
                   
                   # Train Discriminator
                   d_optimizer.zero_grad()
                   
                   # Real images
                   real_pred = discriminator(real_images)
                   real_target = torch.ones_like(real_pred)
                   
                   # Fake images
                   noise = torch.randn(batch_size, 100)
                   fake_images = generator(noise)
                   fake_pred = discriminator(fake_images.detach())
                   fake_target = torch.zeros_like(fake_pred)
                   
                   d_loss = nn.BCELoss()(fake_pred, fake_target) + nn.BCELoss()(real_pred, real_target)
                   d_loss.backward()
                   d_optimizer.step()
                   
                   # Train Generator
                   g_optimizer.zero_grad()
                   
                   fake_pred = discriminator(fake_images)
                   real_target = torch.ones_like(fake_pred)
                   
                   g_loss = nn.BCELoss()(fake_pred, real_target)
                   g_loss.backward()
                   g_optimizer.step()
                   
                   g_loss_total += g_loss.item()
                   d_loss_total += d_loss.item()
               
               if epoch % 20 == 0:
                   print(f"  Epoch {epoch}, G Loss: {g_loss_total/len(dataloader):.4f}, D Loss: {d_loss_total/len(dataloader):.4f}")
           
           training_time = time.time() - start_time
           
           results[optimizer_name] = {
               'final_g_loss': g_loss_total / len(dataloader),
               'final_d_loss': d_loss_total / len(dataloader),
               'training_time': training_time
           }
       
       return results

   # Run generative models benchmark
   results = generative_models_benchmark()

   # Print results
   print("\nGenerative Models Benchmark Results:")
   for optimizer, metrics in results.items():
       print(f"{optimizer}:")
       print(f"   Final G Loss: {metrics['final_g_loss']:.6f}")
       print(f"   Final D Loss: {metrics['final_d_loss']:.6f}")
       print(f"   Training Time: {metrics['training_time']:.2f}s")

Performance Analysis
--------------------

Memory Usage Analysis
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import psutil
   import torch

   def analyze_memory_usage(model, optimizers, dataloader, num_batches=100):
       process = psutil.Process()
       results = {}
       
       for optimizer_name, optimizer_class in optimizers.items():
           print(f"Analyzing memory usage for {optimizer_name}...")
           
           # Create model and optimizer
           model = model_class()
           optimizer = optimizer_class(model.parameters(), lr=1e-3)
           criterion = nn.MSELoss()
           
           # Measure initial memory
           initial_memory = process.memory_info().rss / 1024 / 1024  # MB
           
           # Training loop
           max_memory = initial_memory
           model.train()
           
           for i, batch in enumerate(dataloader):
               if i >= num_batches:
                   break
                   
               optimizer.zero_grad()
               output = model(batch.input)
               loss = criterion(output, batch.target)
               loss.backward()
               optimizer.step()
               
               # Measure current memory
               current_memory = process.memory_info().rss / 1024 / 1024  # MB
               max_memory = max(max_memory, current_memory)
           
           # Calculate memory usage
           memory_usage = max_memory - initial_memory
           
           results[optimizer_name] = {
               'memory_usage': memory_usage,
               'max_memory': max_memory
           }
       
       return results

   # Run memory analysis
   results = analyze_memory_usage(CustomModel, optimizers, dataloader)

   # Print results
   print("\nMemory Usage Analysis Results:")
   sorted_results = sorted(results.items(), key=lambda x: x[1]['memory_usage'])
   for i, (optimizer, metrics) in enumerate(sorted_results):
       print(f"{i+1}. {optimizer}:")
       print(f"   Memory Usage: {metrics['memory_usage']:.2f}MB")
       print(f"   Max Memory: {metrics['max_memory']:.2f}MB")

Convergence Analysis
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def analyze_convergence(model, optimizers, dataloader, num_epochs=100):
       results = {}
       
       for optimizer_name, optimizer_class in optimizers.items():
           print(f"Analyzing convergence for {optimizer_name}...")
           
           # Create model and optimizer
           model = model_class()
           optimizer = optimizer_class(model.parameters(), lr=1e-3)
           criterion = nn.MSELoss()
           
           # Training loop
           losses = []
           model.train()
           
           for epoch in range(num_epochs):
               total_loss = 0
               for batch in dataloader:
                   optimizer.zero_grad()
                   output = model(batch.input)
                   loss = criterion(output, batch.target)
                   loss.backward()
                   optimizer.step()
                   total_loss += loss.item()
               
               avg_loss = total_loss / len(dataloader)
               losses.append(avg_loss)
           
           # Analyze convergence
           final_loss = losses[-1]
           convergence_epoch = None
           for i, loss in enumerate(losses):
               if loss < final_loss * 1.1:  # Within 10% of final loss
                   convergence_epoch = i
                   break
           
           results[optimizer_name] = {
               'final_loss': final_loss,
               'convergence_epoch': convergence_epoch,
               'losses': losses
           }
       
       return results

   # Run convergence analysis
   results = analyze_convergence(CustomModel, optimizers, dataloader)

   # Print results
   print("\nConvergence Analysis Results:")
   sorted_results = sorted(results.items(), key=lambda x: x[1]['final_loss'])
   for i, (optimizer, metrics) in enumerate(sorted_results):
       print(f"{i+1}. {optimizer}:")
       print(f"   Final Loss: {metrics['final_loss']:.6f}")
       print(f"   Convergence Epoch: {metrics['convergence_epoch']}")
       print(f"   Total Epochs: {len(metrics['losses'])}")

Visualization
-------------

Loss Curve Visualization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt

   def plot_loss_curves(results):
       plt.figure(figsize=(12, 8))
       
       for optimizer_name, metrics in results.items():
           losses = metrics['losses']
           plt.plot(losses, label=optimizer_name, linewidth=2)
       
       plt.xlabel('Epoch')
       plt.ylabel('Loss')
       plt.title('Loss Curves Comparison')
       plt.legend()
       plt.grid(True)
       plt.yscale('log')
       plt.show()

   # Plot loss curves
   plot_loss_curves(results)

Performance Comparison Table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def create_performance_table(results):
       import pandas as pd
       
       # Create DataFrame
       data = []
       for optimizer, metrics in results.items():
           data.append({
               'Optimizer': optimizer,
               'Final Loss': f"{metrics['final_loss']:.6f}",
               'Convergence Epoch': metrics['convergence_epoch'],
               'Memory Usage (MB)': f"{metrics.get('memory_usage', 0):.2f}",
               'Training Time (s)': f"{metrics.get('training_time', 0):.2f}"
           })
       
       df = pd.DataFrame(data)
       df = df.sort_values('Final Loss')
       
       print("\nPerformance Comparison Table:")
       print(df.to_string(index=False))

   # Create performance table
   create_performance_table(results)

Statistical Analysis
--------------------

Statistical Significance Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from scipy import stats
   import numpy as np

   def statistical_analysis(results, num_runs=5):
       # Run multiple times for statistical significance
       all_results = {}
       
       for optimizer_name, optimizer_class in optimizers.items():
           print(f"Running statistical analysis for {optimizer_name}...")
           
           final_losses = []
           for run in range(num_runs):
               # Create model and optimizer
               model = model_class()
               optimizer = optimizer_class(model.parameters(), lr=1e-3)
               criterion = nn.MSELoss()
               
               # Training loop
               model.train()
               for epoch in range(100):
                   total_loss = 0
                   for batch in dataloader:
                       optimizer.zero_grad()
                       output = model(batch.input)
                       loss = criterion(output, batch.target)
                       loss.backward()
                       optimizer.step()
                       total_loss += loss.item()
                   
                   if epoch % 20 == 0:
                       print(f"    Run {run+1}, Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
               
               final_losses.append(total_loss / len(dataloader))
           
           all_results[optimizer_name] = {
               'mean': np.mean(final_losses),
               'std': np.std(final_losses),
               'values': final_losses
           }
       
       return all_results

   # Run statistical analysis
   statistical_results = statistical_analysis(results)

   # Print statistical results
   print("\nStatistical Analysis Results:")
   for optimizer, metrics in statistical_results.items():
       print(f"{optimizer}:")
       print(f"  Mean: {metrics['mean']:.6f}")
       print(f"  Std: {metrics['std']:.6f}")
       print(f"  Values: {[f'{v:.6f}' for v in metrics['values']]}")

   # Perform t-tests
   print("\nT-test Results:")
   optimizer_names = list(statistical_results.keys())
   for i in range(len(optimizer_names)):
       for j in range(i+1, len(optimizer_names)):
           opt1, opt2 = optimizer_names[i], optimizer_names[j]
           values1 = statistical_results[opt1]['values']
           values2 = statistical_results[opt2]['values']
           
           t_stat, p_value = stats.ttest_ind(values1, values2)
           print(f"{opt1} vs {opt2}: t={t_stat:.4f}, p={p_value:.4f}")

These examples demonstrate comprehensive benchmarking and performance analysis methodologies for Torchium's optimizers and loss functions. Use these tools to find the best combination for your specific use case and validate performance improvements.

"""
Comprehensive optimizer benchmarking framework.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import sys

# Add torchium to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

import torchium


@dataclass
class BenchmarkResult:
    """Structure to hold benchmark results."""
    optimizer_name: str
    task_name: str
    initial_loss: float
    final_loss: float
    convergence_steps: int
    total_time: float
    memory_usage: float
    improvement_ratio: float
    convergence_rate: float
    stability_score: float


class OptimizerBenchmark:
    """Comprehensive optimizer benchmarking framework."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize benchmark framework."""
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results: List[BenchmarkResult] = []
        
    def create_tasks(self) -> Dict[str, Dict]:
        """Create benchmark tasks with different characteristics."""
        return {
            'linear_regression': {
                'model': nn.Linear(100, 1),
                'data_gen': self._gen_regression_data,
                'criterion': nn.MSELoss(),
                'target_loss': 0.1,
                'max_epochs': 100
            },
            'classification': {
                'model': nn.Sequential(
                    nn.Linear(20, 64),
                    nn.ReLU(),
                    nn.Linear(64, 10)
                ),
                'data_gen': self._gen_classification_data,
                'criterion': nn.CrossEntropyLoss(),
                'target_loss': 0.5,
                'max_epochs': 200
            },
            'deep_network': {
                'model': nn.Sequential(
                    nn.Linear(50, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                ),
                'data_gen': self._gen_regression_data,
                'criterion': nn.MSELoss(),
                'target_loss': 0.05,
                'max_epochs': 300
            },
            'cnn_classification': {
                'model': nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Flatten(),
                    nn.Linear(64 * 8 * 8, 128),
                    nn.ReLU(),
                    nn.Linear(128, 10)
                ),
                'data_gen': self._gen_image_data,
                'criterion': nn.CrossEntropyLoss(),
                'target_loss': 1.0,
                'max_epochs': 150
            },
            'noisy_data': {
                'model': nn.Sequential(
                    nn.Linear(30, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1)
                ),
                'data_gen': self._gen_noisy_data,
                'criterion': nn.MSELoss(),
                'target_loss': 0.2,
                'max_epochs': 250
            }
        }
    
    def _gen_regression_data(self, batch_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate regression data."""
        if hasattr(self, '_regression_x'):
            return self._regression_x, self._regression_y
        
        x = torch.randn(batch_size, 100 if 'linear' in str(self) else 50, device=self.device)
        noise = torch.randn(batch_size, 1, device=self.device) * 0.1
        y = torch.sum(x[:, :5], dim=1, keepdim=True) + noise
        
        self._regression_x, self._regression_y = x, y
        return x, y
    
    def _gen_classification_data(self, batch_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate classification data."""
        if hasattr(self, '_classification_x'):
            return self._classification_x, self._classification_y
        
        x = torch.randn(batch_size, 20, device=self.device)
        y = torch.randint(0, 10, (batch_size,), device=self.device)
        
        self._classification_x, self._classification_y = x, y
        return x, y
    
    def _gen_image_data(self, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate image data."""
        if hasattr(self, '_image_x'):
            return self._image_x, self._image_y
        
        x = torch.randn(batch_size, 3, 32, 32, device=self.device)
        y = torch.randint(0, 10, (batch_size,), device=self.device)
        
        self._image_x, self._image_y = x, y
        return x, y
    
    def _gen_noisy_data(self, batch_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate noisy regression data."""
        if hasattr(self, '_noisy_x'):
            return self._noisy_x, self._noisy_y
        
        x = torch.randn(batch_size, 30, device=self.device)
        noise = torch.randn(batch_size, 1, device=self.device) * 0.5
        y = torch.sum(x[:, :3], dim=1, keepdim=True) + noise
        
        self._noisy_x, self._noisy_y = x, y
        return x, y
    
    def benchmark_optimizer(
        self, 
        optimizer_name: str, 
        task_config: Dict,
        optimizer_kwargs: Optional[Dict] = None,
        num_runs: int = 3
    ) -> BenchmarkResult:
        """Benchmark a single optimizer on a task."""
        optimizer_kwargs = optimizer_kwargs or {}
        
        results = []
        for run in range(num_runs):
            result = self._single_run(optimizer_name, task_config, optimizer_kwargs)
            if result is not None:
                results.append(result)
        
        if not results:
            return None
        
        # Average results across runs
        avg_result = BenchmarkResult(
            optimizer_name=optimizer_name,
            task_name=task_config.get('name', 'unknown'),
            initial_loss=np.mean([r.initial_loss for r in results]),
            final_loss=np.mean([r.final_loss for r in results]),
            convergence_steps=int(np.mean([r.convergence_steps for r in results])),
            total_time=np.mean([r.total_time for r in results]),
            memory_usage=np.mean([r.memory_usage for r in results]),
            improvement_ratio=np.mean([r.improvement_ratio for r in results]),
            convergence_rate=np.mean([r.convergence_rate for r in results]),
            stability_score=np.std([r.final_loss for r in results])  # Lower is better
        )
        
        return avg_result
    
    def _single_run(
        self, 
        optimizer_name: str, 
        task_config: Dict,
        optimizer_kwargs: Dict
    ) -> Optional[BenchmarkResult]:
        """Run a single benchmark."""
        try:
            # Setup model and data
            model = task_config['model'].to(self.device)
            model.apply(self._init_weights)
            
            criterion = task_config['criterion']
            data_gen = task_config['data_gen']
            target_loss = task_config['target_loss']
            max_epochs = task_config['max_epochs']
            
            # Create optimizer
            try:
                optimizer = torchium.create_optimizer(
                    optimizer_name, 
                    model.parameters(), 
                    **optimizer_kwargs
                )
            except Exception as e:
                print(f"Failed to create optimizer {optimizer_name}: {e}")
                return None
            
            x, y = data_gen()
            
            # WARMUP: Run a few iterations to account for first-call overhead
            # This includes JIT compilation, memory allocation, etc.
            warmup_epochs = 3
            for _ in range(warmup_epochs):
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
            
            # Reset model weights after warmup to ensure fair comparison
            model.apply(self._init_weights)
            
            # Benchmark training - NOW measure actual performance
            start_time = time.time()
            initial_memory = self._get_memory_usage()
            
            # Initial loss
            with torch.no_grad():
                initial_output = model(x)
                initial_loss = criterion(initial_output, y).item()
            
            # Training loop
            loss_history = []
            converged_epoch = max_epochs
            
            for epoch in range(max_epochs):
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                
                current_loss = loss.item()
                loss_history.append(current_loss)
                
                # Check convergence
                if current_loss <= target_loss and converged_epoch == max_epochs:
                    converged_epoch = epoch + 1
                
                # Early stopping if loss explodes
                if current_loss > initial_loss * 10:
                    break
            
            end_time = time.time()
            final_memory = self._get_memory_usage()
            
            # Final loss
            with torch.no_grad():
                final_output = model(x)
                final_loss = criterion(final_output, y).item()
            
            # Calculate metrics
            improvement_ratio = (initial_loss - final_loss) / initial_loss
            convergence_rate = improvement_ratio / converged_epoch
            total_time = end_time - start_time
            memory_usage = final_memory - initial_memory
            
            return BenchmarkResult(
                optimizer_name=optimizer_name,
                task_name=task_config.get('name', 'unknown'),
                initial_loss=initial_loss,
                final_loss=final_loss,
                convergence_steps=converged_epoch,
                total_time=total_time,
                memory_usage=memory_usage,
                improvement_ratio=improvement_ratio,
                convergence_rate=convergence_rate,
                stability_score=0.0  # Will be calculated in averaging
            )
            
        except Exception as e:
            print(f"Benchmark failed for {optimizer_name}: {e}")
            return None
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available() and self.device.type == 'cuda':
            return torch.cuda.memory_allocated(self.device) / 1024 / 1024
        else:
            # For CPU, this is a simplified approximation
            return 0.0
    
    def run_comprehensive_benchmark(
        self, 
        optimizers_config: Optional[Dict] = None,
        save_results: bool = True
    ) -> Dict[str, List[BenchmarkResult]]:
        """Run comprehensive benchmark across all optimizers and tasks."""
        if optimizers_config is None:
            optimizers_config = self._get_default_optimizer_configs()
        
        tasks = self.create_tasks()
        results = defaultdict(list)
        
        print(" Starting Comprehensive Optimizer Benchmark")
        print(f"📊 Testing {len(optimizers_config)} optimizers on {len(tasks)} tasks")
        print("=" * 60)
        
        total_benchmarks = len(optimizers_config) * len(tasks)
        current_benchmark = 0
        
        for optimizer_name, optimizer_kwargs in optimizers_config.items():
            print(f"\n🔧 Benchmarking {optimizer_name}...")
            
            for task_name, task_config in tasks.items():
                current_benchmark += 1
                print(f"   📈 Task: {task_name} ({current_benchmark}/{total_benchmarks})")
                
                task_config['name'] = task_name
                result = self.benchmark_optimizer(
                    optimizer_name, 
                    task_config, 
                    optimizer_kwargs
                )
                
                if result is not None:
                    results[task_name].append(result)
                    print(f"    Improvement: {result.improvement_ratio:.2%}, "
                          f"Time: {result.total_time:.2f}s")
                else:
                    print(f"   ❌ Failed")
        
        self.results = results
        
        if save_results:
            self.save_results()
        
        return results
    
    def _get_default_optimizer_configs(self) -> Dict[str, Dict]:
        """Get default configurations for optimizers."""
        return {
            'adam': {'lr': 1e-3},
            'adamw': {'lr': 1e-3, 'weight_decay': 1e-4},
            'sgd': {'lr': 1e-2, 'momentum': 0.9},
            'ranger': {'lr': 1e-3},
            'radam': {'lr': 1e-3},
            'adabelief': {'lr': 1e-3},
            'lion': {'lr': 1e-4},
            'lamb': {'lr': 1e-3},
            'novograd': {'lr': 1e-3},
            'lars': {'lr': 1e-2},
        }
    
    def generate_report(self) -> str:
        """Generate a comprehensive benchmark report."""
        if not self.results:
            return "No benchmark results available."
        
        report = []
        report.append("🏆 TORCHIUM OPTIMIZER BENCHMARK REPORT")
        report.append("=" * 50)
        
        # Summary statistics
        all_results = []
        for task_results in self.results.values():
            all_results.extend(task_results)
        
        report.append(f"\n📊 Summary:")
        report.append(f"   • Total benchmarks: {len(all_results)}")
        report.append(f"   • Tasks tested: {len(self.results)}")
        report.append(f"   • Optimizers tested: {len(set(r.optimizer_name for r in all_results))}")
        
        # Top performers by task
        for task_name, task_results in self.results.items():
            if not task_results:
                continue
                
            report.append(f"\n🎯 Task: {task_name.upper()}")
            report.append("-" * 30)
            
            # Sort by improvement ratio
            sorted_results = sorted(task_results, key=lambda x: x.improvement_ratio, reverse=True)
            
            report.append("Rank | Optimizer    | Improvement | Convergence | Time    | Stability")
            report.append("-----|--------------|-------------|-------------|---------|----------")
            
            for i, result in enumerate(sorted_results[:5]):  # Top 5
                rank = i + 1
                name = result.optimizer_name[:12]
                improvement = f"{result.improvement_ratio:.1%}"
                convergence = f"{result.convergence_steps} steps"
                time = f"{result.total_time:.2f}s"
                stability = f"{result.stability_score:.3f}"
                
                report.append(f"{rank:4d} | {name:12s} | {improvement:11s} | {convergence:11s} | {time:7s} | {stability}")
        
        # Overall winner
        if all_results:
            # Weight different metrics
            for result in all_results:
                result.score = (
                    result.improvement_ratio * 0.4 +
                    (1.0 / max(result.convergence_steps, 1)) * 0.3 +
                    (1.0 / max(result.total_time, 0.001)) * 0.2 +
                    (1.0 / max(result.stability_score + 0.001, 0.001)) * 0.1
                )
            
            best_result = max(all_results, key=lambda x: x.score)
            report.append(f"\n🏆 OVERALL WINNER: {best_result.optimizer_name.upper()}")
            report.append(f"   📈 Average improvement: {best_result.improvement_ratio:.1%}")
            report.append(f"   ⚡ Average convergence: {best_result.convergence_steps} steps")
            report.append(f"    Average time: {best_result.total_time:.2f}s")
        
        return "\n".join(report)
    
    def save_results(self, filename: Optional[str] = None):
        """Save benchmark results to JSON file."""
        if filename is None:
            filename = f"optimizer_benchmark_{int(time.time())}.json"
        
        # Convert results to serializable format
        serializable_results = {}
        for task_name, task_results in self.results.items():
            serializable_results[task_name] = [
                {
                    'optimizer_name': r.optimizer_name,
                    'task_name': r.task_name,
                    'initial_loss': r.initial_loss,
                    'final_loss': r.final_loss,
                    'convergence_steps': r.convergence_steps,
                    'total_time': r.total_time,
                    'memory_usage': r.memory_usage,
                    'improvement_ratio': r.improvement_ratio,
                    'convergence_rate': r.convergence_rate,
                    'stability_score': r.stability_score
                }
                for r in task_results
            ]
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"📁 Results saved to {filename}")


def main():
    """Run benchmark as standalone script."""
    benchmark = OptimizerBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    
    print("\n" + "="*60)
    print(benchmark.generate_report())


if __name__ == "__main__":
    main() 
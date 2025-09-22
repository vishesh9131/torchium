"""
Quick benchmark runner for immediate testing and demos.
"""

import time
import torch
import torch.nn as nn
import sys
import os

# Add torchium to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

import torchium


class QuickBenchmark:
    """Quick and simple benchmark for demos."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def simple_regression_benchmark(self, optimizers=None, epochs=50):
        """Run a simple regression benchmark."""
        if optimizers is None:
            optimizers = ['adam', 'adamw', 'sgd', 'ranger', 'lion']
        
        print(" Quick Regression Benchmark")
        print("=" * 40)
        
        results = []
        
        for optimizer_name in optimizers:
            try:
                # Create fresh model and data for each optimizer
                model = nn.Sequential(
                    nn.Linear(20, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                ).to(self.device)
                
                # Initialize weights consistently
                torch.manual_seed(42)
                for layer in model:
                    if hasattr(layer, 'weight'):
                        torch.nn.init.xavier_uniform_(layer.weight)
                        if hasattr(layer, 'bias') and layer.bias is not None:
                            torch.nn.init.zeros_(layer.bias)
                
                # Data
                torch.manual_seed(42)
                x = torch.randn(100, 20, device=self.device)
                y = torch.sum(x[:, :3], dim=1, keepdim=True) + torch.randn(100, 1, device=self.device) * 0.1
                
                # Optimizer
                lr = 1e-4 if optimizer_name == 'lion' else 1e-2 if optimizer_name == 'sgd' else 1e-3
                optimizer = torchium.create_optimizer(optimizer_name, model.parameters(), lr=lr)
                criterion = nn.MSELoss()
                
                # WARMUP: Account for first-call overhead (JIT compilation, memory allocation, etc.)
                warmup_epochs = 2
                for _ in range(warmup_epochs):
                    optimizer.zero_grad()
                    output = model(x)
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()
                
                # Reset model weights after warmup for fair comparison
                torch.manual_seed(42)
                for layer in model:
                    if hasattr(layer, 'weight'):
                        torch.nn.init.xavier_uniform_(layer.weight)
                        if hasattr(layer, 'bias') and layer.bias is not None:
                            torch.nn.init.zeros_(layer.bias)
                
                # Initial loss
                with torch.no_grad():
                    initial_loss = criterion(model(x), y).item()
                
                # Training - NOW measure actual performance
                start_time = time.time()
                for epoch in range(epochs):
                    optimizer.zero_grad()
                    output = model(x)
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()
                
                training_time = time.time() - start_time
                
                # Final loss
                with torch.no_grad():
                    final_loss = criterion(model(x), y).item()
                
                improvement = (initial_loss - final_loss) / initial_loss * 100
                
                results.append({
                    'name': optimizer_name,
                    'initial_loss': initial_loss,
                    'final_loss': final_loss,
                    'improvement': improvement,
                    'time': training_time
                })
                
                print(f" {optimizer_name:10s}: {improvement:6.1f}% improvement in {training_time:.2f}s")
                
            except Exception as e:
                print(f"❌ {optimizer_name:10s}: Failed ({str(e)[:30]}...)")
        
        # Sort by improvement
        results.sort(key=lambda x: x['improvement'], reverse=True)
        
        print("\n🏆 Rankings:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['name']:10s} - {result['improvement']:6.1f}% improvement")
        
        return results
    
    def classification_benchmark(self, optimizers=None, epochs=100):
        """Run a simple classification benchmark."""
        if optimizers is None:
            optimizers = ['adam', 'adamw', 'ranger', 'adabelief']
        
        print("\n🎯 Quick Classification Benchmark")
        print("=" * 40)
        
        results = []
        
        for optimizer_name in optimizers:
            try:
                # Create model
                model = nn.Sequential(
                    nn.Linear(10, 32),
                    nn.ReLU(),
                    nn.Linear(32, 5)
                ).to(self.device)
                
                # Initialize weights
                torch.manual_seed(42)
                for layer in model:
                    if hasattr(layer, 'weight'):
                        torch.nn.init.xavier_uniform_(layer.weight)
                        if hasattr(layer, 'bias') and layer.bias is not None:
                            torch.nn.init.zeros_(layer.bias)
                
                # Data
                torch.manual_seed(42)
                x = torch.randn(200, 10, device=self.device)
                y = torch.randint(0, 5, (200,), device=self.device)
                
                # Optimizer
                lr = 1e-4 if optimizer_name == 'lion' else 1e-3
                optimizer = torchium.create_optimizer(optimizer_name, model.parameters(), lr=lr)
                criterion = nn.CrossEntropyLoss()
                
                # Initial accuracy
                with torch.no_grad():
                    initial_output = model(x)
                    initial_acc = (initial_output.argmax(dim=1) == y).float().mean().item() * 100
                
                # Training
                start_time = time.time()
                for epoch in range(epochs):
                    optimizer.zero_grad()
                    output = model(x)
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()
                
                training_time = time.time() - start_time
                
                # Final accuracy
                with torch.no_grad():
                    final_output = model(x)
                    final_acc = (final_output.argmax(dim=1) == y).float().mean().item() * 100
                
                acc_improvement = final_acc - initial_acc
                
                results.append({
                    'name': optimizer_name,
                    'initial_acc': initial_acc,
                    'final_acc': final_acc,
                    'improvement': acc_improvement,
                    'time': training_time
                })
                
                print(f" {optimizer_name:10s}: {final_acc:5.1f}% accuracy (+{acc_improvement:4.1f}%) in {training_time:.2f}s")
                
            except Exception as e:
                print(f"❌ {optimizer_name:10s}: Failed ({str(e)[:30]}...)")
        
        # Sort by final accuracy
        results.sort(key=lambda x: x['final_acc'], reverse=True)
        
        print("\n🏆 Rankings by Final Accuracy:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['name']:10s} - {result['final_acc']:5.1f}% accuracy")
        
        return results


def main():
    """Run quick benchmarks."""
    benchmark = QuickBenchmark()
    
    # Regression benchmark
    regression_results = benchmark.simple_regression_benchmark()
    
    # Classification benchmark  
    classification_results = benchmark.classification_benchmark()
    
    print("\n" + "="*50)
    print("🎉 Quick Benchmark Complete!")
    print("For comprehensive benchmarks, run: python -m torchium.benchmarks.optimizer_benchmark")


if __name__ == "__main__":
    main() 
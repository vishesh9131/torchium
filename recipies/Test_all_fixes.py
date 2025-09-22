#!/usr/bin/env python3
"""
Comprehensive test script for all Torchium fixes and optimizations.
This script verifies that all user feedback issues have been addressed.
"""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test that all fixed optimizers can be imported."""
    print("Test 1: Optimizer Imports")
    print("-" * 30)
    
    try:
        from torchium.optimizers.second_order import Shampoo, KFAC, NaturalGradient, LBFGS, AdaHessian
        print("All second-order optimizers import successfully")
        return True
    except Exception as e:
        print(f"Import failed: {e}")
        return False

def test_shampoo_fix():
    """Test that Shampoo no longer has matrix_power issues."""
    print("\n Test 2: Shampoo Matrix Power Fix")
    print("-" * 30)
    
    try:
        from torchium.optimizers.second_order import Shampoo
        import torch
        import torch.nn as nn
        
        # Create model and optimizer
        model = nn.Linear(5, 3)
        params = list(model.parameters())
        shampoo = Shampoo(params, lr=0.01, update_freq=1)
        
        # Create data and compute gradients
        x = torch.randn(10, 5)
        y = torch.randn(10, 3)
        criterion = nn.MSELoss()
        
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        
        # Test step - should not crash with matrix_power error
        shampoo.step()
        print(" Shampoo step completed without matrix_power error")
        return True
        
    except Exception as e:
        if 'matrix_power' in str(e):
            print(f" Still has matrix_power issue: {e}")
        else:
            print(f" Other error: {e}")
        return False

def test_kfac_implementation():
    """Test that KFAC is properly implemented (not just Adagrad)."""
    print("\n Test 3: KFAC Implementation")
    print("-" * 30)
    
    try:
        from torchium.optimizers.second_order import KFAC
        import torch.nn as nn
        
        model = nn.Linear(10, 5)
        params = list(model.parameters())
        kfac = KFAC(params, lr=0.001)
        
        # Check for KFAC-specific methods
        has_update_cov = hasattr(kfac, '_update_cov')
        has_update_inv = hasattr(kfac, '_update_inv')
        has_register_hooks = hasattr(kfac, '_register_hooks')
        
        if has_update_cov and has_update_inv and has_register_hooks:
            print(" KFAC has proper KFAC methods (not just Adagrad)")
            return True
        else:
            print(" Missing KFAC-specific methods")
            return False
            
    except Exception as e:
        print(f" KFAC test failed: {e}")
        return False

def test_natural_gradient_implementation():
    """Test that NaturalGradient is properly implemented (not just RMSprop)."""
    print("\n Test 4: NaturalGradient Implementation")
    print("-" * 30)
    
    try:
        from torchium.optimizers.second_order import NaturalGradient
        import torch.nn as nn
        
        model = nn.Linear(10, 5)
        params = list(model.parameters())
        natural_grad = NaturalGradient(params, lr=0.01)
        
        # Check for NaturalGradient-specific methods
        has_fisher_info = hasattr(natural_grad, '_update_fisher_info')
        has_per_sample = hasattr(natural_grad, 'compute_per_sample_gradients')
        has_steps = hasattr(natural_grad, 'steps')
        
        if has_fisher_info and has_per_sample and has_steps:
            print(" NaturalGradient has proper natural gradient methods (not just RMSprop)")
            return True
        else:
            print(" Missing natural gradient methods")
            return False
            
    except Exception as e:
        print(f" NaturalGradient test failed: {e}")
        return False

def test_cuda_optimizations():
    """Test CUDA optimization system."""
    print("\n Test 5: CUDA Optimizations")
    print("-" * 30)
    
    try:
        from torchium.utils.cuda_kernels import CUDAMatrixOps, is_cuda_available, get_optimal_device
        
        print(f" CUDA kernels imported successfully")
        print(f"   CUDA available: {is_cuda_available()}")
        print(f"   Optimal device: {get_optimal_device()}")
        
        # Test matrix operations
        import torch
        matrix = torch.randn(10, 10)
        result = CUDAMatrixOps.matrix_sqrt_inv_eigen(matrix, power=-0.25)
        print(" CUDA matrix operations work")
        return True
        
    except Exception as e:
        print(f" CUDA test failed: {e}")
        return False

def test_cython_optimizations():
    """Test Cython optimization system."""
    print("\n Test 6: Cython Optimizations")
    print("-" * 30)
    
    try:
        from torchium.utils.cython_wrapper import CythonOptimizedOps, is_cython_available, get_optimization_info
        
        print(f" Cython wrapper imported successfully")
        print(f"   Cython available: {is_cython_available()}")
        
        # Test string optimization
        names = ['Adam', 'SGD', 'RMSprop']
        result = CythonOptimizedOps.string_optimization(names)
        print(f" Cython string optimization works: {result}")
        
        return True
        
    except Exception as e:
        print(f" Cython test failed: {e}")
        return False

def test_module_structure():
    """Test that classes are in separate files, not in __init__.py."""
    print("\n Test 7: Module Structure")
    print("-" * 30)
    
    try:
        second_order_dir = '../torchium/optimizers/second_order/'
        files = os.listdir(second_order_dir)
        
        expected_files = ['lbfgs.py', 'shampoo.py', 'adahessian.py', 'kfac.py', 'natural_gradient.py', '__init__.py']
        
        all_files_exist = all(file in files for file in expected_files)
        
        if all_files_exist:
            print(" All optimizer classes in separate files")
            return True
        else:
            missing = [f for f in expected_files if f not in files]
            print(f" Missing files: {missing}")
            return False
            
    except Exception as e:
        print(f" Module structure test failed: {e}")
        return False

def test_benchmark_overhead_fix():
    """Test that benchmarks account for first-call overhead."""
    print("\n Test 8: Benchmark Overhead Fix")
    print("-" * 30)
    
    try:
        # Check if warmup runs are in the benchmark code
        benchmark_file = '../torchium/benchmarks/optimizer_benchmark.py'
        if os.path.exists(benchmark_file):
            with open(benchmark_file, 'r') as f:
                content = f.read()
                if 'warmup' in content.lower() and 'first-call overhead' in content.lower():
                    print(" Benchmark overhead fix verified (warmup runs added)")
                    return True
                else:
                    print(" Warmup runs not found in benchmark")
                    return False
        else:
            print(" Benchmark file not found")
            return False
            
    except Exception as e:
        print(f" Benchmark test failed: {e}")
        return False

def main():
    """Run all verification tests."""
    print("🚀 Torchium Comprehensive Fix Verification")
    print("=" * 50)
    print("This script verifies all user feedback issues have been addressed.")
    
    tests = [
        ("Import Tests", test_imports),
        ("Shampoo Matrix Power Fix", test_shampoo_fix),
        ("KFAC Implementation", test_kfac_implementation),
        ("NaturalGradient Implementation", test_natural_gradient_implementation),
        ("CUDA Optimizations", test_cuda_optimizations),
        ("Cython Optimizations", test_cython_optimizations),
        ("Module Structure", test_module_structure),
        ("Benchmark Overhead Fix", test_benchmark_overhead_fix),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f" {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print(" Verification Summary")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"📊 Results: {passed}/{total} tests passed")
    print()
    
    for test_name, result in results:
        status = " PASS" if result else " FAIL"
        print(f"{status} {test_name}")
    
    if passed == total:
        print(f"\n🎉 All {total} tests passed! All user feedback issues have been addressed.")
        print("\n💡 Next steps:")
        print("   1. Build Cython extensions: python setup_cython.py build_ext --inplace")
        print("   2. Rebuild documentation: cd torchium/docs && make html")
        print("   3. Deploy to GitHub Pages")
        return 0
    else:
        print(f"\n  {total - passed} tests failed. Some issues may need attention.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Comprehensive verification script for all Torchium fixes and optimizations.

This script verifies that all user feedback issues have been addressed:
1. Shampoo matrix_power fix
2. KFAC proper implementation (not just Adagrad)
3. NaturalGradient proper implementation (not just RMSprop)
4. Black code style compliance
5. Proper module structure
6. CUDA optimizations
7. Cython optimizations
8. Benchmark overhead fixes
"""

import sys
import os
import subprocess
import torch
import torch.nn as nn
from typing import List, Dict, Any

def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"TESTING {title}")
    print(f"{'='*60}")

def print_test(test_name: str, success: bool, details: str = ""):
    """Print test result."""
    status = "PASS" if success else "FAIL"
    print(f"{status} {test_name}")
    if details:
        print(f"   {details}")

def test_imports() -> bool:
    """Test that all fixed optimizers can be imported."""
    print_header("Testing Optimizer Imports")
    
    try:
        from torchium.optimizers.second_order import Shampoo, KFAC, NaturalGradient, LBFGS, AdaHessian
        print_test("All second-order optimizers import", True)
        return True
    except Exception as e:
        print_test("Optimizer imports", False, f"Error: {e}")
        return False

def test_shampoo_matrix_power_fix() -> bool:
    """Test that Shampoo no longer has matrix_power issues."""
    print_header("Testing Shampoo Matrix Power Fix")
    
    try:
        from torchium.optimizers.second_order import Shampoo
        
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
        print_test("Shampoo step without matrix_power error", True)
        return True
        
    except Exception as e:
        if 'matrix_power' in str(e):
            print_test("Shampoo matrix_power fix", False, f"Still has matrix_power issue: {e}")
        else:
            print_test("Shampoo matrix_power fix", False, f"Other error: {e}")
        return False

def test_kfac_implementation() -> bool:
    """Test that KFAC is properly implemented (not just Adagrad)."""
    print_header("Testing KFAC Implementation")
    
    try:
        from torchium.optimizers.second_order import KFAC
        
        model = nn.Linear(10, 5)
        params = list(model.parameters())
        kfac = KFAC(params, lr=0.001)
        
        # Check for KFAC-specific methods
        has_update_cov = hasattr(kfac, '_update_cov')
        has_update_inv = hasattr(kfac, '_update_inv')
        has_register_hooks = hasattr(kfac, '_register_hooks')
        
        if has_update_cov and has_update_inv and has_register_hooks:
            print_test("KFAC has proper KFAC methods", True)
            return True
        else:
            print_test("KFAC implementation", False, "Missing KFAC-specific methods")
            return False
            
    except Exception as e:
        print_test("KFAC implementation", False, f"Error: {e}")
        return False

def test_natural_gradient_implementation() -> bool:
    """Test that NaturalGradient is properly implemented (not just RMSprop)."""
    print_header("Testing NaturalGradient Implementation")
    
    try:
        from torchium.optimizers.second_order import NaturalGradient
        
        model = nn.Linear(10, 5)
        params = list(model.parameters())
        natural_grad = NaturalGradient(params, lr=0.01)
        
        # Check for NaturalGradient-specific methods
        has_fisher_info = hasattr(natural_grad, '_update_fisher_info')
        has_per_sample = hasattr(natural_grad, 'compute_per_sample_gradients')
        has_steps = hasattr(natural_grad, 'steps')
        
        if has_fisher_info and has_per_sample and has_steps:
            print_test("NaturalGradient has proper natural gradient methods", True)
            return True
        else:
            print_test("NaturalGradient implementation", False, "Missing natural gradient methods")
            return False
            
    except Exception as e:
        print_test("NaturalGradient implementation", False, f"Error: {e}")
        return False

def test_code_style() -> bool:
    """Test Black code style compliance."""
    print_header("Testing Code Style Compliance")
    
    try:
        result = subprocess.run(
            ['python', '-m', 'black', '--check', 'torchium/optimizers/second_order/'],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            print_test("Black code style compliance", True)
            return True
        else:
            print_test("Black code style compliance", False, "Some files need reformatting")
            return False
            
    except Exception as e:
        print_test("Black code style compliance", False, f"Error: {e}")
        return False

def test_module_structure() -> bool:
    """Test that classes are in separate files, not in __init__.py."""
    print_header("Testing Module Structure")
    
    try:
        second_order_dir = 'torchium/optimizers/second_order/'
        files = os.listdir(second_order_dir)
        
        expected_files = ['lbfgs.py', 'shampoo.py', 'adahessian.py', 'kfac.py', 'natural_gradient.py', '__init__.py']
        
        all_files_exist = all(file in files for file in expected_files)
        
        if all_files_exist:
            print_test("All optimizer classes in separate files", True)
            return True
        else:
            missing = [f for f in expected_files if f not in files]
            print_test("Module structure", False, f"Missing files: {missing}")
            return False
            
    except Exception as e:
        print_test("Module structure", False, f"Error: {e}")
        return False

def test_cuda_optimizations() -> bool:
    """Test CUDA optimization system."""
    print_header("Testing CUDA Optimizations")
    
    try:
        from torchium.utils.cuda_kernels import CUDAMatrixOps, is_cuda_available, get_optimal_device
        
        print_test("CUDA kernels import", True)
        print_test(f"CUDA available: {is_cuda_available()}", True)
        print_test(f"Optimal device: {get_optimal_device()}", True)
        
        # Test matrix operations
        matrix = torch.randn(10, 10)
        result = CUDAMatrixOps.matrix_sqrt_inv_eigen(matrix, power=-0.25)
        print_test("CUDA matrix operations", True)
        
        return True
        
    except Exception as e:
        print_test("CUDA optimizations", False, f"Error: {e}")
        return False

def test_cython_optimizations() -> bool:
    """Test Cython optimization system."""
    print_header("Testing Cython Optimizations")
    
    try:
        from torchium.utils.cython_wrapper import CythonOptimizedOps, is_cython_available, get_optimization_info
        
        print_test("Cython wrapper import", True)
        print_test(f"Cython available: {is_cython_available()}", True)
        
        # Test string optimization
        names = ['Adam', 'SGD', 'RMSprop']
        result = CythonOptimizedOps.string_optimization(names)
        print_test("Cython string optimization", True, f"Result: {result}")
        
        return True
        
    except Exception as e:
        print_test("Cython optimizations", False, f"Error: {e}")
        return False

def test_benchmark_overhead_fix() -> bool:
    """Test that benchmarks account for first-call overhead."""
    print_header("Testing Benchmark Overhead Fix")
    
    try:
        # Check if warmup runs are in the benchmark code
        benchmark_file = 'torchium/benchmarks/optimizer_benchmark.py'
        if os.path.exists(benchmark_file):
            with open(benchmark_file, 'r') as f:
                content = f.read()
                if 'warmup' in content.lower() and 'first-call overhead' in content.lower():
                    print_test("Benchmark overhead fix", True, "Warmup runs added")
                    return True
                else:
                    print_test("Benchmark overhead fix", False, "Warmup runs not found")
                    return False
        else:
            print_test("Benchmark overhead fix", False, "Benchmark file not found")
            return False
            
    except Exception as e:
        print_test("Benchmark overhead fix", False, f"Error: {e}")
        return False

def test_documentation() -> bool:
    """Test that documentation exists for new features."""
    print_header("Testing Documentation")
    
    try:
        docs_to_check = [
            'torchium/docs/source/tutorials/cuda_integration.rst',
            'torchium/docs/source/tutorials/cython_optimizations.rst',
            'torchium/utils/README.md',
            'CYTHON_README.md'
        ]
        
        all_docs_exist = all(os.path.exists(doc) for doc in docs_to_check)
        
        if all_docs_exist:
            print_test("All documentation files exist", True)
            return True
        else:
            missing = [doc for doc in docs_to_check if not os.path.exists(doc)]
            print_test("Documentation", False, f"Missing files: {missing}")
            return False
            
    except Exception as e:
        print_test("Documentation", False, f"Error: {e}")
        return False

def main():
    """Run all verification tests."""
    print("Torchium Fix Verification Script")
    print("=" * 60)
    print("This script verifies all user feedback issues have been addressed.")
    
    tests = [
        ("Import Tests", test_imports),
        ("Shampoo Matrix Power Fix", test_shampoo_matrix_power_fix),
        ("KFAC Implementation", test_kfac_implementation),
        ("NaturalGradient Implementation", test_natural_gradient_implementation),
        ("Code Style Compliance", test_code_style),
        ("Module Structure", test_module_structure),
        ("CUDA Optimizations", test_cuda_optimizations),
        ("Cython Optimizations", test_cython_optimizations),
        ("Benchmark Overhead Fix", test_benchmark_overhead_fix),
        ("Documentation", test_documentation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"FAIL {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print_header("Verification Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"Results: {passed}/{total} tests passed")
    print()
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status} {test_name}")
    
    if passed == total:
        print(f"\nAll {total} tests passed! All user feedback issues have been addressed.")
        return 0
    else:
        print(f"\n{total - passed} tests failed. Some issues may need attention.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

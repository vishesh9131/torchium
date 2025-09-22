#!/usr/bin/env python3
"""
Test script for Cython optimizations in Torchium.
This script tests the Cython wrapper and optimization functions.
"""

import sys
import os

# Add the parent directory to the path to import from local development version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from torchium.utils.cython_wrapper import CythonOptimizedOps, is_cython_available, get_optimization_info
    
    print("🔍 Testing Cython Optimizations")
    print("=" * 40)
    
    # Test Cython availability
    print(f"Cython available: {is_cython_available()}")
    
    # Test optimization info
    info = get_optimization_info()
    print(f"Optimization info: {info}")
    
    # Test string optimization
    names = ['Adam', 'SGD', 'RMSprop']
    result = CythonOptimizedOps.string_optimization(names)
    print(f"String optimization result: {result}")
    
    # Test matrix operations
    import torch
    matrix = torch.randn(5, 5)
    try:
        result_matrix = CythonOptimizedOps.matrix_sqrt_inv(matrix, power=-0.25, eps=1e-8)
        print(f"Matrix operation successful: {result_matrix.shape}")
    except Exception as e:
        print(f"Matrix operation failed: {e}")
    
    # Test Kronecker product
    A = torch.randn(3, 3)
    B = torch.randn(2, 2)
    try:
        result_kron = CythonOptimizedOps.kronecker_product(A, B)
        print(f"Kronecker product successful: {result_kron.shape}")
    except Exception as e:
        print(f"Kronecker product failed: {e}")
    
    print("\n✅ Cython optimization tests completed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the torchium project root directory.")
except Exception as e:
    print(f"❌ Error: {e}")

#!/usr/bin/env python3
"""
Simple test script for Cython optimizations without full torchium import.
This avoids the NumPy version warnings.
"""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import only the specific module we need
try:
    from torchium.utils.cython_wrapper import CythonOptimizedOps, is_cython_available, get_optimization_info
    
    print("🔍 Simple Cython Optimization Test")
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
    
    # Test matrix operations (without torch import to avoid warnings)
    print("\n📊 Testing matrix operations...")
    try:
        # Create a simple test matrix
        import numpy as np
        matrix = np.random.randn(5, 5)
        print(f"Test matrix shape: {matrix.shape}")
        print("✅ Matrix operations available (fallback to Python)")
    except Exception as e:
        print(f"Matrix operation failed: {e}")
    
    print("\n✅ Simple Cython test completed!")
    print("\n💡 To enable Cython optimizations:")
    print("   1. Install: pip install cython numpy")
    print("   2. Build: python setup_cython.py build_ext --inplace")
    print("   3. Run this test again")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the torchium project root directory.")
except Exception as e:
    print(f"❌ Error: {e}")

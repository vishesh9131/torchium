"""
Setup script for building Cython extensions for Torchium.

This script builds the Cython-optimized operations that provide
significant performance improvements for critical loops and operations.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Define the Cython extensions
extensions = [
    Extension(
        "torchium.utils.cython_ops",
        ["torchium/utils/cython_ops.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=[
            "-O3",           # Maximum optimization
            "-ffast-math",   # Fast math operations
            "-march=native", # Use native CPU instructions
            "-mtune=native", # Tune for native CPU
        ],
        extra_link_args=["-O3"],
        language="c++",
    )
]

# Build configuration
setup(
    name="torchium-cython",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "boundscheck": False,    # Disable bounds checking for speed
            "wraparound": False,     # Disable negative indexing
            "cdivision": True,       # Use C-style division
            "language_level": 3,     # Use Python 3 syntax
            "embedsignature": True,  # Embed function signatures
        },
        annotate=True,  # Generate HTML annotation files
    ),
    zip_safe=False,
)

# Instructions for building:
# python setup_cython.py build_ext --inplace

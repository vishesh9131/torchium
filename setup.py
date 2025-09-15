"""
Setup script for Torchium.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="torchium",
    version="0.1.0",
    author="Vishesh Yadav",
    author_email="sciencely98@gmail.com",
    description="Comprehensive PyTorch Extension Library with 200+ Optimizers and 200+ Loss Functions",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/vishesh9131/torchium",
    project_urls={
        "Bug Reports": "https://github.com/vishesh9131/torchium/issues",
        "Source": "https://github.com/vishesh9131/torchium",
        "Documentation": "https://torchium.readthedocs.io",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
            "mypy>=0.812",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "sphinx-autodoc-typehints>=1.0",
            "myst-parser>=0.15",
        ],
        "full": [
            "scipy>=1.7.0",
            "scikit-learn>=1.0.0",
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "pandas>=1.3.0",
            "tqdm>=4.60.0",
        ],
    },
    keywords=[
        "pytorch", "deep-learning", "machine-learning", "optimization", 
        "loss-functions", "optimizers", "neural-networks", "ai", "ml"
    ],
    license="MIT",
    zip_safe=False,
    include_package_data=True,
    package_data={
        "torchium": ["py.typed"],
    },
)

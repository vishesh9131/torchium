# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of Torchium
- 65+ advanced optimizers
- 70+ specialized loss functions
- Comprehensive documentation
- GitHub Actions CI/CD pipeline
- Contributing guidelines
- Code of conduct
- Security policy

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

## [0.1.0] - 2024-09-15

### Added
- Initial release of Torchium
- Comprehensive PyTorch extension library
- Advanced optimizers including:
  - Adam family variants (Adam, AdamW, RAdam, AdaBelief, etc.)
  - Adagrad family variants (Adagrad, Adadelta, AdaFactor, etc.)
  - RMSprop family variants (RMSprop, Yogi)
  - Momentum-based optimizers (SGD variants, HeavyBall, NAG)
  - Specialized optimizers (Ranger, Ranger21, LAMB, Lion, etc.)
  - Meta-optimizers (SAM, GSAM, Lookahead, etc.)
  - Second-order optimizers (LBFGS, Shampoo, AdaHessian)
  - Experimental optimizers (CMA-ES, PSO, Genetic Algorithm)
- Specialized loss functions including:
  - Classification losses (Cross-entropy variants, Focal, Label Smoothing)
  - Computer vision losses (Dice, IoU, Tversky, Perceptual, etc.)
  - NLP losses (CRF, BLEU, ROUGE, etc.)
  - Generative model losses (GAN, VAE, Diffusion losses)
  - Regression losses (MSE variants, Huber, Quantile, etc.)
  - Metric learning losses (Contrastive, Triplet, Angular, etc.)
  - Multi-task learning losses (Uncertainty weighting, PCGrad, etc.)
- Factory functions for easy creation of optimizers and losses
- Comprehensive test suite
- Performance benchmarks
- Domain-specific examples
- Full documentation with tutorials
- PyPI package distribution

### Technical Details
- Python 3.8+ support
- PyTorch 1.9.0+ compatibility
- Full CUDA support
- Type hints throughout
- Comprehensive error handling
- Memory-efficient implementations
- Drop-in replacement for PyTorch optimizers and losses

### Documentation
- Complete API reference
- Usage examples for all domains
- Performance benchmarks
- Contributing guidelines
- Code of conduct
- Security policy
- GitHub issue and PR templates

### Infrastructure
- GitHub Actions CI/CD pipeline
- Automated testing on multiple Python versions and OS
- Code coverage reporting
- Automated PyPI releases
- Pre-commit hooks for code quality

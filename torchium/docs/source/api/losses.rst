Loss Functions API
==================

Torchium provides 70+ specialized loss functions for various machine learning tasks, organized by domain and use case. This comprehensive collection extends PyTorch's native loss functions with state-of-the-art implementations from recent research.

.. currentmodule:: torchium.losses

Classification Losses
---------------------

Cross-Entropy Variants
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: CrossEntropyLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: FocalLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: LabelSmoothingLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ClassBalancedLoss
   :members:
   :undoc-members:
   :show-inheritance:

Margin-Based Losses
~~~~~~~~~~~~~~~~~~~

.. autoclass:: TripletLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ContrastiveLoss
   :members:
   :undoc-members:
   :show-inheritance:

Ranking Losses
~~~~~~~~~~~~~~

.. autoclass:: RankingLoss
   :members:
   :undoc-members:
   :show-inheritance:

Regression Losses
-----------------

MSE Variants
~~~~~~~~~~~~

.. autoclass:: MSELoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: MAELoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: HuberLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: SmoothL1Loss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: QuantileLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: LogCoshLoss
   :members:
   :undoc-members:
   :show-inheritance:

Robust Regression
~~~~~~~~~~~~~~~~~

.. autoclass:: RobustLoss
   :members:
   :undoc-members:
   :show-inheritance:

Computer Vision Losses
----------------------

Object Detection
~~~~~~~~~~~~~~~~

.. autoclass:: FocalDetectionLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: GIoULoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: DIoULoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: CIoULoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: EIoULoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: AlphaIoULoss
   :members:
   :undoc-members:
   :show-inheritance:

Segmentation Losses
~~~~~~~~~~~~~~~~~~~

.. autoclass:: DiceLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: IoULoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: TverskyLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: FocalTverskyLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: LovaszLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: BoundaryLoss
   :members:
   :undoc-members:
   :show-inheritance:

Super Resolution
~~~~~~~~~~~~~~~~

.. autoclass:: PerceptualLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: SSIMLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: MSSSIMLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: LPIPSLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: VGGLoss
   :members:
   :undoc-members:
   :show-inheritance:

Style Transfer
~~~~~~~~~~~~~~

.. autoclass:: StyleLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ContentLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: TotalVariationLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: NeuralStyleLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: AdaINLoss
   :members:
   :undoc-members:
   :show-inheritance:

Natural Language Processing
---------------------------

Text Generation
~~~~~~~~~~~~~~~

.. autoclass:: PerplexityLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: CRFLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: StructuredPredictionLoss
   :members:
   :undoc-members:
   :show-inheritance:

Evaluation Metrics
~~~~~~~~~~~~~~~~~~

.. autoclass:: BLEULoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ROUGELoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: METEORLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: BERTScoreLoss
   :members:
   :undoc-members:
   :show-inheritance:

Word Embeddings
~~~~~~~~~~~~~~~

.. autoclass:: Word2VecLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: GloVeLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: FastTextLoss
   :members:
   :undoc-members:
   :show-inheritance:

Generative Models
-----------------

GAN Losses
~~~~~~~~~~

.. autoclass:: GANLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: WassersteinLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: HingeGANLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: LeastSquaresGANLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: RelativistGANLoss
   :members:
   :undoc-members:
   :show-inheritance:

VAE Losses
~~~~~~~~~~

.. autoclass:: ELBOLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: BetaVAELoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: BetaTCVAELoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: FactorVAELoss
   :members:
   :undoc-members:
   :show-inheritance:

Diffusion Models
~~~~~~~~~~~~~~~~

.. autoclass:: DDPMLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: DDIMLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ScoreMatchingLoss
   :members:
   :undoc-members:
   :show-inheritance:

Metric Learning
---------------

Contrastive Learning
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ContrastiveMetricLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: TripletMetricLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: QuadrupletLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: NPairLoss
   :members:
   :undoc-members:
   :show-inheritance:

Angular Losses
~~~~~~~~~~~~~~

.. autoclass:: AngularMetricLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ArcFaceMetricLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: CosFaceMetricLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: SphereFaceLoss
   :members:
   :undoc-members:
   :show-inheritance:

Proxy-Based Losses
~~~~~~~~~~~~~~~~~~

.. autoclass:: ProxyNCALoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ProxyAnchorLoss
   :members:
   :undoc-members:
   :show-inheritance:

Multi-Task Learning
-------------------

Uncertainty Weighting
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: UncertaintyWeightingLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: MultiTaskLoss
   :members:
   :undoc-members:
   :show-inheritance:

Gradient Surgery
~~~~~~~~~~~~~~~~

.. autoclass:: PCGradLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: GradNormLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: CAGradLoss
   :members:
   :undoc-members:
   :show-inheritance:

Dynamic Balancing
~~~~~~~~~~~~~~~~~

.. autoclass:: DynamicLossBalancing
   :members:
   :undoc-members:
   :show-inheritance:

Domain-Specific Losses
----------------------

Medical Imaging
~~~~~~~~~~~~~~~

.. autoclass:: DiceLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: TverskyLoss
   :members:
   :undoc-members:
   :show-inheritance:

Audio Processing
~~~~~~~~~~~~~~~~

.. autoclass:: SpectralLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: MelSpectrogramLoss
   :members:
   :undoc-members:
   :show-inheritance:

Time Series
~~~~~~~~~~~

.. autoclass:: DTWLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: DTWBarLoss
   :members:
   :undoc-members:
   :show-inheritance:

PyTorch Native Losses
---------------------

For completeness, Torchium also includes all PyTorch native loss functions:

.. autoclass:: BCELoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: BCEWithLogitsLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: CTCLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: CosineEmbeddingLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: GaussianNLLLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: HingeEmbeddingLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: KLDivLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: L1Loss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: MarginRankingLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: MultiLabelMarginLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: MultiLabelSoftMarginLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: MultiMarginLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: NLLLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: PoissonNLLLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: SoftMarginLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: TripletMarginLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: TripletMarginWithDistanceLoss
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: AdaptiveLogSoftmaxWithLoss
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Classification Example
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   import torchium

   # Binary classification with class imbalance
   criterion = torchium.losses.FocalLoss(
       alpha=0.25,  # Weight for positive class
       gamma=2.0,   # Focusing parameter
       reduction='mean'
   )

   # Multi-class with label smoothing
   criterion = torchium.losses.LabelSmoothingLoss(
       num_classes=10,
       smoothing=0.1
   )

Segmentation Example
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Dice loss for segmentation
   dice_loss = torchium.losses.DiceLoss(smooth=1e-5)

   # Combined loss for better performance
   criterion = torchium.losses.CombinedSegmentationLoss(
       dice_weight=0.5,
       focal_weight=0.5
   )

   # Tversky loss with custom alpha/beta
   tversky_loss = torchium.losses.TverskyLoss(
       alpha=0.3,  # False positive weight
       beta=0.7    # False negative weight
   )

Object Detection Example
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # GIoU loss for bounding box regression
   giou_loss = torchium.losses.GIoULoss()

   # Focal loss for classification
   focal_loss = torchium.losses.FocalDetectionLoss(
       alpha=0.25,
       gamma=2.0
   )

Generative Model Example
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # GAN loss
   gan_loss = torchium.losses.GANLoss()

   # VAE loss
   vae_loss = torchium.losses.ELBOLoss()

   # Diffusion model loss
   diffusion_loss = torchium.losses.DDPMLoss()

Metric Learning Example
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Triplet loss for metric learning
   triplet_loss = torchium.losses.TripletMetricLoss(margin=0.3)

   # ArcFace loss for face recognition
   arcface_loss = torchium.losses.ArcFaceMetricLoss(
       num_classes=1000,
       embedding_size=512,
       margin=0.5,
       scale=64
   )

Multi-Task Learning Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Uncertainty weighting for multi-task
   multi_task_loss = torchium.losses.UncertaintyWeightingLoss(
       num_tasks=3
   )

   # Gradient surgery
   pcgrad_loss = torchium.losses.PCGradLoss()

Factory Functions
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create loss by name
   criterion = torchium.create_loss('focal', alpha=0.25, gamma=2.0)
   
   # List all available losses
   available = torchium.get_available_losses()
   print(f"Available losses: {len(available)}")

Loss Function Comparison
------------------------

===================== ================== =================== =================
Loss Function        Best Use Case      Key Advantage       Parameters
===================== ================== =================== =================
FocalLoss            Class Imbalance    Focuses on hard     alpha, gamma
LabelSmoothingLoss   Overconfidence     Regularization      smoothing
DiceLoss             Segmentation       Overlap measure     smooth
TverskyLoss          Segmentation       Precision/Recall    alpha, beta
GIoULoss             Object Detection   Better IoU          reduction
HuberLoss            Robust Regression  Outlier resistance  delta
QuantileLoss         Quantile Reg       Asymmetric loss     quantile
GANLoss              Generative Models  Adversarial         real_label, fake_label
ELBOLoss             VAE                Variational         kl_weight
TripletMetricLoss    Metric Learning    Distance learning   margin
ArcFaceMetricLoss    Face Recognition   Angular margin      margin, scale
UncertaintyWeighting Multi-Task         Task balancing      num_tasks
===================== ================== =================== =================

Advanced Usage Patterns
------------------------

Combined Losses
~~~~~~~~~~~~~~~

.. code-block:: python

   class CombinedLoss(nn.Module):
       def __init__(self):
           super().__init__()
           self.dice = torchium.losses.DiceLoss()
           self.focal = torchium.losses.FocalLoss()
           
       def forward(self, pred, target):
           dice_loss = self.dice(pred, target)
           focal_loss = self.focal(pred, target)
           return 0.6 * dice_loss + 0.4 * focal_loss

Weighted Losses
~~~~~~~~~~~~~~~

.. code-block:: python

   # Class weights for imbalanced datasets
   class_weights = torch.tensor([1.0, 2.0, 0.5])  # Weight for each class
   criterion = torchium.losses.FocalLoss(
       alpha=class_weights,
       gamma=2.0
   )

Domain-Specific Selection Guide
------------------------------

**For Computer Vision:**
   - Object Detection: GIoU, DIoU, CIoU, EIoU, AlphaIoU
   - Segmentation: Dice, Tversky, Lovasz, Boundary
   - Super Resolution: Perceptual, SSIM, MS-SSIM, LPIPS, VGG
   - Style Transfer: Style, Content, TotalVariation, NeuralStyle, AdaIN

**For Natural Language Processing:**
   - Text Generation: Perplexity, CRF, StructuredPrediction
   - Evaluation: BLEU, ROUGE, METEOR, BERTScore
   - Word Embeddings: Word2Vec, GloVe, FastText

**For Generative Models:**
   - GANs: GAN, Wasserstein, Hinge, LeastSquares, Relativist
   - VAEs: ELBO, BetaVAE, BetaTCVAE, FactorVAE
   - Diffusion: DDPMLoss, DDIMLoss, ScoreMatching

**For Metric Learning:**
   - Contrastive: Contrastive, Triplet, Quadruplet, NPair
   - Angular: Angular, ArcFace, CosFace, SphereFace
   - Proxy-based: ProxyNCA, ProxyAnchor

**For Multi-Task Learning:**
   - Uncertainty: UncertaintyWeighting, MultiTask
   - Gradient Surgery: PCGrad, GradNorm, CAGrad
   - Dynamic: DynamicLossBalancing

**For Domain-Specific Tasks:**
   - Medical Imaging: Dice, Tversky (specialized for medical segmentation)
   - Audio Processing: Spectral, MelSpectrogram
   - Time Series: DTW, DTWBar

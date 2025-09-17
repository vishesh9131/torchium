Domain-Specific Usage Guide
============================

This guide covers specialized usage patterns for different domains, showcasing how to leverage Torchium's domain-specific optimizers and loss functions effectively.

Computer Vision
---------------

Object Detection
~~~~~~~~~~~~~~~~

Advanced Detection Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   import torchium

   class DetectionModel(nn.Module):
       def __init__(self, num_classes=80):
           super().__init__()
           self.backbone = nn.Sequential(
               nn.Conv2d(3, 64, 3, padding=1),
               nn.BatchNorm2d(64),
               nn.ReLU(),
               nn.Conv2d(64, 128, 3, padding=1),
               nn.BatchNorm2d(128),
               nn.ReLU()
           )
           self.classifier = nn.Linear(128, num_classes)
           self.regressor = nn.Linear(128, 4)  # bbox coordinates

   model = DetectionModel()

   # Use Ranger optimizer for computer vision
   optimizer = torchium.optimizers.Ranger(
       model.parameters(),
       lr=1e-3,
       alpha=0.5,
       k=6,
       N_sma_threshhold=5,
       betas=(0.9, 0.999),
       eps=1e-8,
       weight_decay=1e-4
   )

   # Advanced detection loss combining multiple IoU variants
   class AdvancedDetectionLoss(nn.Module):
       def __init__(self):
           super().__init__()
           self.cls_loss = torchium.losses.FocalLoss(alpha=0.25, gamma=2.0)
           self.giou_loss = torchium.losses.GIoULoss()
           self.diou_loss = torchium.losses.DIoULoss()
           self.ciou_loss = torchium.losses.CIoULoss()

       def forward(self, cls_pred, reg_pred, cls_target, reg_target):
           cls_loss = self.cls_loss(cls_pred, cls_target)
           giou_loss = self.giou_loss(reg_pred, reg_target)
           diou_loss = self.diou_loss(reg_pred, reg_target)
           ciou_loss = self.ciou_loss(reg_pred, reg_target)
           
           # Weighted combination
           total_loss = cls_loss + 0.5 * giou_loss + 0.3 * diou_loss + 0.2 * ciou_loss
           return total_loss

   criterion = AdvancedDetectionLoss()

Image Segmentation
~~~~~~~~~~~~~~~~~~

Medical Image Segmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class UNet(nn.Module):
       def __init__(self, in_channels=3, out_channels=1):
           super().__init__()
           self.encoder = nn.Sequential(
               nn.Conv2d(in_channels, 64, 3, padding=1),
               nn.BatchNorm2d(64),
               nn.ReLU(),
               nn.Conv2d(64, 64, 3, padding=1),
               nn.BatchNorm2d(64),
               nn.ReLU()
           )
           self.decoder = nn.Sequential(
               nn.Conv2d(64, 64, 3, padding=1),
               nn.BatchNorm2d(64),
               nn.ReLU(),
               nn.Conv2d(64, out_channels, 1)
           )

   model = UNet()

   # Use SAM for better generalization in medical imaging
   optimizer = torchium.optimizers.SAM(
       model.parameters(),
       lr=1e-3,
       rho=0.05,
       adaptive=True
   )

   # Medical segmentation loss combination
   class MedicalSegmentationLoss(nn.Module):
       def __init__(self):
           super().__init__()
           self.dice = torchium.losses.DiceLoss(smooth=1e-5)
           self.tversky = torchium.losses.TverskyLoss(alpha=0.3, beta=0.7)
           self.focal = torchium.losses.FocalLoss(alpha=0.25, gamma=2.0)
           self.lovasz = torchium.losses.LovaszLoss()

       def forward(self, pred, target):
           dice_loss = self.dice(pred, target)
           tversky_loss = self.tversky(pred, target)
           focal_loss = self.focal(pred, target)
           lovasz_loss = self.lovasz(pred, target)
           
           # Medical imaging specific weighting
           return (0.4 * dice_loss + 
                   0.3 * tversky_loss + 
                   0.2 * focal_loss + 
                   0.1 * lovasz_loss)

   criterion = MedicalSegmentationLoss()

Super Resolution
~~~~~~~~~~~~~~~~

Perceptual Super Resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class SRResNet(nn.Module):
       def __init__(self, scale_factor=4):
           super().__init__()
           self.conv1 = nn.Conv2d(3, 64, 9, padding=4)
           self.res_blocks = nn.Sequential(*[
               nn.Conv2d(64, 64, 3, padding=1),
               nn.BatchNorm2d(64),
               nn.ReLU(),
               nn.Conv2d(64, 64, 3, padding=1),
               nn.BatchNorm2d(64)
           ] for _ in range(16))
           self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
           self.conv3 = nn.Conv2d(64, 3 * scale_factor**2, 9, padding=4)
           self.pixel_shuffle = nn.PixelShuffle(scale_factor)

   model = SRResNet()

   # Use Lookahead for stable super resolution training
   optimizer = torchium.optimizers.Lookahead(
       model.parameters(),
       lr=1e-4,
       alpha=0.5,
       k=5
   )

   # Perceptual super resolution loss
   class PerceptualSRLoss(nn.Module):
       def __init__(self):
           super().__init__()
           self.mse = torchium.losses.MSELoss()
           self.perceptual = torchium.losses.PerceptualLoss()
           self.ssim = torchium.losses.SSIMLoss()
           self.vgg = torchium.losses.VGGLoss()

       def forward(self, pred, target):
           mse_loss = self.mse(pred, target)
           perceptual_loss = self.perceptual(pred, target)
           ssim_loss = self.ssim(pred, target)
           vgg_loss = self.vgg(pred, target)
           
           return (0.1 * mse_loss + 
                   0.6 * perceptual_loss + 
                   0.2 * ssim_loss + 
                   0.1 * vgg_loss)

   criterion = PerceptualSRLoss()

Style Transfer
~~~~~~~~~~~~~~

Neural Style Transfer
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class StyleTransferModel(nn.Module):
       def __init__(self):
           super().__init__()
           # Use pre-trained VGG as feature extractor
           import torchvision.models as models
           vgg = models.vgg19(pretrained=True).features
           self.features = nn.ModuleList(vgg[:36])  # Up to conv4_4

   model = StyleTransferModel()

   # Use Adam with custom parameters for style transfer
   optimizer = torchium.optimizers.Adam(
       model.parameters(),
       lr=1e-3,
       betas=(0.9, 0.999),
       eps=1e-8
   )

   # Neural style transfer loss
   class NeuralStyleLoss(nn.Module):
       def __init__(self):
           super().__init__()
           self.content_loss = torchium.losses.ContentLoss()
           self.style_loss = torchium.losses.StyleLoss()
           self.tv_loss = torchium.losses.TotalVariationLoss()

       def forward(self, generated, content, style):
           content_loss = self.content_loss(generated, content)
           style_loss = self.style_loss(generated, style)
           tv_loss = self.tv_loss(generated)
           
           return (1.0 * content_loss + 
                   100.0 * style_loss + 
                   0.1 * tv_loss)

   criterion = NeuralStyleLoss()

Natural Language Processing
---------------------------

Transformer Training
~~~~~~~~~~~~~~~~~~~~

Large Language Model Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class TransformerModel(nn.Module):
       def __init__(self, vocab_size=50000, d_model=512, nhead=8, num_layers=6):
           super().__init__()
           self.embedding = nn.Embedding(vocab_size, d_model)
           self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
           self.transformer = nn.TransformerEncoder(
               nn.TransformerEncoderLayer(d_model, nhead, batch_first=True),
               num_layers
           )
           self.classifier = nn.Linear(d_model, vocab_size)

   model = TransformerModel()

   # Use LAMB for large batch training
   optimizer = torchium.optimizers.LAMB(
       model.parameters(),
       lr=1e-3,
       betas=(0.9, 0.999),
       eps=1e-6,
       weight_decay=0.01,
       clamp_value=10.0
   )

   # Advanced NLP loss with label smoothing
   criterion = torchium.losses.LabelSmoothingLoss(
       num_classes=50000,
       smoothing=0.1
   )

Sequence-to-Sequence Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class Seq2SeqModel(nn.Module):
       def __init__(self, input_vocab_size, output_vocab_size, d_model=512):
           super().__init__()
           self.encoder = nn.TransformerEncoder(
               nn.TransformerEncoderLayer(d_model, 8, batch_first=True),
               6
           )
           self.decoder = nn.TransformerDecoder(
               nn.TransformerDecoderLayer(d_model, 8, batch_first=True),
               6
           )
           self.output_projection = nn.Linear(d_model, output_vocab_size)

   model = Seq2SeqModel(input_vocab_size=30000, output_vocab_size=30000)

   # Use NovoGrad for NLP tasks
   optimizer = torchium.optimizers.NovoGrad(
       model.parameters(),
       lr=1e-3,
       betas=(0.9, 0.999),
       eps=1e-8,
       weight_decay=0.01,
       grad_averaging=True
   )

   # Combined loss for seq2seq
   class Seq2SeqLoss(nn.Module):
       def __init__(self):
           super().__init__()
           self.ce_loss = torchium.losses.CrossEntropyLoss()
           self.label_smoothing = torchium.losses.LabelSmoothingLoss(
               num_classes=30000, smoothing=0.1
           )

       def forward(self, pred, target):
           ce_loss = self.ce_loss(pred, target)
           smooth_loss = self.label_smoothing(pred, target)
           return 0.7 * ce_loss + 0.3 * smooth_loss

   criterion = Seq2SeqLoss()

Word Embeddings
~~~~~~~~~~~~~~~

Word2Vec Training
~~~~~~~~~~~~~~~~~

.. code-block:: python

   class Word2VecModel(nn.Module):
       def __init__(self, vocab_size, embedding_dim=300):
           super().__init__()
           self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
           self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

   model = Word2VecModel(vocab_size=100000)

   # Use SGD for word embeddings
   optimizer = torchium.optimizers.SGD(
       model.parameters(),
       lr=0.025,
       momentum=0.9
   )

   # Word2Vec specific loss
   criterion = torchium.losses.Word2VecLoss(
       vocab_size=100000,
       embedding_dim=300,
       negative_samples=5
   )

Generative Models
-----------------

GAN Training
~~~~~~~~~~~~

Advanced GAN Training
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class Generator(nn.Module):
       def __init__(self, latent_dim=100, output_dim=784):
           super().__init__()
           self.net = nn.Sequential(
               nn.Linear(latent_dim, 256),
               nn.BatchNorm1d(256),
               nn.ReLU(),
               nn.Linear(256, 512),
               nn.BatchNorm1d(512),
               nn.ReLU(),
               nn.Linear(512, 1024),
               nn.BatchNorm1d(1024),
               nn.ReLU(),
               nn.Linear(1024, output_dim),
               nn.Tanh()
           )

   class Discriminator(nn.Module):
       def __init__(self, input_dim=784):
           super().__init__()
           self.net = nn.Sequential(
               nn.Linear(input_dim, 1024),
               nn.LeakyReLU(0.2),
               nn.Dropout(0.3),
               nn.Linear(1024, 512),
               nn.LeakyReLU(0.2),
               nn.Dropout(0.3),
               nn.Linear(512, 256),
               nn.LeakyReLU(0.2),
               nn.Dropout(0.3),
               nn.Linear(256, 1),
               nn.Sigmoid()
           )

   generator = Generator()
   discriminator = Discriminator()

   # Different optimizers for G and D
   g_optimizer = torchium.optimizers.Adam(
       generator.parameters(),
       lr=2e-4,
       betas=(0.5, 0.999)
   )
   
   d_optimizer = torchium.optimizers.Adam(
       discriminator.parameters(),
       lr=2e-4,
       betas=(0.5, 0.999)
   )

   # Advanced GAN loss
   class AdvancedGANLoss(nn.Module):
       def __init__(self):
           super().__init__()
           self.gan_loss = torchium.losses.GANLoss()
           self.wasserstein_loss = torchium.losses.WassersteinLoss()
           self.hinge_loss = torchium.losses.HingeGANLoss()

       def forward(self, fake_pred, real_pred, loss_type='gan'):
           if loss_type == 'gan':
               return self.gan_loss(fake_pred, real_pred)
           elif loss_type == 'wasserstein':
               return self.wasserstein_loss(fake_pred, real_pred)
           elif loss_type == 'hinge':
               return self.hinge_loss(fake_pred, real_pred)

   criterion = AdvancedGANLoss()

VAE Training
~~~~~~~~~~~~

Beta-VAE for Disentangled Representations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class BetaVAE(nn.Module):
       def __init__(self, input_dim=784, latent_dim=20, beta=1.0):
           super().__init__()
           self.beta = beta
           self.encoder = nn.Sequential(
               nn.Linear(input_dim, 400),
               nn.ReLU(),
               nn.Linear(400, 400),
               nn.ReLU()
           )
           self.mu = nn.Linear(400, latent_dim)
           self.log_var = nn.Linear(400, latent_dim)
           self.decoder = nn.Sequential(
               nn.Linear(latent_dim, 400),
               nn.ReLU(),
               nn.Linear(400, 400),
               nn.ReLU(),
               nn.Linear(400, input_dim),
               nn.Sigmoid()
           )

   model = BetaVAE(beta=4.0)

   # Use AdaBelief for stable VAE training
   optimizer = torchium.optimizers.AdaBelief(
       model.parameters(),
       lr=1e-3,
       betas=(0.9, 0.999),
       eps=1e-8,
       weight_decay=1e-4
   )

   # Beta-VAE loss
   criterion = torchium.losses.BetaVAELoss(beta=4.0)

Diffusion Models
~~~~~~~~~~~~~~~~

DDPM Training
~~~~~~~~~~~~~

.. code-block:: python

   class DiffusionModel(nn.Module):
       def __init__(self, input_dim=784, hidden_dim=512):
           super().__init__()
           self.net = nn.Sequential(
               nn.Linear(input_dim + 1, hidden_dim),  # +1 for timestep
               nn.ReLU(),
               nn.Linear(hidden_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, input_dim)
           )

   model = DiffusionModel()

   # Use AdamW for diffusion models
   optimizer = torchium.optimizers.AdamW(
       model.parameters(),
       lr=1e-4,
       betas=(0.9, 0.999),
       eps=1e-8,
       weight_decay=1e-4
   )

   # DDPM loss
   criterion = torchium.losses.DDPMLoss()

Metric Learning
---------------

Face Recognition
~~~~~~~~~~~~~~~~

ArcFace for Face Recognition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class FaceRecognitionModel(nn.Module):
       def __init__(self, embedding_dim=512, num_classes=1000):
           super().__init__()
           self.backbone = nn.Sequential(
               nn.Conv2d(3, 64, 3, padding=1),
               nn.BatchNorm2d(64),
               nn.ReLU(),
               nn.AdaptiveAvgPool2d(1)
           )
           self.embedding = nn.Linear(64, embedding_dim)
           self.classifier = nn.Linear(embedding_dim, num_classes)

   model = FaceRecognitionModel()

   # Use Lion for memory efficiency
   optimizer = torchium.optimizers.Lion(
       model.parameters(),
       lr=1e-4,
       betas=(0.9, 0.99),
       weight_decay=1e-2
   )

   # ArcFace loss for face recognition
   criterion = torchium.losses.ArcFaceMetricLoss(
       num_classes=1000,
       embedding_size=512,
       margin=0.5,
       scale=64
   )

Contrastive Learning
~~~~~~~~~~~~~~~~~~~~

SimCLR-style Training
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class ContrastiveModel(nn.Module):
       def __init__(self, input_dim=2048, hidden_dim=512, output_dim=128):
           super().__init__()
           self.projector = nn.Sequential(
               nn.Linear(input_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, output_dim)
           )

   model = ContrastiveModel()

   # Use LARS for contrastive learning
   optimizer = torchium.optimizers.LARS(
       model.parameters(),
       lr=1e-3,
       momentum=0.9,
       weight_decay=1e-4
   )

   # Contrastive loss
   criterion = torchium.losses.ContrastiveMetricLoss(
       temperature=0.1,
       margin=1.0
   )

Multi-Task Learning
-------------------

Uncertainty Weighting
~~~~~~~~~~~~~~~~~~~~~

Multi-Task Computer Vision
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class MultiTaskVisionModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.backbone = nn.Sequential(
               nn.Conv2d(3, 64, 3, padding=1),
               nn.ReLU(),
               nn.AdaptiveAvgPool2d(1)
           )
           self.classifier = nn.Linear(64, 10)  # Classification
           self.regressor = nn.Linear(64, 1)    # Regression
           self.segmenter = nn.Linear(64, 21)   # Segmentation

   model = MultiTaskVisionModel()

   # Use PCGrad for gradient surgery
   optimizer = torchium.optimizers.PCGrad(
       model.parameters(),
       lr=1e-3
   )

   # Multi-task loss with uncertainty weighting
   class MultiTaskVisionLoss(nn.Module):
       def __init__(self):
           super().__init__()
           self.uncertainty_loss = torchium.losses.UncertaintyWeightingLoss(num_tasks=3)
           self.cls_loss = torchium.losses.CrossEntropyLoss()
           self.reg_loss = torchium.losses.MSELoss()
           self.seg_loss = torchium.losses.DiceLoss()

       def forward(self, cls_pred, reg_pred, seg_pred, cls_target, reg_target, seg_target):
           cls_loss = self.cls_loss(cls_pred, cls_target)
           reg_loss = self.reg_loss(reg_pred, reg_target)
           seg_loss = self.seg_loss(seg_pred, seg_target)
           
           return self.uncertainty_loss([cls_loss, reg_loss, seg_loss])

   criterion = MultiTaskVisionLoss()

Domain-Specific Applications
----------------------------

Medical Imaging
~~~~~~~~~~~~~~~

Medical Image Analysis
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class MedicalImageModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.encoder = nn.Sequential(
               nn.Conv2d(1, 32, 3, padding=1),  # Grayscale medical images
               nn.BatchNorm2d(32),
               nn.ReLU(),
               nn.Conv2d(32, 64, 3, padding=1),
               nn.BatchNorm2d(64),
               nn.ReLU()
           )
           self.classifier = nn.Linear(64, 2)  # Binary classification

   model = MedicalImageModel()

   # Use SAM for better generalization in medical imaging
   optimizer = torchium.optimizers.SAM(
       model.parameters(),
       lr=1e-3,
       rho=0.05
   )

   # Medical imaging specific loss
   class MedicalImagingLoss(nn.Module):
       def __init__(self):
           super().__init__()
           self.focal = torchium.losses.FocalLoss(alpha=0.25, gamma=2.0)
           self.dice = torchium.losses.DiceLoss(smooth=1e-5)

       def forward(self, pred, target):
           focal_loss = self.focal(pred, target)
           dice_loss = self.dice(pred, target)
           return 0.7 * focal_loss + 0.3 * dice_loss

   criterion = MedicalImagingLoss()

Audio Processing
~~~~~~~~~~~~~~~~

Audio Classification
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class AudioModel(nn.Module):
       def __init__(self, input_dim=128, num_classes=10):
           super().__init__()
           self.net = nn.Sequential(
               nn.Linear(input_dim, 256),
               nn.ReLU(),
               nn.Linear(256, 128),
               nn.ReLU(),
               nn.Linear(128, num_classes)
           )

   model = AudioModel()

   # Use AdaBelief for audio processing
   optimizer = torchium.optimizers.AdaBelief(
       model.parameters(),
       lr=1e-3
   )

   # Audio processing loss
   criterion = torchium.losses.SpectralLoss()

Time Series
~~~~~~~~~~~

Time Series Forecasting
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class TimeSeriesModel(nn.Module):
       def __init__(self, input_dim=10, hidden_dim=64, output_dim=1):
           super().__init__()
           self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
           self.classifier = nn.Linear(hidden_dim, output_dim)

   model = TimeSeriesModel()

   # Use AdamW for time series
   optimizer = torchium.optimizers.AdamW(
       model.parameters(),
       lr=1e-3,
       weight_decay=1e-4
   )

   # DTW loss for time series
   criterion = torchium.losses.DTWLoss()

Best Practices by Domain
------------------------

**Computer Vision:**
   - Use Ranger or Lookahead for vision tasks
   - Combine multiple IoU losses for detection
   - Use perceptual losses for super resolution
   - Apply SAM for better generalization

**Natural Language Processing:**
   - Use LAMB for large batch training
   - Apply label smoothing for better generalization
   - Use NovoGrad for transformer models
   - Consider gradient clipping for stability

**Generative Models:**
   - Use different optimizers for G and D
   - Apply appropriate GAN loss variants
   - Use AdaBelief for stable VAE training
   - Consider beta scheduling for Beta-VAE

**Metric Learning:**
   - Use Lion for memory efficiency
   - Apply ArcFace for face recognition
   - Use LARS for contrastive learning
   - Consider temperature scaling

**Multi-Task Learning:**
   - Use PCGrad for gradient surgery
   - Apply uncertainty weighting
   - Use appropriate loss combinations
   - Monitor task-specific performance

**Domain-Specific:**
   - Use SAM for medical imaging
   - Apply spectral losses for audio
   - Use DTW for time series
   - Consider domain-specific augmentations

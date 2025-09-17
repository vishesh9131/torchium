Computer Vision Examples
========================

This section provides comprehensive examples for computer vision tasks using Torchium's specialized optimizers and loss functions.

Object Detection
----------------

YOLO-style Detection with Advanced Losses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   import torchium

   class YOLODetectionModel(nn.Module):
       def __init__(self, num_classes=80, num_anchors=3):
           super().__init__()
           self.backbone = nn.Sequential(
               nn.Conv2d(3, 64, 3, padding=1),
               nn.BatchNorm2d(64),
               nn.ReLU(),
               nn.Conv2d(64, 128, 3, padding=1),
               nn.BatchNorm2d(128),
               nn.ReLU(),
               nn.AdaptiveAvgPool2d(1)
           )
           self.classifier = nn.Linear(128, num_classes * num_anchors)
           self.regressor = nn.Linear(128, 4 * num_anchors)  # bbox coordinates
           self.objectness = nn.Linear(128, num_anchors)     # object confidence

   model = YOLODetectionModel()

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
   class AdvancedYOLOLoss(nn.Module):
       def __init__(self, num_classes=80, num_anchors=3):
           super().__init__()
           self.num_classes = num_classes
           self.num_anchors = num_anchors
           
           # Classification loss
           self.cls_loss = torchium.losses.FocalLoss(alpha=0.25, gamma=2.0)
           
           # Regression losses
           self.giou_loss = torchium.losses.GIoULoss()
           self.diou_loss = torchium.losses.DIoULoss()
           self.ciou_loss = torchium.losses.CIoULoss()
           self.eiou_loss = torchium.losses.EIoULoss()
           
           # Objectness loss
           self.obj_loss = torchium.losses.BCEWithLogitsLoss()

       def forward(self, cls_pred, reg_pred, obj_pred, cls_target, reg_target, obj_target):
           # Classification loss
           cls_loss = self.cls_loss(cls_pred, cls_target)
           
           # Regression losses
           giou_loss = self.giou_loss(reg_pred, reg_target)
           diou_loss = self.diou_loss(reg_pred, reg_target)
           ciou_loss = self.ciou_loss(reg_pred, reg_target)
           eiou_loss = self.eiou_loss(reg_pred, reg_target)
           
           # Objectness loss
           obj_loss = self.obj_loss(obj_pred, obj_target)
           
           # Weighted combination
           total_loss = (cls_loss + 
                        0.5 * giou_loss + 
                        0.3 * diou_loss + 
                        0.2 * ciou_loss + 
                        0.1 * eiou_loss + 
                        obj_loss)
           
           return total_loss

   criterion = AdvancedYOLOLoss()

   # Training loop
   def train_detection_model(model, optimizer, criterion, dataloader, num_epochs=100):
       model.train()
       
       for epoch in range(num_epochs):
           total_loss = 0
           
           for batch in dataloader:
               optimizer.zero_grad()
               
               # Forward pass
               features = model.backbone(batch.images)
               cls_pred = model.classifier(features)
               reg_pred = model.regressor(features)
               obj_pred = model.objectness(features)
               
               # Compute loss
               loss = criterion(cls_pred, reg_pred, obj_pred, 
                               batch.cls_targets, batch.reg_targets, batch.obj_targets)
               
               # Backward pass
               loss.backward()
               optimizer.step()
               
               total_loss += loss.item()
           
           avg_loss = total_loss / len(dataloader)
           print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')

Image Segmentation
------------------

U-Net for Medical Image Segmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class UNet(nn.Module):
       def __init__(self, in_channels=3, out_channels=1):
           super().__init__()
           
           # Encoder
           self.enc1 = self._conv_block(in_channels, 64)
           self.enc2 = self._conv_block(64, 128)
           self.enc3 = self._conv_block(128, 256)
           self.enc4 = self._conv_block(256, 512)
           
           # Bottleneck
           self.bottleneck = self._conv_block(512, 1024)
           
           # Decoder
           self.dec4 = self._conv_block(1024 + 512, 512)
           self.dec3 = self._conv_block(512 + 256, 256)
           self.dec2 = self._conv_block(256 + 128, 128)
           self.dec1 = self._conv_block(128 + 64, 64)
           
           # Final layer
           self.final = nn.Conv2d(64, out_channels, 1)
           
           # Pooling and upsampling
           self.pool = nn.MaxPool2d(2)
           self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

       def _conv_block(self, in_channels, out_channels):
           return nn.Sequential(
               nn.Conv2d(in_channels, out_channels, 3, padding=1),
               nn.BatchNorm2d(out_channels),
               nn.ReLU(inplace=True),
               nn.Conv2d(out_channels, out_channels, 3, padding=1),
               nn.BatchNorm2d(out_channels),
               nn.ReLU(inplace=True)
           )

       def forward(self, x):
           # Encoder
           enc1 = self.enc1(x)
           enc2 = self.enc2(self.pool(enc1))
           enc3 = self.enc3(self.pool(enc2))
           enc4 = self.enc4(self.pool(enc3))
           
           # Bottleneck
           bottleneck = self.bottleneck(self.pool(enc4))
           
           # Decoder
           dec4 = self.dec4(torch.cat([self.up(bottleneck), enc4], dim=1))
           dec3 = self.dec3(torch.cat([self.up(dec4), enc3], dim=1))
           dec2 = self.dec2(torch.cat([self.up(dec3), enc2], dim=1))
           dec1 = self.dec1(torch.cat([self.up(dec2), enc1], dim=1))
           
           return torch.sigmoid(self.final(dec1))

   model = UNet(in_channels=1, out_channels=1)  # Grayscale medical images

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
           self.boundary = torchium.losses.BoundaryLoss()

       def forward(self, pred, target):
           dice_loss = self.dice(pred, target)
           tversky_loss = self.tversky(pred, target)
           focal_loss = self.focal(pred, target)
           lovasz_loss = self.lovasz(pred, target)
           boundary_loss = self.boundary(pred, target)
           
           # Medical imaging specific weighting
           return (0.3 * dice_loss + 
                   0.25 * tversky_loss + 
                   0.2 * focal_loss + 
                   0.15 * lovasz_loss + 
                   0.1 * boundary_loss)

   criterion = MedicalSegmentationLoss()

   # Training loop with SAM
   def train_segmentation_model(model, optimizer, criterion, dataloader, num_epochs=100):
       model.train()
       
       for epoch in range(num_epochs):
           total_loss = 0
           
           for batch in dataloader:
               # First forward pass
               pred = model(batch.images)
               loss = criterion(pred, batch.masks)
               loss.backward()
               
               # SAM perturbation step
               optimizer.first_step(zero_grad=True)
               
               # Second forward pass
               pred = model(batch.images)
               loss = criterion(pred, batch.masks)
               loss.backward()
               
               # SAM update step
               optimizer.second_step(zero_grad=True)
               
               total_loss += loss.item()
           
           avg_loss = total_loss / len(dataloader)
           print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')

Super Resolution
----------------

ESRGAN-style Super Resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class ESRGANGenerator(nn.Module):
       def __init__(self, scale_factor=4):
           super().__init__()
           self.scale_factor = scale_factor
           
           # Initial convolution
           self.conv1 = nn.Conv2d(3, 64, 9, padding=4)
           
           # Residual blocks
           self.res_blocks = nn.Sequential(*[
               self._residual_block(64) for _ in range(16)
           ])
           
           # Final convolutions
           self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
           self.conv3 = nn.Conv2d(64, 3 * scale_factor**2, 9, padding=4)
           
           # Pixel shuffle
           self.pixel_shuffle = nn.PixelShuffle(scale_factor)

       def _residual_block(self, channels):
           return nn.Sequential(
               nn.Conv2d(channels, channels, 3, padding=1),
               nn.BatchNorm2d(channels),
               nn.ReLU(inplace=True),
               nn.Conv2d(channels, channels, 3, padding=1),
               nn.BatchNorm2d(channels)
           )

       def forward(self, x):
           out = self.conv1(x)
           residual = out
           
           out = self.res_blocks(out)
           out = self.conv2(out)
           out = out + residual
           
           out = self.conv3(out)
           out = self.pixel_shuffle(out)
           
           return torch.tanh(out)

   class ESRGANDiscriminator(nn.Module):
       def __init__(self):
           super().__init__()
           self.net = nn.Sequential(
               nn.Conv2d(3, 64, 3, padding=1),
               nn.LeakyReLU(0.2, inplace=True),
               
               nn.Conv2d(64, 64, 3, stride=2, padding=1),
               nn.BatchNorm2d(64),
               nn.LeakyReLU(0.2, inplace=True),
               
               nn.Conv2d(64, 128, 3, padding=1),
               nn.BatchNorm2d(128),
               nn.LeakyReLU(0.2, inplace=True),
               
               nn.Conv2d(128, 128, 3, stride=2, padding=1),
               nn.BatchNorm2d(128),
               nn.LeakyReLU(0.2, inplace=True),
               
               nn.Conv2d(128, 256, 3, padding=1),
               nn.BatchNorm2d(256),
               nn.LeakyReLU(0.2, inplace=True),
               
               nn.Conv2d(256, 256, 3, stride=2, padding=1),
               nn.BatchNorm2d(256),
               nn.LeakyReLU(0.2, inplace=True),
               
               nn.Conv2d(256, 512, 3, padding=1),
               nn.BatchNorm2d(512),
               nn.LeakyReLU(0.2, inplace=True),
               
               nn.Conv2d(512, 512, 3, stride=2, padding=1),
               nn.BatchNorm2d(512),
               nn.LeakyReLU(0.2, inplace=True),
               
               nn.AdaptiveAvgPool2d(1),
               nn.Conv2d(512, 1024, 1),
               nn.LeakyReLU(0.2, inplace=True),
               nn.Conv2d(1024, 1, 1)
           )

       def forward(self, x):
           return self.net(x)

   generator = ESRGANGenerator(scale_factor=4)
   discriminator = ESRGANDiscriminator()

   # Use different optimizers for G and D
   g_optimizer = torchium.optimizers.Adam(
       generator.parameters(),
       lr=1e-4,
       betas=(0.9, 0.999)
   )
   
   d_optimizer = torchium.optimizers.Adam(
       discriminator.parameters(),
       lr=1e-4,
       betas=(0.9, 0.999)
   )

   # Perceptual super resolution loss
   class PerceptualSRLoss(nn.Module):
       def __init__(self):
           super().__init__()
           self.mse = torchium.losses.MSELoss()
           self.perceptual = torchium.losses.PerceptualLoss()
           self.ssim = torchium.losses.SSIMLoss()
           self.vgg = torchium.losses.VGGLoss()
           self.gan_loss = torchium.losses.GANLoss()

       def forward(self, sr_pred, hr_target, sr_features, hr_features, 
                   fake_pred, real_pred, loss_type='generator'):
           if loss_type == 'generator':
               mse_loss = self.mse(sr_pred, hr_target)
               perceptual_loss = self.perceptual(sr_features, hr_features)
               ssim_loss = self.ssim(sr_pred, hr_target)
               vgg_loss = self.vgg(sr_pred, hr_target)
               gan_loss = self.gan_loss(fake_pred, real_pred)
               
               return (0.01 * mse_loss + 
                       0.006 * perceptual_loss + 
                       0.01 * ssim_loss + 
                       0.01 * vgg_loss + 
                       0.005 * gan_loss)
           else:  # discriminator
               return self.gan_loss(fake_pred, real_pred)

   criterion = PerceptualSRLoss()

   # Training loop for super resolution
   def train_sr_model(generator, discriminator, g_optimizer, d_optimizer, 
                     criterion, dataloader, num_epochs=100):
       generator.train()
       discriminator.train()
       
       for epoch in range(num_epochs):
           g_loss_total = 0
           d_loss_total = 0
           
           for batch in dataloader:
               lr_images = batch.lr_images
               hr_images = batch.hr_images
               
               # Train Discriminator
               d_optimizer.zero_grad()
               
               # Real images
               real_pred = discriminator(hr_images)
               real_target = torch.ones_like(real_pred)
               
               # Fake images
               fake_images = generator(lr_images)
               fake_pred = discriminator(fake_images.detach())
               fake_target = torch.zeros_like(fake_pred)
               
               d_loss = criterion(None, None, None, None, fake_pred, real_pred, 'discriminator')
               d_loss.backward()
               d_optimizer.step()
               
               # Train Generator
               g_optimizer.zero_grad()
               
               fake_images = generator(lr_images)
               fake_pred = discriminator(fake_images)
               real_target = torch.ones_like(fake_pred)
               
               # Extract features for perceptual loss
               sr_features = extract_features(fake_images)
               hr_features = extract_features(hr_images)
               
               g_loss = criterion(fake_images, hr_images, sr_features, hr_features, 
                                 fake_pred, real_target, 'generator')
               g_loss.backward()
               g_optimizer.step()
               
               g_loss_total += g_loss.item()
               d_loss_total += d_loss.item()
           
           print(f'Epoch {epoch}, G Loss: {g_loss_total/len(dataloader):.4f}, '
                 f'D Loss: {d_loss_total/len(dataloader):.4f}')

Style Transfer
--------------

Neural Style Transfer with Advanced Losses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torchvision.models as models

   class StyleTransferModel(nn.Module):
       def __init__(self):
           super().__init__()
           # Use pre-trained VGG as feature extractor
           vgg = models.vgg19(pretrained=True).features
           self.features = nn.ModuleList(vgg[:36])  # Up to conv4_4

       def forward(self, x):
           features = []
           for layer in self.features:
               x = layer(x)
               if isinstance(layer, nn.Conv2d):
                   features.append(x)
           return features

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

       def forward(self, generated_features, content_features, style_features):
           content_loss = self.content_loss(generated_features, content_features)
           style_loss = self.style_loss(generated_features, style_features)
           tv_loss = self.tv_loss(generated_features)
           
           return (1.0 * content_loss + 
                   100.0 * style_loss + 
                   0.1 * tv_loss)

   criterion = NeuralStyleLoss()

   # Training loop for style transfer
   def train_style_transfer(model, optimizer, criterion, content_image, style_image, num_epochs=1000):
       model.train()
       
       # Initialize generated image
       generated_image = content_image.clone().requires_grad_(True)
       
       for epoch in range(num_epochs):
           optimizer.zero_grad()
           
           # Extract features
           generated_features = model(generated_image)
           content_features = model(content_image)
           style_features = model(style_image)
           
           # Compute loss
           loss = criterion(generated_features, content_features, style_features)
           
           # Backward pass
           loss.backward()
           optimizer.step()
           
           if epoch % 100 == 0:
               print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

Multi-Task Computer Vision
--------------------------

Multi-Task Learning for Vision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class MultiTaskVisionModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.backbone = nn.Sequential(
               nn.Conv2d(3, 64, 3, padding=1),
               nn.BatchNorm2d(64),
               nn.ReLU(),
               nn.Conv2d(64, 128, 3, padding=1),
               nn.BatchNorm2d(128),
               nn.ReLU(),
               nn.AdaptiveAvgPool2d(1)
           )
           self.classifier = nn.Linear(128, 10)  # Classification
           self.regressor = nn.Linear(128, 1)    # Regression
           self.segmenter = nn.Linear(128, 21)   # Segmentation

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

   # Training loop for multi-task learning
   def train_multitask_model(model, optimizer, criterion, dataloader, num_epochs=100):
       model.train()
       
       for epoch in range(num_epochs):
           total_loss = 0
           
           for batch in dataloader:
               optimizer.zero_grad()
               
               # Forward pass
               features = model.backbone(batch.images)
               cls_pred = model.classifier(features)
               reg_pred = model.regressor(features)
               seg_pred = model.segmenter(features)
               
               # Compute loss
               loss = criterion(cls_pred, reg_pred, seg_pred, 
                               batch.cls_targets, batch.reg_targets, batch.seg_targets)
               
               # Backward pass with gradient surgery
               loss.backward()
               optimizer.step()
               
               total_loss += loss.item()
           
           avg_loss = total_loss / len(dataloader)
           print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')

These examples demonstrate the power of Torchium's specialized optimizers and loss functions for various computer vision tasks. Each example shows how to combine different components effectively for optimal performance.

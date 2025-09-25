Generative Models Examples
==========================

This section provides comprehensive examples for generative models using Torchium's specialized optimizers and loss functions.

GAN Training
------------

DCGAN Implementation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   import torchium

   class DCGANGenerator(nn.Module):
       def __init__(self, latent_dim=100, output_channels=3, output_size=64):
           super().__init__()
           self.latent_dim = latent_dim
           
           # Calculate the initial size after transposed convolutions
           initial_size = output_size // 8
           
           self.main = nn.Sequential(
               # Input: latent_dim x 1 x 1
               nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
               nn.BatchNorm2d(512),
               nn.ReLU(True),
               
               # State: 512 x 4 x 4
               nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
               nn.BatchNorm2d(256),
               nn.ReLU(True),
               
               # State: 256 x 8 x 8
               nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
               nn.BatchNorm2d(128),
               nn.ReLU(True),
               
               # State: 128 x 16 x 16
               nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
               nn.BatchNorm2d(64),
               nn.ReLU(True),
               
               # State: 64 x 32 x 32
               nn.ConvTranspose2d(64, output_channels, 4, 2, 1, bias=False),
               nn.Tanh()
               # Output: output_channels x 64 x 64
           )

       def forward(self, input):
           return self.main(input)

   class DCGANDiscriminator(nn.Module):
       def __init__(self, input_channels=3, input_size=64):
           super().__init__()
           
           self.main = nn.Sequential(
               # Input: input_channels x 64 x 64
               nn.Conv2d(input_channels, 64, 4, 2, 1, bias=False),
               nn.LeakyReLU(0.2, inplace=True),
               
               # State: 64 x 32 x 32
               nn.Conv2d(64, 128, 4, 2, 1, bias=False),
               nn.BatchNorm2d(128),
               nn.LeakyReLU(0.2, inplace=True),
               
               # State: 128 x 16 x 16
               nn.Conv2d(128, 256, 4, 2, 1, bias=False),
               nn.BatchNorm2d(256),
               nn.LeakyReLU(0.2, inplace=True),
               
               # State: 256 x 8 x 8
               nn.Conv2d(256, 512, 4, 2, 1, bias=False),
               nn.BatchNorm2d(512),
               nn.LeakyReLU(0.2, inplace=True),
               
               # State: 512 x 4 x 4
               nn.Conv2d(512, 1, 4, 1, 0, bias=False),
               nn.Sigmoid()
               # Output: 1 x 1 x 1
           )

       def forward(self, input):
           return self.main(input).view(-1, 1).squeeze(1)

   generator = DCGANGenerator(latent_dim=100, output_channels=3, output_size=64)
   discriminator = DCGANDiscriminator(input_channels=3, input_size=64)

   # Use different optimizers for G and D
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
           self.least_squares_loss = torchium.losses.LeastSquaresGANLoss()

       def forward(self, fake_pred, real_pred, loss_type='gan'):
           if loss_type == 'gan':
               return self.gan_loss(fake_pred, real_pred)
           elif loss_type == 'wasserstein':
               return self.wasserstein_loss(fake_pred, real_pred)
           elif loss_type == 'hinge':
               return self.hinge_loss(fake_pred, real_pred)
           elif loss_type == 'least_squares':
               return self.least_squares_loss(fake_pred, real_pred)

   criterion = AdvancedGANLoss()

   # Training loop for DCGAN
   def train_dcgan(generator, discriminator, g_optimizer, d_optimizer, 
                   criterion, dataloader, num_epochs=100, loss_type='gan'):
       generator.train()
       discriminator.train()
       
       for epoch in range(num_epochs):
           g_loss_total = 0
           d_loss_total = 0
           
           for batch in dataloader:
               real_images = batch.images
               batch_size = real_images.size(0)
               
               # Train Discriminator
               d_optimizer.zero_grad()
               
               # Real images
               real_pred = discriminator(real_images)
               real_target = torch.ones_like(real_pred)
               
               # Fake images
               noise = torch.randn(batch_size, 100, 1, 1)
               fake_images = generator(noise)
               fake_pred = discriminator(fake_images.detach())
               fake_target = torch.zeros_like(fake_pred)
               
               d_loss = criterion(fake_pred, real_pred, loss_type)
               d_loss.backward()
               d_optimizer.step()
               
               # Train Generator
               g_optimizer.zero_grad()
               
               fake_pred = discriminator(fake_images)
               real_target = torch.ones_like(fake_pred)
               
               g_loss = criterion(fake_pred, real_target, loss_type)
               g_loss.backward()
               g_optimizer.step()
               
               g_loss_total += g_loss.item()
               d_loss_total += d_loss.item()
           
           print(f'Epoch {epoch}, G Loss: {g_loss_total/len(dataloader):.4f}, '
                 f'D Loss: {d_loss_total/len(dataloader):.4f}')

Wasserstein GAN
~~~~~~~~~~~~~~~

.. code-block:: python

   class WGANGenerator(nn.Module):
       def __init__(self, latent_dim=100, output_channels=3, output_size=64):
           super().__init__()
           self.latent_dim = latent_dim
           
           self.main = nn.Sequential(
               nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
               nn.BatchNorm2d(512),
               nn.ReLU(True),
               
               nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
               nn.BatchNorm2d(256),
               nn.ReLU(True),
               
               nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
               nn.BatchNorm2d(128),
               nn.ReLU(True),
               
               nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
               nn.BatchNorm2d(64),
               nn.ReLU(True),
               
               nn.ConvTranspose2d(64, output_channels, 4, 2, 1, bias=False),
               nn.Tanh()
           )

       def forward(self, input):
           return self.main(input)

   class WGANDiscriminator(nn.Module):
       def __init__(self, input_channels=3, input_size=64):
           super().__init__()
           
           self.main = nn.Sequential(
               nn.Conv2d(input_channels, 64, 4, 2, 1, bias=False),
               nn.LeakyReLU(0.2, inplace=True),
               
               nn.Conv2d(64, 128, 4, 2, 1, bias=False),
               nn.BatchNorm2d(128),
               nn.LeakyReLU(0.2, inplace=True),
               
               nn.Conv2d(128, 256, 4, 2, 1, bias=False),
               nn.BatchNorm2d(256),
               nn.LeakyReLU(0.2, inplace=True),
               
               nn.Conv2d(256, 512, 4, 2, 1, bias=False),
               nn.BatchNorm2d(512),
               nn.LeakyReLU(0.2, inplace=True),
               
               nn.Conv2d(512, 1, 4, 1, 0, bias=False)
           )

       def forward(self, input):
           return self.main(input).view(-1, 1).squeeze(1)

   generator = WGANGenerator(latent_dim=100, output_channels=3, output_size=64)
   discriminator = WGANDiscriminator(input_channels=3, input_size=64)

   # Use RMSprop for WGAN
   g_optimizer = torchium.optimizers.RMSprop(
       generator.parameters(),
       lr=5e-5
   )
   
   d_optimizer = torchium.optimizers.RMSprop(
       discriminator.parameters(),
       lr=5e-5
   )

   # Wasserstein loss
   criterion = torchium.losses.WassersteinLoss()

   # Training loop for WGAN
   def train_wgan(generator, discriminator, g_optimizer, d_optimizer, 
                  criterion, dataloader, num_epochs=100, n_critic=5):
       generator.train()
       discriminator.train()
       
       for epoch in range(num_epochs):
           g_loss_total = 0
           d_loss_total = 0
           
           for batch in dataloader:
               real_images = batch.images
               batch_size = real_images.size(0)
               
               # Train Discriminator (n_critic times)
               for _ in range(n_critic):
                   d_optimizer.zero_grad()
                   
                   # Real images
                   real_pred = discriminator(real_images)
                   
                   # Fake images
                   noise = torch.randn(batch_size, 100, 1, 1)
                   fake_images = generator(noise)
                   fake_pred = discriminator(fake_images.detach())
                   
                   d_loss = criterion(fake_pred, real_pred)
                   d_loss.backward()
                   
                   # Gradient clipping for WGAN
                   torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=0.01)
                   
                   d_optimizer.step()
                   
                   d_loss_total += d_loss.item()
               
               # Train Generator
               g_optimizer.zero_grad()
               
               fake_pred = discriminator(fake_images)
               
               g_loss = criterion(fake_pred, real_pred)
               g_loss.backward()
               g_optimizer.step()
               
               g_loss_total += g_loss.item()
           
           print(f'Epoch {epoch}, G Loss: {g_loss_total/len(dataloader):.4f}, '
                 f'D Loss: {d_loss_total/len(dataloader):.4f}')

VAE Training
------------

Beta-VAE Implementation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class BetaVAE(nn.Module):
       def __init__(self, input_dim=784, latent_dim=20, beta=1.0):
           super().__init__()
           self.beta = beta
           
           # Encoder
           self.encoder = nn.Sequential(
               nn.Linear(input_dim, 400),
               nn.ReLU(),
               nn.Linear(400, 400),
               nn.ReLU()
           )
           
           self.mu = nn.Linear(400, latent_dim)
           self.log_var = nn.Linear(400, latent_dim)
           
           # Decoder
           self.decoder = nn.Sequential(
               nn.Linear(latent_dim, 400),
               nn.ReLU(),
               nn.Linear(400, 400),
               nn.ReLU(),
               nn.Linear(400, input_dim),
               nn.Sigmoid()
           )

       def encode(self, x):
           h = self.encoder(x)
           mu = self.mu(h)
           log_var = self.log_var(h)
           return mu, log_var

       def reparameterize(self, mu, log_var):
           std = torch.exp(0.5 * log_var)
           eps = torch.randn_like(std)
           return mu + eps * std

       def decode(self, z):
           return self.decoder(z)

       def forward(self, x):
           mu, log_var = self.encode(x)
           z = self.reparameterize(mu, log_var)
           recon_x = self.decode(z)
           return recon_x, mu, log_var

   model = BetaVAE(input_dim=784, latent_dim=20, beta=4.0)

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

   # Training loop for Beta-VAE
   def train_beta_vae(model, optimizer, criterion, dataloader, num_epochs=100):
       model.train()
       
       for epoch in range(num_epochs):
           total_loss = 0
           
           for batch in dataloader:
               optimizer.zero_grad()
               
               # Forward pass
               recon_x, mu, log_var = model(batch.images)
               
               # Compute loss
               loss = criterion(recon_x, batch.images, mu, log_var)
               
               # Backward pass
               loss.backward()
               optimizer.step()
               
               total_loss += loss.item()
           
           avg_loss = total_loss / len(dataloader)
           print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')

Factor-VAE Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class FactorVAE(nn.Module):
       def __init__(self, input_dim=784, latent_dim=20, gamma=1.0):
           super().__init__()
           self.gamma = gamma
           
           # Encoder
           self.encoder = nn.Sequential(
               nn.Linear(input_dim, 400),
               nn.ReLU(),
               nn.Linear(400, 400),
               nn.ReLU()
           )
           
           self.mu = nn.Linear(400, latent_dim)
           self.log_var = nn.Linear(400, latent_dim)
           
           # Decoder
           self.decoder = nn.Sequential(
               nn.Linear(latent_dim, 400),
               nn.ReLU(),
               nn.Linear(400, 400),
               nn.ReLU(),
               nn.Linear(400, input_dim),
               nn.Sigmoid()
           )

       def encode(self, x):
           h = self.encoder(x)
           mu = self.mu(h)
           log_var = self.log_var(h)
           return mu, log_var

       def reparameterize(self, mu, log_var):
           std = torch.exp(0.5 * log_var)
           eps = torch.randn_like(std)
           return mu + eps * std

       def decode(self, z):
           return self.decoder(z)

       def forward(self, x):
           mu, log_var = self.encode(x)
           z = self.reparameterize(mu, log_var)
           recon_x = self.decode(z)
           return recon_x, mu, log_var, z

   model = FactorVAE(input_dim=784, latent_dim=20, gamma=1.0)

   # Use AdamW for Factor-VAE
   optimizer = torchium.optimizers.AdamW(
       model.parameters(),
       lr=1e-3,
       betas=(0.9, 0.999),
       eps=1e-8,
       weight_decay=1e-4
   )

   # Factor-VAE loss
   criterion = torchium.losses.FactorVAELoss(gamma=1.0)

   # Training loop for Factor-VAE
   def train_factor_vae(model, optimizer, criterion, dataloader, num_epochs=100):
       model.train()
       
       for epoch in range(num_epochs):
           total_loss = 0
           
           for batch in dataloader:
               optimizer.zero_grad()
               
               # Forward pass
               recon_x, mu, log_var, z = model(batch.images)
               
               # Compute loss
               loss = criterion(recon_x, batch.images, mu, log_var, z)
               
               # Backward pass
               loss.backward()
               optimizer.step()
               
               total_loss += loss.item()
           
           avg_loss = total_loss / len(dataloader)
           print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')

Diffusion Models
----------------

DDPM Implementation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class DDPM(nn.Module):
       def __init__(self, input_dim=784, hidden_dim=512, num_timesteps=1000):
           super().__init__()
           self.num_timesteps = num_timesteps
           
           # Noise prediction network
           self.net = nn.Sequential(
               nn.Linear(input_dim + 1, hidden_dim),  # +1 for timestep
               nn.ReLU(),
               nn.Linear(hidden_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, input_dim)
           )

       def forward(self, x, t):
           # Concatenate timestep
           t_emb = t.unsqueeze(-1).float() / self.num_timesteps
           x_with_t = torch.cat([x, t_emb], dim=-1)
           
           return self.net(x_with_t)

   model = DDPM(input_dim=784, hidden_dim=512, num_timesteps=1000)

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

   # Training loop for DDPM
   def train_ddpm(model, optimizer, criterion, dataloader, num_epochs=100):
       model.train()
       
       for epoch in range(num_epochs):
           total_loss = 0
           
           for batch in dataloader:
               optimizer.zero_grad()
               
               # Sample random timesteps
               t = torch.randint(0, model.num_timesteps, (batch.images.size(0),))
               
               # Add noise
               noise = torch.randn_like(batch.images)
               noisy_images = batch.images + noise
               
               # Predict noise
               predicted_noise = model(noisy_images, t)
               
               # Compute loss
               loss = criterion(predicted_noise, noise)
               
               # Backward pass
               loss.backward()
               optimizer.step()
               
               total_loss += loss.item()
           
           avg_loss = total_loss / len(dataloader)
           print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')

DDIM Implementation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class DDIM(nn.Module):
       def __init__(self, input_dim=784, hidden_dim=512, num_timesteps=1000):
           super().__init__()
           self.num_timesteps = num_timesteps
           
           # Noise prediction network
           self.net = nn.Sequential(
               nn.Linear(input_dim + 1, hidden_dim),  # +1 for timestep
               nn.ReLU(),
               nn.Linear(hidden_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, input_dim)
           )

       def forward(self, x, t):
           # Concatenate timestep
           t_emb = t.unsqueeze(-1).float() / self.num_timesteps
           x_with_t = torch.cat([x, t_emb], dim=-1)
           
           return self.net(x_with_t)

   model = DDIM(input_dim=784, hidden_dim=512, num_timesteps=1000)

   # Use AdamW for DDIM
   optimizer = torchium.optimizers.AdamW(
       model.parameters(),
       lr=1e-4,
       betas=(0.9, 0.999),
       eps=1e-8,
       weight_decay=1e-4
   )

   # DDIM loss
   criterion = torchium.losses.DDIMLoss()

   # Training loop for DDIM
   def train_ddim(model, optimizer, criterion, dataloader, num_epochs=100):
       model.train()
       
       for epoch in range(num_epochs):
           total_loss = 0
           
           for batch in dataloader:
               optimizer.zero_grad()
               
               # Sample random timesteps
               t = torch.randint(0, model.num_timesteps, (batch.images.size(0),))
               
               # Add noise
               noise = torch.randn_like(batch.images)
               noisy_images = batch.images + noise
               
               # Predict noise
               predicted_noise = model(noisy_images, t)
               
               # Compute loss
               loss = criterion(predicted_noise, noise)
               
               # Backward pass
               loss.backward()
               optimizer.step()
               
               total_loss += loss.item()
           
           avg_loss = total_loss / len(dataloader)
           print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')

Score Matching
~~~~~~~~~~~~~~

.. code-block:: python

   class ScoreMatchingModel(nn.Module):
       def __init__(self, input_dim=784, hidden_dim=512):
           super().__init__()
           
           # Score prediction network
           self.net = nn.Sequential(
               nn.Linear(input_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, input_dim)
           )

       def forward(self, x):
           return self.net(x)

   model = ScoreMatchingModel(input_dim=784, hidden_dim=512)

   # Use AdamW for score matching
   optimizer = torchium.optimizers.AdamW(
       model.parameters(),
       lr=1e-4,
       betas=(0.9, 0.999),
       eps=1e-8,
       weight_decay=1e-4
   )

   # Score matching loss
   criterion = torchium.losses.ScoreMatchingLoss()

   # Training loop for score matching
   def train_score_matching(model, optimizer, criterion, dataloader, num_epochs=100):
       model.train()
       
       for epoch in range(num_epochs):
           total_loss = 0
           
           for batch in dataloader:
               optimizer.zero_grad()
               
               # Add noise
               noise = torch.randn_like(batch.images)
               noisy_images = batch.images + noise
               
               # Predict score
               predicted_score = model(noisy_images)
               
               # Compute loss
               loss = criterion(predicted_score, noise)
               
               # Backward pass
               loss.backward()
               optimizer.step()
               
               total_loss += loss.item()
           
           avg_loss = total_loss / len(dataloader)
           print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')

Multi-Task Generative Models
----------------------------

Multi-Task GAN
~~~~~~~~~~~~~~

.. code-block:: python

   class MultiTaskGANGenerator(nn.Module):
       def __init__(self, latent_dim=100, output_channels=3, output_size=64, num_tasks=3):
           super().__init__()
           self.latent_dim = latent_dim
           self.num_tasks = num_tasks
           
           # Shared layers
           self.shared = nn.Sequential(
               nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
               nn.BatchNorm2d(512),
               nn.ReLU(True),
               
               nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
               nn.BatchNorm2d(256),
               nn.ReLU(True)
           )
           
           # Task-specific heads
           self.task_heads = nn.ModuleList([
               nn.Sequential(
                   nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                   nn.BatchNorm2d(128),
                   nn.ReLU(True),
                   
                   nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                   nn.BatchNorm2d(64),
                   nn.ReLU(True),
                   
                   nn.ConvTranspose2d(64, output_channels, 4, 2, 1, bias=False),
                   nn.Tanh()
               ) for _ in range(num_tasks)
           ])

       def forward(self, input, task_id):
           shared_features = self.shared(input)
           return self.task_heads[task_id](shared_features)

   class MultiTaskGANDiscriminator(nn.Module):
       def __init__(self, input_channels=3, input_size=64, num_tasks=3):
           super().__init__()
           self.num_tasks = num_tasks
           
           # Shared layers
           self.shared = nn.Sequential(
               nn.Conv2d(input_channels, 64, 4, 2, 1, bias=False),
               nn.LeakyReLU(0.2, inplace=True),
               
               nn.Conv2d(64, 128, 4, 2, 1, bias=False),
               nn.BatchNorm2d(128),
               nn.LeakyReLU(0.2, inplace=True),
               
               nn.Conv2d(128, 256, 4, 2, 1, bias=False),
               nn.BatchNorm2d(256),
               nn.LeakyReLU(0.2, inplace=True)
           )
           
           # Task-specific heads
           self.task_heads = nn.ModuleList([
               nn.Sequential(
                   nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                   nn.BatchNorm2d(512),
                   nn.LeakyReLU(0.2, inplace=True),
                   
                   nn.Conv2d(512, 1, 4, 1, 0, bias=False),
                   nn.Sigmoid()
               ) for _ in range(num_tasks)
           ])

       def forward(self, input, task_id):
           shared_features = self.shared(input)
           return self.task_heads[task_id](shared_features).view(-1, 1).squeeze(1)

   generator = MultiTaskGANGenerator(latent_dim=100, output_channels=3, output_size=64, num_tasks=3)
   discriminator = MultiTaskGANDiscriminator(input_channels=3, input_size=64, num_tasks=3)

   # Use different optimizers for G and D
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

   # Multi-task GAN loss
   class MultiTaskGANLoss(nn.Module):
       def __init__(self, num_tasks=3):
           super().__init__()
           self.num_tasks = num_tasks
           self.uncertainty_loss = torchium.losses.UncertaintyWeightingLoss(num_tasks)
           self.gan_loss = torchium.losses.GANLoss()

       def forward(self, fake_preds, real_preds):
           losses = []
           for i in range(self.num_tasks):
               loss = self.gan_loss(fake_preds[i], real_preds[i])
               losses.append(loss)
           
           return self.uncertainty_loss(losses)

   criterion = MultiTaskGANLoss(num_tasks=3)

   # Training loop for multi-task GAN
   def train_multitask_gan(generator, discriminator, g_optimizer, d_optimizer, 
                           criterion, dataloader, num_epochs=100):
       generator.train()
       discriminator.train()
       
       for epoch in range(num_epochs):
           g_loss_total = 0
           d_loss_total = 0
           
           for batch in dataloader:
               real_images = batch.images
               task_ids = batch.task_ids
               batch_size = real_images.size(0)
               
               # Train Discriminator
               d_optimizer.zero_grad()
               
               # Real images
               real_preds = []
               for i in range(batch_size):
                   real_pred = discriminator(real_images[i:i+1], task_ids[i])
                   real_preds.append(real_pred)
               
               # Fake images
               noise = torch.randn(batch_size, 100, 1, 1)
               fake_images = generator(noise, task_ids)
               fake_preds = []
               for i in range(batch_size):
                   fake_pred = discriminator(fake_images[i:i+1].detach(), task_ids[i])
                   fake_preds.append(fake_pred)
               
               d_loss = criterion(fake_preds, real_preds)
               d_loss.backward()
               d_optimizer.step()
               
               # Train Generator
               g_optimizer.zero_grad()
               
               fake_preds = []
               for i in range(batch_size):
                   fake_pred = discriminator(fake_images[i:i+1], task_ids[i])
                   fake_preds.append(fake_pred)
               
               g_loss = criterion(fake_preds, real_preds)
               g_loss.backward()
               g_optimizer.step()
               
               g_loss_total += g_loss.item()
               d_loss_total += d_loss.item()
           
           print(f'Epoch {epoch}, G Loss: {g_loss_total/len(dataloader):.4f}, '
                 f'D Loss: {d_loss_total/len(dataloader):.4f}')

These examples demonstrate the power of Torchium's specialized optimizers and loss functions for various generative models. Each example shows how to combine different components effectively for optimal performance in generative modeling.

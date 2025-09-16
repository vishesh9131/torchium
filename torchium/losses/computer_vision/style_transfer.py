import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

# Optional torchvision import with fallback
try:
    import torchvision.models as models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    models = None

def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """Compute Gram matrix for style representation"""
    batch_size, channels, height, width = features.size()
    features = features.view(batch_size, channels, height * width)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (channels * height * width)

class StyleLoss(nn.Module):
    """Style loss using Gram matrices for neural style transfer"""
    def __init__(self, style_layers: Optional[List[str]] = None, style_weights: Optional[List[float]] = None):
        super().__init__()
        if style_layers is None:
            style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        if style_weights is None:
            style_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        
        self.style_layers = style_layers
        self.style_weights = style_weights
        
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for StyleLoss. Install with: pip install torchvision")
        
        # load pre-trained VGG19
        vgg = models.vgg19(pretrained=True).features
        
        # freeze parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        # create feature extractors
        self.feature_extractors = nn.ModuleDict()
        layer_indices = {
            'conv_1': 1, 'conv_2': 6, 'conv_3': 11, 'conv_4': 20, 'conv_5': 29
        }
        
        for layer_name in style_layers:
            if layer_name in layer_indices:
                self.feature_extractors[layer_name] = nn.Sequential(
                    *list(vgg.children())[:layer_indices[layer_name]+1]
                )
        
        self.mse_loss = nn.MSELoss()
    
    def forward(self, generated: torch.Tensor, style_target: torch.Tensor) -> torch.Tensor:
        # normalize inputs to [0,1] if needed
        if generated.max() > 1.0:
            generated = generated / 255.0
        if style_target.max() > 1.0:
            style_target = style_target / 255.0
        
        # normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(generated.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(generated.device)
        
        generated_norm = (generated - mean) / std
        style_target_norm = (style_target - mean) / std
        
        total_style_loss = 0
        for i, layer_name in enumerate(self.style_layers):
            # extract features
            generated_features = self.feature_extractors[layer_name](generated_norm)
            style_features = self.feature_extractors[layer_name](style_target_norm)
            
            # compute Gram matrices
            generated_gram = gram_matrix(generated_features)
            style_gram = gram_matrix(style_features)
            
            # compute style loss for this layer
            layer_style_loss = self.mse_loss(generated_gram, style_gram)
            total_style_loss += self.style_weights[i] * layer_style_loss
        
        return total_style_loss

class ContentLoss(nn.Module):
    """Content loss for neural style transfer"""
    def __init__(self, content_layers: Optional[List[str]] = None, content_weights: Optional[List[float]] = None):
        super().__init__()
        if content_layers is None:
            content_layers = ['conv_4']  # typically use conv4_2 layer
        if content_weights is None:
            content_weights = [1.0] * len(content_layers)
        
        self.content_layers = content_layers
        self.content_weights = content_weights
        
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for ContentLoss. Install with: pip install torchvision")
        
        # load pre-trained VGG19
        vgg = models.vgg19(pretrained=True).features
        
        # freeze parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        # create feature extractors
        self.feature_extractors = nn.ModuleDict()
        layer_indices = {
            'conv_1': 1, 'conv_2': 6, 'conv_3': 11, 'conv_4': 20, 'conv_5': 29
        }
        
        for layer_name in content_layers:
            if layer_name in layer_indices:
                self.feature_extractors[layer_name] = nn.Sequential(
                    *list(vgg.children())[:layer_indices[layer_name]+1]
                )
        
        self.mse_loss = nn.MSELoss()
    
    def forward(self, generated: torch.Tensor, content_target: torch.Tensor) -> torch.Tensor:
        # normalize inputs
        if generated.max() > 1.0:
            generated = generated / 255.0
        if content_target.max() > 1.0:
            content_target = content_target / 255.0
        
        # normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(generated.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(generated.device)
        
        generated_norm = (generated - mean) / std
        content_target_norm = (content_target - mean) / std
        
        total_content_loss = 0
        for i, layer_name in enumerate(self.content_layers):
            # extract features
            generated_features = self.feature_extractors[layer_name](generated_norm)
            content_features = self.feature_extractors[layer_name](content_target_norm)
            
            # compute content loss for this layer
            layer_content_loss = self.mse_loss(generated_features, content_features)
            total_content_loss += self.content_weights[i] * layer_content_loss
        
        return total_content_loss

class TotalVariationLoss(nn.Module):
    """Total Variation loss for image smoothing and noise reduction"""
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, channels, height, width]
        batch_size, channels, height, width = x.size()
        
        # compute differences
        tv_height = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        tv_width = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
        
        # total variation
        tv_loss = self.weight * (tv_height + tv_width) / (batch_size * channels * height * width)
        
        return tv_loss

class NeuralStyleLoss(nn.Module):
    """Combined loss for neural style transfer"""
    def __init__(self, content_weight: float = 1.0, style_weight: float = 1000.0, tv_weight: float = 1.0):
        super().__init__()
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        
        self.content_loss = ContentLoss()
        self.style_loss = StyleLoss()
        self.tv_loss = TotalVariationLoss()
    
    def forward(self, generated: torch.Tensor, content_target: torch.Tensor, style_target: torch.Tensor) -> torch.Tensor:
        content_loss = self.content_loss(generated, content_target)
        style_loss = self.style_loss(generated, style_target)
        tv_loss = self.tv_loss(generated)
        
        total_loss = (self.content_weight * content_loss + 
                     self.style_weight * style_loss + 
                     self.tv_weight * tv_loss)
        
        return total_loss

class AdaINLoss(nn.Module):
    """Adaptive Instance Normalization loss for real-time style transfer"""
    def __init__(self, content_weight: float = 1.0, style_weight: float = 10.0):
        super().__init__()
        self.content_weight = content_weight
        self.style_weight = style_weight
        
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for AdaINLoss. Install with: pip install torchvision")
        
        # pre-trained VGG for feature extraction
        vgg = models.vgg19(pretrained=True).features
        for param in vgg.parameters():
            param.requires_grad = False
        
        # encoder up to relu4_1
        self.encoder = nn.Sequential(*list(vgg.children())[:21])
        
        self.mse_loss = nn.MSELoss()
    
    def forward(self, generated: torch.Tensor, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        # normalize inputs
        if generated.max() > 1.0:
            generated = generated / 255.0
        if content.max() > 1.0:
            content = content / 255.0
        if style.max() > 1.0:
            style = style / 255.0
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(generated.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(generated.device)
        
        generated_norm = (generated - mean) / std
        content_norm = (content - mean) / std
        style_norm = (style - mean) / std
        
        # extract features
        generated_feat = self.encoder(generated_norm)
        content_feat = self.encoder(content_norm)
        style_feat = self.encoder(style_norm)
        
        # content loss
        content_loss = self.mse_loss(generated_feat, content_feat)
        
        # style loss (using mean and std matching)
        generated_mean = torch.mean(generated_feat, dim=[2, 3], keepdim=True)
        generated_std = torch.std(generated_feat, dim=[2, 3], keepdim=True)
        style_mean = torch.mean(style_feat, dim=[2, 3], keepdim=True)
        style_std = torch.std(style_feat, dim=[2, 3], keepdim=True)
        
        style_loss = self.mse_loss(generated_mean, style_mean) + self.mse_loss(generated_std, style_std)
        
        total_loss = self.content_weight * content_loss + self.style_weight * style_loss
        
        return total_loss

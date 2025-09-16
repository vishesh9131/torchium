import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math

# Optional torchvision import with fallback
try:
    import torchvision.models as models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    models = None

# Placeholder implementations
class FocalDetectionLoss(nn.Module):
    def forward(self, x, y): return torch.tensor(0.0)

class GIoULoss(nn.Module):
    def forward(self, x, y): return torch.tensor(0.0)

class DIoULoss(nn.Module):
    def forward(self, x, y): return torch.tensor(0.0)

class CIoULoss(nn.Module):
    def forward(self, x, y): return torch.tensor(0.0)

class EIoULoss(nn.Module):
    def forward(self, x, y): return torch.tensor(0.0)

class AlphaIoULoss(nn.Module):
    def forward(self, x, y): return torch.tensor(0.0)

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features for image quality"""
    def __init__(self, feature_layers: Optional[List[int]] = None, use_gpu: bool = True):
        super().__init__()
        if feature_layers is None:
            feature_layers = [3, 8, 15, 22]  # conv1_2, conv2_2, conv3_3, conv4_3
        
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for PerceptualLoss. Install with: pip install torchvision")
        
        self.feature_layers = feature_layers
        vgg = models.vgg16(pretrained=True).features
        
        # freeze VGG parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        self.vgg = vgg
        self.mse_loss = nn.MSELoss()
        
        if use_gpu and torch.cuda.is_available():
            self.vgg.cuda()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # normalize inputs to [0,1] if they're not already
        if pred.max() > 1.0:
            pred = pred / 255.0
        if target.max() > 1.0:
            target = target / 255.0
        
        # VGG expects inputs normalized with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std
        
        # extract features
        pred_features = self._extract_features(pred_norm)
        target_features = self._extract_features(target_norm)
        
        # compute perceptual loss
        loss = 0
        for pred_feat, target_feat in zip(pred_features, target_features):
            loss += self.mse_loss(pred_feat, target_feat)
        
        return loss
    
    def _extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        return features

class SSIMLoss(nn.Module):
    """Structural Similarity Index loss for image quality"""
    def __init__(self, window_size: int = 11, sigma: float = 1.5, k1: float = 0.01, k2: float = 0.03, L: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.k1 = k1
        self.k2 = k2
        self.L = L
        
        # create gaussian window
        window = gaussian_kernel(window_size, sigma)
        self.register_buffer('window', window.unsqueeze(0).unsqueeze(0))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = pred.shape
        
        # create multi-channel window
        window = self.window.expand(channels, 1, self.window_size, self.window_size)
        
        # compute means
        mu1 = F.conv2d(pred, window, padding=self.window_size//2, groups=channels)
        mu2 = F.conv2d(target, window, padding=self.window_size//2, groups=channels)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # compute variances and covariance
        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size//2, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size//2, groups=channels) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size//2, groups=channels) - mu1_mu2
        
        # compute SSIM
        c1 = (self.k1 * self.L) ** 2
        c2 = (self.k2 * self.L) ** 2
        
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return 1 - ssim_map.mean()

class MSSSIMLoss(nn.Module):
    """Multi-Scale Structural Similarity Index loss"""
    def __init__(self, window_size: int = 11, sigma: float = 1.5, weights: Optional[List[float]] = None):
        super().__init__()
        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]  # default MS-SSIM weights
        
        self.weights = weights
        self.levels = len(weights)
        
        self.ssim_modules = nn.ModuleList([
            SSIMLoss(window_size, sigma) for _ in range(self.levels)
        ])
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mssim_values = []
        
        # compute SSIM at multiple scales
        for i in range(self.levels):
            ssim_val = self.ssim_modules[i](pred, target)
            mssim_values.append(ssim_val)
            
            # downsample for next scale
            if i < self.levels - 1:
                pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
                target = F.avg_pool2d(target, kernel_size=2, stride=2)
        
        # weighted combination
        ms_ssim = sum(w * ssim for w, ssim in zip(self.weights, mssim_values))
        return ms_ssim

class LPIPSLoss(nn.Module):
    """Learned Perceptual Image Patch Similarity (simplified version)"""
    def __init__(self, net_type: str = 'vgg', use_dropout: bool = True):
        super().__init__()
        self.net_type = net_type
        self.use_dropout = use_dropout
        
        if net_type == 'vgg':
            self.net = models.vgg16(pretrained=True).features
            self.feature_layers = [3, 8, 15, 22, 29]  # VGG conv layers
        elif net_type == 'alex':
            self.net = models.alexnet(pretrained=True).features
            self.feature_layers = [1, 4, 7, 9, 11]  # AlexNet conv layers
        else:
            raise ValueError(f"Unsupported network type: {net_type}")
        
        # freeze network parameters
        for param in self.net.parameters():
            param.requires_grad = False
        
        # learnable linear layers (simplified - in practice these would be learned)
        self.lin_layers = nn.ModuleList([
            nn.Sequential(nn.Identity(), nn.Dropout() if use_dropout else nn.Identity())
            for _ in self.feature_layers
        ])
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # normalize inputs
        if pred.max() > 1.0:
            pred = pred / 255.0
        if target.max() > 1.0:
            target = target / 255.0
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std
        
        # extract and compare features
        pred_features = self._extract_features(pred_norm)
        target_features = self._extract_features(target_norm)
        
        total_loss = 0
        for i, (pred_feat, target_feat) in enumerate(zip(pred_features, target_features)):
            # normalize features
            pred_feat = F.normalize(pred_feat, dim=1)
            target_feat = F.normalize(target_feat, dim=1)
            
            # compute spatial average of squared differences
            diff = (pred_feat - target_feat) ** 2
            loss = diff.mean(dim=[2, 3])  # spatial average
            loss = self.lin_layers[i](loss).mean()  # channel average
            total_loss += loss
        
        return total_loss
    
    def _extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        return features

class VGGLoss(nn.Module):
    """VGG-based perceptual loss for style transfer and super resolution"""
    def __init__(self, layers: Optional[List[str]] = None, weights: Optional[List[float]] = None):
        super().__init__()
        if layers is None:
            layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4']
        if weights is None:
            weights = [1.0] * len(layers)
        
        self.layers = layers
        self.weights = weights
        
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for VGGLoss. Install with: pip install torchvision")
        
        vgg = models.vgg19(pretrained=True).features
        
        # freeze parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        # create feature extractors
        self.feature_extractors = nn.ModuleDict()
        layer_names = {
            'conv_1': 1, 'conv_2': 6, 'conv_3': 11, 'conv_4': 20, 'conv_5': 29
        }
        
        for layer_name in layers:
            if layer_name in layer_names:
                self.feature_extractors[layer_name] = nn.Sequential(*list(vgg.children())[:layer_names[layer_name]+1])
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # normalize inputs
        if pred.max() > 1.0:
            pred = pred / 255.0
        if target.max() > 1.0:
            target = target / 255.0
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std
        
        total_loss = 0
        for i, layer_name in enumerate(self.layers):
            pred_feat = self.feature_extractors[layer_name](pred_norm)
            target_feat = self.feature_extractors[layer_name](target_norm)
            
            loss = F.mse_loss(pred_feat, target_feat)
            total_loss += self.weights[i] * loss
        
        return total_loss

class StyleLoss(nn.Module):
    def forward(self, x, y): return torch.tensor(0.0)

class ContentLoss(nn.Module):
    def forward(self, x, y): return torch.tensor(0.0)

class TotalVariationLoss(nn.Module):
    def forward(self, x, y): return torch.tensor(0.0)

def gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    """Create a 2D Gaussian kernel"""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.outer(g)

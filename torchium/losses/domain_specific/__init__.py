import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple
import math

# Medical Imaging Loss Functions
class DiceLoss(nn.Module):
    """Dice loss for medical image segmentation"""
    def __init__(self, smooth: float = 1e-5, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice_score

class MedicalImagingLoss(nn.Module):
    """Combined loss for medical imaging tasks"""
    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)
        return self.dice_weight * dice + self.ce_weight * ce

# Audio Processing Loss Functions
class SpectralLoss(nn.Module):
    """Spectral loss for audio processing"""
    def __init__(self, n_fft: int = 2048, alpha: float = 1.0):
        super().__init__()
        self.n_fft = n_fft
        self.alpha = alpha
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_spec = torch.stft(pred.squeeze(), n_fft=self.n_fft, return_complex=True)
        target_spec = torch.stft(target.squeeze(), n_fft=self.n_fft, return_complex=True)
        
        pred_mag = torch.abs(pred_spec)
        target_mag = torch.abs(target_spec)
        
        return F.l1_loss(pred_mag, target_mag)

class AudioProcessingLoss(nn.Module):
    """Multi-scale audio loss combining time and frequency domain"""
    def __init__(self, time_weight: float = 0.5, freq_weight: float = 0.5):
        super().__init__()
        self.time_weight = time_weight
        self.freq_weight = freq_weight
        self.spectral_loss = SpectralLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        time_loss = F.mse_loss(pred, target)
        freq_loss = self.spectral_loss(pred, target)
        return self.time_weight * time_loss + self.freq_weight * freq_loss

# Time Series Loss Functions
class DTWLoss(nn.Module):
    """Dynamic Time Warping loss for time series"""
    def __init__(self, use_cuda: bool = False):
        super().__init__()
        self.use_cuda = use_cuda
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # simplified DTW approximation using soft-DTW
        batch_size, seq_len, features = pred.shape
        
        # compute pairwise distances
        pred_expanded = pred.unsqueeze(2)  # [B, T, 1, F]
        target_expanded = target.unsqueeze(1)  # [B, 1, T, F]
        
        distances = torch.sum((pred_expanded - target_expanded) ** 2, dim=-1)  # [B, T, T]
        
        # soft minimum for differentiability
        gamma = 1.0
        min_distances = -gamma * torch.logsumexp(-distances / gamma, dim=-1)
        
        return torch.mean(min_distances)

class TimeSeriesLoss(nn.Module):
    """Comprehensive time series loss with trend and seasonality preservation"""
    def __init__(self, trend_weight: float = 0.3, seasonal_weight: float = 0.3, residual_weight: float = 0.4):
        super().__init__()
        self.trend_weight = trend_weight
        self.seasonal_weight = seasonal_weight
        self.residual_weight = residual_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # MSE for overall fit
        mse_loss = F.mse_loss(pred, target)
        
        # trend preservation (first differences)
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        trend_loss = F.mse_loss(pred_diff, target_diff)
        
        # seasonal pattern preservation (second differences)
        if pred.shape[1] > 2:
            pred_diff2 = pred_diff[:, 1:] - pred_diff[:, :-1]
            target_diff2 = target_diff[:, 1:] - target_diff[:, :-1]
            seasonal_loss = F.mse_loss(pred_diff2, target_diff2)
        else:
            seasonal_loss = torch.tensor(0.0, device=pred.device)
        
        return (self.residual_weight * mse_loss + 
                self.trend_weight * trend_loss + 
                self.seasonal_weight * seasonal_loss)

# NLP Losses
class PerplexityLoss(nn.Module):
    """Perplexity-based loss for language modeling"""
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce_loss(logits.view(-1, logits.size(-1)), targets.view(-1))
        return torch.exp(ce_loss)  # perplexity is exp(cross_entropy)

class CRFLoss(nn.Module):
    """Conditional Random Field loss for sequence labeling"""
    def __init__(self, num_tags: int, batch_first: bool = True):
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        
        # transition matrix: transitions[i][j] = score of transitioning from tag i to tag j
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
        # constrain transitions
        self.transitions.data[0, :] = -10000  # no transitions to START
        self.transitions.data[:, 0] = -10000  # no transitions from STOP
    
    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute negative log likelihood"""
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)
        
        # forward algorithm
        forward_score = self._forward_alg(emissions, mask)
        # score of gold sequence
        gold_score = self._score_sentence(emissions, tags, mask)
        
        return forward_score - gold_score
    
    def _forward_alg(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_tags = emissions.shape
        
        # initialize
        alpha = emissions[:, 0].clone()  # [batch_size, num_tags]
        
        for i in range(1, seq_len):
            emit_score = emissions[:, i].unsqueeze(1)  # [batch_size, 1, num_tags]
            trans_score = self.transitions.unsqueeze(0)  # [1, num_tags, num_tags]
            next_tag_var = alpha.unsqueeze(2) + trans_score + emit_score  # [batch_size, num_tags, num_tags]
            
            alpha_new = torch.logsumexp(next_tag_var, dim=1)  # [batch_size, num_tags]
            
            # apply mask
            mask_i = mask[:, i].unsqueeze(1)  # [batch_size, 1]
            alpha = alpha_new * mask_i + alpha * (1 - mask_i)
        
        return torch.logsumexp(alpha, dim=1)  # [batch_size]
    
    def _score_sentence(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = tags.shape
        
        # emission scores
        emit_scores = emissions.gather(2, tags.unsqueeze(2)).squeeze(2)  # [batch_size, seq_len]
        emit_scores = emit_scores * mask
        
        # transition scores
        trans_scores = torch.zeros(batch_size, device=emissions.device)
        for i in range(seq_len - 1):
            cur_tag = tags[:, i]
            next_tag = tags[:, i + 1]
            trans_scores += self.transitions[cur_tag, next_tag] * mask[:, i + 1]
        
        return emit_scores.sum(dim=1) + trans_scores

class StructuredPredictionLoss(nn.Module):
    """Structured prediction loss using max-margin"""
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # scores: [batch_size, num_structures]
        # targets: [batch_size] (indices of correct structures)
        batch_size = scores.size(0)
        
        correct_scores = scores.gather(1, targets.unsqueeze(1))  # [batch_size, 1]
        
        # max-margin loss
        margins = self.margin + scores - correct_scores  # [batch_size, num_structures]
        margins = margins.clamp(min=0)
        
        # zero out the correct class margin
        margins.scatter_(1, targets.unsqueeze(1), 0)
        
        return margins.max(dim=1)[0].mean()

# Metric Learning text similarity losses
class BLEULoss(nn.Module):
    """BLEU score based loss (1 - BLEU)"""
    def __init__(self, n_gram: int = 4, smooth: bool = True):
        super().__init__()
        self.n_gram = n_gram
        self.smooth = smooth
    
    def forward(self, pred_tokens: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        # simplified BLEU calculation for differentiable training
        # in practice, you'd use proper tokenization and n-gram counting
        
        # convert to sets for overlap calculation (simplified)
        batch_size = pred_tokens.size(0)
        bleu_scores = []
        
        for i in range(batch_size):
            pred_set = set(pred_tokens[i].cpu().numpy())
            target_set = set(target_tokens[i].cpu().numpy())
            
            if len(target_set) == 0:
                bleu_scores.append(0.0)
            else:
                overlap = len(pred_set.intersection(target_set))
                precision = overlap / len(pred_set) if len(pred_set) > 0 else 0.0
                bleu_scores.append(precision)
        
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        return torch.tensor(1.0 - avg_bleu, device=pred_tokens.device, requires_grad=True)

# Additional placeholder implementations for completeness

ROUGELoss = PerplexityLoss  # placeholder - would need proper ROUGE implementation
METEORLoss = PerplexityLoss  # placeholder - would need proper METEOR implementation
BERTScoreLoss = PerplexityLoss  # placeholder - would need BERT model for scoring

# Word embedding losses
class Word2VecLoss(nn.Module):
    """Skip-gram with negative sampling loss"""
    def __init__(self, vocab_size: int, embed_dim: int, num_negative: int = 5):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_negative = num_negative
        
        self.center_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)
    
    def forward(self, center_words: torch.Tensor, context_words: torch.Tensor, 
                negative_words: torch.Tensor) -> torch.Tensor:
        center_embeds = self.center_embeddings(center_words)  # [batch_size, embed_dim]
        context_embeds = self.context_embeddings(context_words)  # [batch_size, embed_dim]
        negative_embeds = self.context_embeddings(negative_words)  # [batch_size, num_negative, embed_dim]
        
        # positive score
        pos_score = torch.sum(center_embeds * context_embeds, dim=1)  # [batch_size]
        pos_loss = F.logsigmoid(pos_score)
        
        # negative scores
        neg_scores = torch.bmm(negative_embeds, center_embeds.unsqueeze(2)).squeeze(2)  # [batch_size, num_negative]
        neg_loss = F.logsigmoid(-neg_scores).sum(dim=1)  # [batch_size]
        
        return -(pos_loss + neg_loss).mean()

GloVeLoss = Word2VecLoss  # similar structure, would need matrix factorization approach
FastTextLoss = Word2VecLoss  # similar to word2vec but with character n-grams

# GAN Losses
class GANLoss(nn.Module):
    """Standard GAN loss (BCE)"""
    def __init__(self, use_lsgan: bool = False, target_real_label: float = 1.0, target_fake_label: float = 0.0):
        super().__init__()
        self.use_lsgan = use_lsgan
        self.target_real_label = target_real_label
        self.target_fake_label = target_fake_label
        
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        if target_is_real:
            target_tensor = torch.full_like(prediction, self.target_real_label)
        else:
            target_tensor = torch.full_like(prediction, self.target_fake_label)
        
        return self.loss(prediction, target_tensor)

class WassersteinLoss(nn.Module):
    """Wasserstein GAN loss"""
    def forward(self, real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        return fake_pred.mean() - real_pred.mean()

class HingeGANLoss(nn.Module):
    """Hinge loss for GANs"""
    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        if target_is_real:
            return F.relu(1.0 - prediction).mean()
        else:
            return F.relu(1.0 + prediction).mean()

class LeastSquaresGANLoss(nn.Module):
    """Least squares GAN loss"""
    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        if target_is_real:
            return 0.5 * torch.mean((prediction - 1) ** 2)
        else:
            return 0.5 * torch.mean(prediction ** 2)

class RelativistGANLoss(nn.Module):
    """Relativistic GAN loss"""
    def forward(self, real_pred: torch.Tensor, fake_pred: torch.Tensor, for_discriminator: bool = True) -> torch.Tensor:
        if for_discriminator:
            return (F.binary_cross_entropy_with_logits(real_pred - fake_pred.mean(), torch.ones_like(real_pred)) +
                    F.binary_cross_entropy_with_logits(fake_pred - real_pred.mean(), torch.zeros_like(fake_pred))) / 2
        else:
            return (F.binary_cross_entropy_with_logits(fake_pred - real_pred.mean(), torch.ones_like(fake_pred)) +
                    F.binary_cross_entropy_with_logits(real_pred - fake_pred.mean(), torch.zeros_like(real_pred))) / 2

# VAE Losses
class ELBOLoss(nn.Module):
    """Evidence Lower Bound loss for VAE"""
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # reconstruction loss
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + self.beta * kl_loss

class BetaVAELoss(ELBOLoss):
    """Beta-VAE loss with adjustable beta parameter"""
    pass

class BetaTCVAELoss(nn.Module):
    """Beta-TC-VAE loss for disentanglement"""
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta 
        self.gamma = gamma
    
    def forward(self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, 
                logvar: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # reconstruction term
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        
        # KL terms (simplified implementation)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # total correlation term (simplified)
        tc_loss = kl_loss  # in practice, this would be computed differently
        
        return recon_loss + self.alpha * kl_loss + self.beta * tc_loss

FactorVAELoss = BetaTCVAELoss  # similar structure

# Diffusion Model Losses  
class DDPMLoss(nn.Module):
    """DDPM (Denoising Diffusion Probabilistic Models) loss"""
    def forward(self, noise_pred: torch.Tensor, noise_true: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(noise_pred, noise_true)

class DDIMLoss(DDPMLoss):
    """DDIM loss (similar to DDPM)"""
    pass

class ScoreMatchingLoss(nn.Module):
    """Score matching loss for diffusion models"""
    def forward(self, score_pred: torch.Tensor, score_true: torch.Tensor) -> torch.Tensor:
        return 0.5 * F.mse_loss(score_pred, score_true)

# Metric Learning Losses
class ContrastiveMetricLoss(nn.Module):
    """Contrastive loss for metric learning"""
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, output1: torch.Tensor, output2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

class TripletMetricLoss(nn.Module):
    """Triplet loss for metric learning"""
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

class QuadrupletLoss(nn.Module):
    """Quadruplet loss extending triplet loss"""
    def __init__(self, margin1: float = 1.0, margin2: float = 0.5):
        super().__init__()
        self.margin1 = margin1
        self.margin2 = margin2
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: torch.Tensor, negative2: torch.Tensor) -> torch.Tensor:
        d_ap = F.pairwise_distance(anchor, positive)
        d_an = F.pairwise_distance(anchor, negative)
        d_nn = F.pairwise_distance(negative, negative2)
        
        loss1 = F.relu(d_ap - d_an + self.margin1)
        loss2 = F.relu(d_ap - d_nn + self.margin2)
        
        return (loss1 + loss2).mean()

class NPairLoss(nn.Module):
    """N-pair loss for metric learning"""
    def __init__(self, l2_reg: float = 0.02):
        super().__init__()
        self.l2_reg = l2_reg
    
    def forward(self, anchors: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        # simplified n-pair loss implementation
        anchor_dot_positive = torch.sum(anchors * positives, dim=1)
        anchor_dot_negative = torch.mm(anchors, negatives.t())
        
        logits = torch.cat([anchor_dot_positive.unsqueeze(1), anchor_dot_negative], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        
        # L2 regularization
        l2_loss = torch.mean(anchors ** 2) + torch.mean(positives ** 2)
        
        return loss + self.l2_reg * l2_loss

class AngularMetricLoss(nn.Module):
    """Angular loss for face recognition"""
    def __init__(self, margin: float = 0.5, scale: float = 64):
        super().__init__()
        self.margin = margin
        self.scale = scale
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # normalize features and weights
        features = F.normalize(features, p=2, dim=1)
        weight = F.normalize(weight, p=2, dim=1)
        
        # compute cosine similarity
        cosine = F.linear(features, weight)
        
        # apply angular margin
        phi = cosine - self.margin
        
        # create one-hot labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        output = one_hot * phi + (1.0 - one_hot) * cosine
        output = output * self.scale
        
        return F.cross_entropy(output, labels)

class ArcFaceMetricLoss(nn.Module):
    """ArcFace loss for face recognition"""
    def __init__(self, margin: float = 0.5, scale: float = 64):
        super().__init__()
        self.margin = margin
        self.scale = scale
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, p=2, dim=1)
        weight = F.normalize(weight, p=2, dim=1)
        
        cosine = F.linear(features, weight)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * math.cos(self.margin) - sine * math.sin(self.margin)
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale
        
        return F.cross_entropy(output, labels)

class CosFaceMetricLoss(nn.Module):
    """CosFace loss for face recognition"""
    def __init__(self, margin: float = 0.35, scale: float = 64):
        super().__init__()
        self.margin = margin
        self.scale = scale
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, p=2, dim=1)
        weight = F.normalize(weight, p=2, dim=1)
        
        cosine = F.linear(features, weight)
        phi = cosine - self.margin
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale
        
        return F.cross_entropy(output, labels)

class SphereFaceLoss(nn.Module):
    """SphereFace loss (A-Softmax)"""
    def __init__(self, margin: int = 4, scale: float = 64):
        super().__init__()
        self.margin = margin
        self.scale = scale
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, p=2, dim=1)
        weight = F.normalize(weight, p=2, dim=1)
        
        cosine = F.linear(features, weight)
        
        # compute cos(m*theta) using chebyshev polynomials (simplified for m=4)
        cos_m_theta = 8 * torch.pow(cosine, 4) - 8 * torch.pow(cosine, 2) + 1
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        output = (one_hot * cos_m_theta) + ((1.0 - one_hot) * cosine)
        output = output * self.scale
        
        return F.cross_entropy(output, labels)

class ProxyNCALoss(nn.Module):
    """Proxy-NCA loss for metric learning"""
    def __init__(self, num_classes: int, embed_dim: int, scale: float = 32):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.scale = scale
        self.proxies = nn.Parameter(torch.randn(num_classes, embed_dim))
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, p=2, dim=1)
        proxies = F.normalize(self.proxies, p=2, dim=1)
        
        # compute similarities
        similarities = torch.mm(embeddings, proxies.t()) * self.scale
        
        return F.cross_entropy(similarities, labels)

class ProxyAnchorLoss(nn.Module):
    """Proxy-Anchor loss for metric learning"""
    def __init__(self, num_classes: int, embed_dim: int, margin: float = 0.1, alpha: float = 32):
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.alpha = alpha
        self.proxies = nn.Parameter(torch.randn(num_classes, embed_dim))
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, p=2, dim=1)
        proxies = F.normalize(self.proxies, p=2, dim=1)
        
        similarities = torch.mm(embeddings, proxies.t())
        
        pos_loss = torch.tensor(0.0, device=embeddings.device)
        neg_loss = torch.tensor(0.0, device=embeddings.device)
        
        for i in range(self.num_classes):
            pos_mask = (labels == i)
            neg_mask = (labels != i)
            
            if pos_mask.sum() > 0:
                pos_exp = torch.exp(-self.alpha * (similarities[pos_mask, i] - self.margin))
                pos_loss += torch.log(1 + pos_exp.sum())
            
            if neg_mask.sum() > 0:
                neg_exp = torch.exp(self.alpha * (similarities[neg_mask, i] + self.margin))
                neg_loss += torch.log(1 + neg_exp.sum())
        
        return pos_loss + neg_loss

# Multi-task Learning Losses
class UncertaintyWeightingLoss(nn.Module):
    """Uncertainty-based weighting for multi-task learning"""
    def __init__(self, num_tasks: int):
        super().__init__()
        self.num_tasks = num_tasks
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses: list) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=self.log_vars.device)
        
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        
        return total_loss

class MultiTaskLoss(nn.Module):
    """Simple multi-task loss with fixed weights"""
    def __init__(self, weights: Optional[list] = None):
        super().__init__()
        self.weights = weights
    
    def forward(self, losses: list) -> torch.Tensor:
        if self.weights is None:
            return sum(losses)
        else:
            return sum(w * loss for w, loss in zip(self.weights, losses))

class PCGradLoss(nn.Module):
    """PCGrad-style gradient surgery for multi-task learning"""
    def __init__(self):
        super().__init__()
    
    def forward(self, losses: list) -> torch.Tensor:
        # simplified implementation - in practice, gradient surgery happens at optimization level
        return sum(losses)

class GradNormLoss(nn.Module):
    """GradNorm for balancing gradients in multi-task learning"""
    def __init__(self, num_tasks: int, alpha: float = 1.5):
        super().__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.weights = nn.Parameter(torch.ones(num_tasks))
    
    def forward(self, losses: list) -> torch.Tensor:
        weighted_losses = [w * loss for w, loss in zip(self.weights, losses)]
        return sum(weighted_losses)

class CAGradLoss(nn.Module):
    """Conflict-Averse Gradient descent for multi-task learning"""
    def __init__(self, num_tasks: int, c: float = 0.5):
        super().__init__()
        self.num_tasks = num_tasks
        self.c = c
    
    def forward(self, losses: list) -> torch.Tensor:
        # simplified implementation
        return sum(losses)

class DynamicLossBalancing(nn.Module):
    """Dynamic loss balancing based on task difficulty"""
    def __init__(self, num_tasks: int, temp: float = 2.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.temp = temp
        self.task_losses = []
    
    def forward(self, losses: list) -> torch.Tensor:
        # track loss history for dynamic balancing
        current_losses = torch.stack(losses)
        
        if len(self.task_losses) == 0:
            weights = torch.ones_like(current_losses)
        else:
            prev_losses = torch.stack(self.task_losses[-1])
            loss_ratios = current_losses / (prev_losses + 1e-8)
            weights = F.softmax(loss_ratios / self.temp, dim=0) * self.num_tasks
        
        self.task_losses.append(losses)
        if len(self.task_losses) > 10:  # keep only recent history
            self.task_losses.pop(0)
        
        return sum(w * loss for w, loss in zip(weights, losses))

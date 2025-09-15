"""
Ranking loss functions.
"""

import torch
import torch.nn as nn

# Placeholder classes
class NDCGLoss(nn.Module):
    def forward(self, x, y): return torch.tensor(0.0)

class MRRLoss(nn.Module):
    def forward(self, x, y): return torch.tensor(0.0)

class MAPLoss(nn.Module):
    def forward(self, x, y): return torch.tensor(0.0)

class RankNetLoss(nn.Module):
    def forward(self, x, y): return torch.tensor(0.0)

class LambdaRankLoss(nn.Module):
    def forward(self, x, y): return torch.tensor(0.0)

import torch.nn as nn
import torch

class TukeyLoss(nn.Module):
    def forward(self, x, y): return torch.tensor(0.0)

class CauchyLoss(nn.Module):
    def forward(self, x, y): return torch.tensor(0.0)

class WelschLoss(nn.Module):
    def forward(self, x, y): return torch.tensor(0.0)

class FairLoss(nn.Module):
    def forward(self, x, y): return torch.tensor(0.0)

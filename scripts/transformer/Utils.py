import torch
from torch import nn
from torch.nn import functional as F

class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_n_hidden, ffn_n_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_n_hidden)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_n_outputs)

    def forward(self, X: torch.Tensor):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    def __init__(self, norm_shape, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(norm_shape)

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        return self.layer_norm(self.dropout(Y) + X)
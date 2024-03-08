import torch
from torch import nn
from torch.nn import functional as F
from Attention import DotProductAttention, MultiHeadAttention
from Utils import PositionWiseFFN, AddNorm

class TransformerEncoderBlock(nn.Module):
    def __init__(self, n_hidden, ffn_n_hidden, n_heads, dropout=0.1, use_bias=False):
        super().__init__()
        self.attention = MultiHeadAttention(n_hidden, n_heads, dropout, use_bias)
        self.add_norm1 = AddNorm(n_hidden, dropout)
        self.ffn = PositionWiseFFN(ffn_n_hidden, n_hidden)
        self.add_norm2 = AddNorm(n_hidden, dropout)

    def forward(self, X: torch.Tensor, valid_lengths: torch.Tensor = None):
        sublayer1 = self.add_norm1(X, self.attention(X, X, X, valid_lengths))
        sublayer2 = self.add_norm2(sublayer1, self.ffn(sublayer1))
        return sublayer2

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, n_blocks, n_hidden, ffn_n_hidden, n_heads, dropout=0.1, use_bias=False):
        super().__init__()
        self.n_hidden = n_hidden
        self.embedding = nn.Embedding(vocab_size, n_hidden)
        self.pos_embedding = nn.Embedding(vocab_size, n_hidden)
        self.blocks = nn.Sequential()
        for i in range(n_blocks):
            self.blocks.add_module(f'block_{i}', TransformerEncoderBlock(n_hidden, ffn_n_hidden, n_heads, dropout, use_bias))
        self.attention_weights = None

    def forward(self, X: torch.Tensor, valid_lengths: torch.Tensor = None):
        seq_emb = self.embedding(X)
        pos_emb = self.pos_embedding(torch.arange(X.shape[-1]))
        X = seq_emb + pos_emb

        self.attention_weights = [None] * len(self.blocks)
        for i, block in enumerate(self.blocks):
            X = block(X, valid_lengths)
            self.attention_weights[i] = block.attention.attention.attention_weights

        return X
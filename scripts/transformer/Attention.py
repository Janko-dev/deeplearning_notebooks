import torch
from torch import nn
from torch.nn import functional as F

def masked_softmax(X: torch.Tensor, valid_lenghts: torch.Tensor) -> torch.Tensor:
    # input: 3d tensor (batch_size, key_size, query_size)
    # valid_lengths: 1d tensor (valid_lens) or 2d tensor (key_size, valid_lens)

    if valid_lenghts is None:
        return F.softmax(X, dim=-1)
    else:
        shape = X.shape
        valid_lenghts = torch.repeat_interleave(valid_lenghts, shape[1]) \
            if valid_lenghts.dim() == 1 else valid_lenghts.reshape(-1)

        X = X.reshape(-1, shape[-1])
        max_len = X.shape[1]
        mask = torch.arange(max_len, dtype=torch.float32)[None, :] < valid_lenghts[:, None]
        X[~mask] = -1e6
        return F.softmax(X.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):
    def __init__(self, num_hidden, dropout_rate):
        super().__init__()
        self.W_k = nn.LazyLinear(num_hidden, bias=False)
        self.W_q = nn.LazyLinear(num_hidden, bias=False)
        self.w_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.attention_weights = None

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, valid_lengths=None):
        # Q: (batch_size, n_queries, d)
        # K: (batch_size, n_keyvals, d)
        # Q: (batch_size, n_keyvals, value_dim)
        queries, keys = self.W_q(Q), self.W_k(K)
        # to sum,
        # queries needs to be (batch_size, n_queries, 1, d) and
        # keys needs to be (batch_size, 1, n_queries, d)
        features = queries.unsqueeze(dim=2) + keys.unsqueeze(dim=1)
        features = F.tanh(features)
        scores = self.w_v(features).squeeze(dim=-1)
        self.attention_weights = masked_softmax(scores, valid_lengths)
        return torch.bmm(self.dropout(self.attention_weights), V)

class DotProductAttention(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.attention_weights = None

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, valid_lengths=None):
        # Q: (batch_size, n_queries, d)
        # K: (batch_size, n_keyvals, d)
        # Q: (batch_size, n_keyvals, value_dim)
        d = Q.shape[-1]
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / d** .5
        self.attention_weights = masked_softmax(attention_scores, valid_lengths)
        return torch.bmm(self.dropout(self.attention_weights), V)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_hidden, n_heads, dropout=0.1, bias=False):
        super().__init__()
        self.n_heads = n_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.LazyLinear(n_hidden, bias=bias)
        self.W_k = nn.LazyLinear(n_hidden, bias=bias)
        self.W_v = nn.LazyLinear(n_hidden, bias=bias)
        self.W_o = nn.LazyLinear(n_hidden, bias=bias)

    def transpose_qkv(self, X: torch.Tensor):
        # Transpose input shape X
        # (batch_size, n_queries, n_hidden) into
        # (batch_size * num_heads, n_queries, n_hidden / n_heads)

        X = X.reshape(*X.shape[:2], self.n_heads, -1)  # (batch_size, n_queries, n_heads, n_hidden / n_heads)
        X = X.permute(0, 2, 1, 3)  # (batch_size, n_heads, n_queries, n_hidden / n_heads)
        X = X.reshape(-1, *X.shape[2:])  # (batch_size * n_heads, n_queries, n_hidden / n_heads)
        return X

    def transpose_output(self, X: torch.Tensor):
        # reverse transpose_qkv
        # (batch_size * num_heads, n_queries, n_hidden / n_heads) into
        # (batch_size, n_queries, n_hidden)

        X = X.reshape(-1, self.n_heads, *X.shape[1:])  # (batch_size, n_heads, n_queries, n_hidden / n_heads)
        X = X.permute(0, 2, 1, 3)  # (batch_size, n_queries, n_heads, n_hidden / n_heads)
        X = X.reshape(*X.shape[:2], -1)  # (batch_size, n_queries, n_heads, n_hidden / n_heads)
        return X

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, valid_lengths=None):
        # shape queries, keys, values: (batch_size, n_queries or n_keyvalue_pairs, n_hidden)
        # shape valid_lengths: (batch_size, ) or (batch_size, n_queries)

        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_q(keys))
        values = self.transpose_qkv(self.W_q(values))
        # after transposing, shape queries, keys, values:
        # (batch_size * n_heads, n_queries or n_keyvalue_pairs, n_hidden / n_heads)

        if valid_lengths is not None:
            # (batch_size, ) --> (batch_size, n_heads)
            # (batch_size, n_queries) --> (batch_size, n_heads, n_queries)
            valid_lengths = torch.repeat_interleave(valid_lengths, repeats=self.n_heads, dim=0)

        output = self.attention(queries, keys, values,
                                valid_lengths)  # (batch_size * n_heads, n_queries, n_hidden / n_heads)
        output = self.transpose_output(output)  # (batch_size, n_queries, n_hidden)
        return self.W_o(output)
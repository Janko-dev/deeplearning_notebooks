import torch
from torch import nn
from torch.nn import functional as F
from Attention import MultiHeadAttention
from Utils import PositionWiseFFN, AddNorm


class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_hidden, ffn_n_hidden, n_heads, i, dropout=0.1):
        super().__init__()
        self.i = i  # ith decoder block
        self.attention1 = MultiHeadAttention(n_hidden, n_heads, dropout)
        self.add_norm1 = AddNorm(n_hidden, dropout)

        self.attention2 = MultiHeadAttention(n_hidden, n_heads, dropout)
        self.add_norm2 = AddNorm(n_hidden, dropout)

        self.ffn = PositionWiseFFN(ffn_n_hidden, n_hidden)
        self.add_norm3 = AddNorm(n_hidden, dropout)

    def forward(self, X: torch.Tensor, state: list[torch.Tensor]):
        # state[0] encoder outputs: (batch_size, n_queries (steps), n_hidden)
        # state[1] encoder valid lengths: (batch_size, ) or (batch_size, n_queries)
        # state[2] sequence of keys/values of all steps up until this step: (batch_size, n_queries (steps), n_hidden)
        enc_outputs, enc_valid_lens = state[0], state[1]

        if state[2][self.i] is None:
            key_values = X  # no state yet
        else:
            # concatenate previous steps and X
            # shape: (batch_size, n_queries+1, n_hidden)
            key_values = torch.cat([state[2][self.i], X], dim=1)

        state[2][self.i] = key_values

        if self.training:
            batch_size, n_steps, _ = X.shape
            # shape dec_valid_lens: (batch_size, n_steps)
            dec_valid_lens = torch.arange(1, n_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.add_norm1(X, X2)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.add_norm2(Y, Y2)
        return self.add_norm3(Z, self.ffn(Z)), state


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, n_blocks, n_hidden, ffn_n_hidden, n_heads, dropout=0.1):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_blocks = n_blocks
        self.embedding = nn.Embedding(vocab_size, n_hidden)
        self.pos_embedding = nn.Embedding(vocab_size, n_hidden)
        self.blocks = nn.Sequential()
        for i in range(n_blocks):
            self.blocks.add_module(f'block_{i}', TransformerDecoderBlock(n_hidden, ffn_n_hidden, n_heads, i, dropout))
        self.dense = nn.LazyLinear(vocab_size)
        self.attention_weights = None

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.n_blocks]

    def forward(self, X: torch.Tensor, state: list[torch.Tensor]):
        seq_emb = self.embedding(X)
        pos_emb = self.pos_embedding(torch.arange(X.shape[-1]))
        X = seq_emb + pos_emb

        self.attention_weights = [[None] * len(self.blocks) for _ in range(2)]
        for i, block in enumerate(self.blocks):
            X, state = block(X, state)

            # self attention
            self.attention_weights[0][i] = block.attention1.attention.attention_weights
            # cross attention
            self.attention_weights[1][i] = block.attention1.attention.attention_weights

        return self.dense(X), state
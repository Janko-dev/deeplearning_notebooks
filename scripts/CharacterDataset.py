import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset

class VocabDataset(Dataset):
    def __init__(self, file_patch: str, seq_len: int):
        f = open(file_patch, 'r')
        corpus = f.read()
        self.seq_len = seq_len
        self.vocab = sorted(list(set(corpus)))
        self.vocab_size = len(self.vocab)
        self.stoi = dict(zip(self.vocab, range(1, self.vocab_size + 1)))

        self.encoder = np.vectorize(lambda c: self.stoi[c])
        self.decoder = np.vectorize(lambda x: self.vocab[x])
        self.data = self.encoder(list(corpus))

    def __getitem__(self, index):
        return self.data[index:index + self.seq_len + 1]

    def __len__(self):
        return len(self.data) - self.seq_len

def preprocess_batch(batch: torch.Tensor):
    # batch (B, C)
    x = batch.reshape(batch.shape[0], 1, batch.shape[1])
    x = x.repeat_interleave(batch.shape[1]-1, dim=1)
    x = x[:, :, :-1]
    tril = torch.tril(torch.ones_like(x))
    x = x*tril
    # output: (B, T, C), (B, T+1)
    return x, batch[:, 1:]


ds = VocabDataset("../datasets/shakespeare.txt", seq_len=8)

batch = np.array([ds[i] for i in range(4)])
# x, y = preprocess_batch(torch.tensor(batch))

dl = DataLoader(ds, batch_size=16)
emb = nn.Embedding(ds.vocab_size, 2)

for i, batch in enumerate(dl):
    x, y = preprocess_batch(batch)

    print(x.shape, y.shape)
    print(emb(x).shape, emb(y).shape)

    y_pred = emb(x)
    y = emb(y)
    loss = F.cross_entropy(y_pred, y)
    print(loss.item())
    if i == 10:
        break
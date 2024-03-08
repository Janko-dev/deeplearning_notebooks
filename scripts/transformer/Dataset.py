from collections import Counter
from torch.utils.data import IterableDataset, DataLoader
import torch
from torch import nn

class Vocab:

    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']


class SeqFrenchEnglish(IterableDataset):
    FILE_PATH = '../../datasets/fra.txt'

    def __init__(self, batch_size, num_steps=9, num_train=512, num_val=128):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_train = num_train
        self.num_val = num_val
        self.arrays, self.src_vocab, self.tgt_vocab = self.build_arrays(self.import_dataset(self.FILE_PATH))

    @staticmethod
    def import_dataset(file_name):
        f = open(file_name, 'r', encoding='utf-8')
        text = f.read()
        text.replace('\u202f', ' ').replace('\xa0', ' ')  # non-breaking space --> space
        no_space = lambda char, prev_char: char in ',.!?' and prev_char != ' '
        out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
               for i, char in enumerate(text)]
        return ''.join(out)

    def build_arrays(self, text, src_vocab=None, tgt_vocab=None):
        def _build_array(sentences, vocab, is_tgt=False):
            pad_or_trim = lambda seq, t: (
                seq[:t] if len(seq) > t else seq + ['<pad>'] * (t - len(seq)))
            sentences = [pad_or_trim(s, self.num_steps) for s in sentences]
            if is_tgt:
                sentences = [['<bos>'] + s for s in sentences]
            if vocab is None:
                vocab = Vocab(sentences, min_freq=2)
            array = torch.tensor([vocab[s] for s in sentences])
            valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
            return array, vocab, valid_len

        src, tgt = self.tokenize(text, self.num_train + self.num_val)
        src_array, src_vocab, src_valid_len = _build_array(src, src_vocab)
        tgt_array, tgt_vocab, _ = _build_array(tgt, tgt_vocab, True)
        return ((src_array, tgt_array[:, :-1], src_valid_len, tgt_array[:, 1:]),
                src_vocab, tgt_vocab)

    @staticmethod
    def tokenize(text, max_examples=None):
        src, tgt = [], []
        for i, line, in enumerate(text.split('\n')):
            if max_examples and i > max_examples: break
            parts = line.split('\t')
            if len(parts) == 2:
                src.append([t for t in f'{parts[0]} <eos>'.split(' ') if t])
                tgt.append([t for t in f'{parts[1]} <eos>'.split(' ') if t])

        return src, tgt

    def __getitem__(self, index):
        return [self.arrays[0][index], self.arrays[1][index], self.arrays[2][index], self.arrays[3][index]]

    def __len__(self):
        return self.num_train + self.num_val

    def get_dataloader(self):
        return DataLoader(self, self.batch_size)

    def __iter__(self):
        return iter(zip(*self.arrays))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from Decoder import TransformerDecoder
from Encoder import TransformerEncoder
from Dataset import SeqFrenchEnglish

class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder, tgt_pad=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_pad = tgt_pad

    def forward(self, enc_X, dec_X):
        enc_all_outputs = self.encoder(enc_X)
        dec_state = self.decoder.init_state(enc_all_outputs, None)
        # Return decoder output only
        return self.decoder(dec_X, dec_state)[0]

    def predict_step(self, batch, device, num_steps, save_attention_weights=False):

        batch = [a.to(device) for a in batch]
        src, tgt, src_valid_len, _ = batch
        enc_all_outputs = self.encoder(src, src_valid_len)
        dec_state = self.decoder.init_state(enc_all_outputs, src_valid_len)
        outputs, attention_weights = [torch.unsqueeze(tgt[:, 0], 1), ], []
        for _ in range(num_steps):
            Y, dec_state = self.decoder(outputs[-1], dec_state)
            outputs.append(torch.argmax(Y, dim=2))
            # Save attention weights
            if save_attention_weights:
                attention_weights.append(self.decoder.attention_weights)
        return torch.concat(outputs[1:], 1), attention_weights

if __name__ == '__main__':
    data = SeqFrenchEnglish(batch_size=128)
    data_loader = data.get_dataloader()

    n_hidden, n_blocks, dropout = 256, 2, 0.2
    ffn_num_hidden, n_heads = 64, 4
    EPOCH = 30

    encoder = TransformerEncoder(len(data.src_vocab), n_blocks, n_hidden, ffn_num_hidden, n_heads, dropout)
    decoder = TransformerDecoder(len(data.tgt_vocab), n_blocks, n_hidden, ffn_num_hidden, n_heads, dropout)
    model = EncoderDecoder(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'])

    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []

    for epoch in range(EPOCH):
        batch_loss = []
        for i, batch in enumerate(data_loader):
            src, tgt, pad, label = batch
            y_pred = model(src, tgt)
            y_pred = y_pred.transpose(1, 2)
            loss = F.cross_entropy(y_pred, label)
            batch_loss.append(loss.item())

            optim.zero_grad()
            loss.backward()
            optim.step()

        history.append(np.mean(batch_loss))
        print(f"{epoch=}, batch loss={history[-1]}")

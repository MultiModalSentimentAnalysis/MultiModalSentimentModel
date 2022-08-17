import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class SimpleDenseNetwork(nn.Module):
    def __init__(self, n_classes, embedding_dimension):
        super(SimpleDenseNetwork, self).__init__()

        self.n_classes = n_classes
        self.embedding_dimension = embedding_dimension

        self.fc = nn.Sequential(
            nn.Linear(
                in_features=self.embedding_dimension,
                out_features=512,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=3),
            nn.ReLU(inplace=True),
            nn.Softmax(),
        )

    def forward(self, input_batch):
        x = input_batch
        x = self.fc(x)
        output_batch = x

        return output_batch


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, n_class, n_inp, n_head=2, n_hid=192, n_layers=2, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(n_inp, dropout)
        encoder_layers = TransformerEncoderLayer(n_inp, n_head, n_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.n_inp = n_inp
        self.decoder = nn.Linear(n_inp, n_class)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src *= math.sqrt(self.n_inp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)

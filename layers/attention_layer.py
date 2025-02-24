import torch
import torch.nn as nn
import math
import csv
import numpy as np
from data.BrainNet import name2coor_path


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #  x = [batch size, seq len, hid dim]
        x = self.dropout(torch.relu(self.fc_1(x)))
        #  x = [batch size, seq len, pf dim]
        x = self.fc_2(x)
        #  x = [batch size, seq len, hid dim]

        return x


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class IdentitylEncoding(nn.Module):
    "Implement the IdentitylEncoding function."
    def __init__(self, d_model, node_num, dropout):
        super(IdentitylEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        self.node_identity = nn.Parameter(torch.zeros(node_num, d_model), requires_grad=True)
        self.mlp = nn.Linear(d_model, d_model)

        nn.init.kaiming_normal_(self.node_identity)

    def forward(self, x):
        bz, _, _, = x.shape
        pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
        emb = x + pos_emb
        x = x + self.mlp(emb)
        return self.dropout(x)


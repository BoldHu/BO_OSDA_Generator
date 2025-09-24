import argparse
import math
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

PAD = 0
UNK = 1
EOS = 2
SOS = 3
MASK = 4

class PositionalEncoding(nn.Module):
    "Implement the PE function. No batch support?"
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model) # (T,H)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (T, B, H)
        # self.pe: (1, max_len, d_model)
        # self.pe[:, :T, :]: (1, T, d_model)
        # permute(1, 0, 2): (T, 1, d_model)
        pos_enc = self.pe[:, :x.size(0), :].permute(1, 0, 2)  # => (T, 1, H)
        x = x + pos_enc
        return self.dropout(x)

class TrfmSeq2seq(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, n_layers, dropout=0.1):
        super(TrfmSeq2seq, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(in_size, hidden_size)
        self.pe = PositionalEncoding(hidden_size, dropout)
        self.trfm = nn.Transformer(d_model=hidden_size, nhead=4, 
        num_encoder_layers=n_layers, num_decoder_layers=n_layers, dim_feedforward=hidden_size)
        self.out = nn.Linear(hidden_size, out_size)

    def forward(self, src):
        # src: (T,B)
        embedded = self.embed(src)  # (T,B,H)
        embedded = self.pe(embedded) # (T,B,H)
        hidden = self.trfm(embedded, embedded) # (T,B,H)
        out = self.out(hidden) # (T,B,V)
        out = F.log_softmax(out, dim=2) # (T,B,V)
        return out # (T,B,V)

    def _encode(self, src):
        # src: (T,B)
        embedded = self.embed(src)  # (T,B,H)
        embedded = self.pe(embedded) # (T,B,H)
        output = embedded
        for i in range(self.trfm.encoder.num_layers - 1):
            output = self.trfm.encoder.layers[i](output, src_mask=None)  # (T,B,H)
        penul = output.detach()  # (T,B,H), 保留为张量
        output = self.trfm.encoder.layers[-1](output, src_mask=None)  # (T,B,H)
        if self.trfm.encoder.norm:
            output = self.trfm.encoder.norm(output) # (T,B,H)
        # mean, max, first*2
        output = output.detach()
        mean = output.mean(dim=0)  # (B, H)
        max_ = output.max(dim=0).values  # (B, H)
        first = output[0, :, :]  # (B, H)
        penul_first = penul[0, :, :]  # (B, H)
        return torch.cat([mean, max_, first, penul_first], dim=1)  # (B, 4H)
    
    def _encode_mean(self, src):
        # src: (T,B)
        embedded = self.embed(src)  # (T,B,H)
        embedded = self.pe(embedded) # (T,B,H)
        output = embedded
        for i in range(self.trfm.encoder.num_layers - 1):
            output = self.trfm.encoder.layers[i](output, src_mask=None)  # (T,B,H)
        penul = output.detach()  # (T,B,H), 保留为张量
        output = self.trfm.encoder.layers[-1](output, src_mask=None)  # (T,B,H)
        if self.trfm.encoder.norm:
            output = self.trfm.encoder.norm(output) # (T,B,H)
        # mean, max, first*2
        output = output.detach()
        mean = output.mean(dim=0)  # (B, H)
        max_ = output.max(dim=0).values  # (B, H)
        first = output[0, :, :]  # (B, H)
        penul_first = penul[0, :, :]  # (B, H)
        return torch.cat([mean], dim=1)  # (B, 4H)
    
    def _encode_max(self, src):
        # src: (T,B)
        embedded = self.embed(src)  # (T,B,H)
        embedded = self.pe(embedded) # (T,B,H)
        output = embedded
        for i in range(self.trfm.encoder.num_layers - 1):
            output = self.trfm.encoder.layers[i](output, src_mask=None)  # (T,B,H)
        penul = output.detach()  # (T,B,H), 保留为张量
        output = self.trfm.encoder.layers[-1](output, src_mask=None)  # (T,B,H)
        if self.trfm.encoder.norm:
            output = self.trfm.encoder.norm(output) # (T,B,H)
        # mean, max, first*2
        output = output.detach()
        mean = output.mean(dim=0)  # (B, H)
        max_ = output.max(dim=0).values  # (B, H)
        first = output[0, :, :]  # (B, H)
        penul_first = penul[0, :, :]  # (B, H)
        return torch.cat([max_], dim=1)  # (B, 4H)
    
    def _encode_first(self, src):
        # src: (T,B)
        embedded = self.embed(src)  # (T,B,H)
        embedded = self.pe(embedded) # (T,B,H)
        output = embedded
        for i in range(self.trfm.encoder.num_layers - 1):
            output = self.trfm.encoder.layers[i](output, src_mask=None)  # (T,B,H)
        penul = output.detach()  # (T,B,H), 保留为张量
        output = self.trfm.encoder.layers[-1](output, src_mask=None)  # (T,B,H)
        if self.trfm.encoder.norm:
            output = self.trfm.encoder.norm(output) # (T,B,H)
        # mean, max, first*2
        output = output.detach()
        mean = output.mean(dim=0)  # (B, H)
        max_ = output.max(dim=0).values  # (B, H)
        first = output[0, :, :]  # (B, H)
        penul_first = penul[0, :, :]  # (B, H)
        return torch.cat([first], dim=1)  # (B, 4H)
    
    def _encode_penul_first(self, src):
        # src: (T,B)
        embedded = self.embed(src)  # (T,B,H)
        embedded = self.pe(embedded) # (T,B,H)
        output = embedded
        for i in range(self.trfm.encoder.num_layers - 1):
            output = self.trfm.encoder.layers[i](output, src_mask=None)  # (T,B,H)
        penul = output.detach()  # (T,B,H), 保留为张量
        output = self.trfm.encoder.layers[-1](output, src_mask=None)  # (T,B,H)
        if self.trfm.encoder.norm:
            output = self.trfm.encoder.norm(output) # (T,B,H)
        # mean, max, first*2
        output = output.detach()
        mean = output.mean(dim=0)  # (B, H)
        max_ = output.max(dim=0).values  # (B, H)
        first = output[0, :, :]  # (B, H)
        penul_first = penul[0, :, :]  # (B, H)
        return torch.cat([penul_first], dim=1)  # (B, 4H)
        
    
    
    def encode(self, src):
        # src: (T,B)
        batch_size = src.size(1)
        device = src.device
        return self._encode(src)
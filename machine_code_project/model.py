# -*- coding: utf-8 -*-
import math
import string
import torch
import torch.nn as nn
import torch.nn.functional as F

# Глобальные константы для токенизации
CHAR_VOCAB = list(string.printable)
base_vocab_size = len(CHAR_VOCAB)
PAD_TOKEN, SOS_TOKEN, EOS_TOKEN = "<PAD>", "<SOS>", "<EOS>"
PAD_IDX, SOS_IDX, EOS_IDX = base_vocab_size, base_vocab_size + 1, base_vocab_size + 2
VOCAB_SIZE = base_vocab_size + 3

# Словарь для преобразования символов в индексы
char_to_idx = {c: i for i, c in enumerate(CHAR_VOCAB)}

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_size, nhead, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.embed_size = embed_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=nhead,
                                                    dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_hidden = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
        
    def _init_weights(self):
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_hidden.weight.data.uniform_(-initrange, initrange)
        self.fc_hidden.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        
    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    
    def forward(self, src):
        x = self.token_embedding(src) * math.sqrt(self.embed_size)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        mask = self.generate_square_subsequent_mask(x.size(0)).to(x.device)
        x = self.transformer_encoder(x, mask)
        x = x.transpose(0, 1)
        hidden = F.relu(self.fc_hidden(x))
        return self.fc_out(hidden)

def generate_text(model, prompt, max_generate, device, temperature=1.0):
    model.eval()
    tokens = [char_to_idx.get(c, 0) for c in prompt]
    tokens = [SOS_IDX] + tokens
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    generated = tokens.copy()
    for _ in range(max_generate):
        with torch.no_grad():
            logits = model(input_ids)
        next_token = torch.multinomial(torch.softmax(logits[0, -1, :]/temperature, dim=0), 1).item()
        if next_token == EOS_IDX:
            break
        generated.append(next_token)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1)
    return ''.join(CHAR_VOCAB[token] if token < base_vocab_size else 
                   (PAD_TOKEN if token==PAD_IDX else (SOS_TOKEN if token==SOS_IDX else EOS_TOKEN))
                   for token in generated[1:])

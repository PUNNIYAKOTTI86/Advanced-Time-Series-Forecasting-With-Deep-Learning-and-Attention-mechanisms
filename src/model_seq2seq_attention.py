# src/model_seq2seq_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.Wq = nn.Linear(hidden_size, hidden_size)
        self.Wk = nn.Linear(hidden_size, hidden_size)
        self.Wv = nn.Linear(hidden_size, hidden_size)

        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, query, keys, values):
        """
        query: (B, 1, H)
        keys: (B, L, H)
        values: (B, L, H)
        """
        B = query.size(0)

        Q = self.Wq(query)
        K = self.Wk(keys)
        V = self.Wv(values)

        def split_heads(x):
            return x.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, 1, self.hidden_size)

        out = self.out_proj(context)

        return out, attn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, x):
        outputs, (h, c) = self.lstm(x)
        return outputs, (h, c)


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, horizon, num_layers=1, dropout=0.2, num_heads=4):
        super().__init__()

        self.horizon = horizon
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.attn = MultiHeadAttention(hidden_size, num_heads=num_heads)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, hidden, decoder_input):
        """
        encoder_outputs: (B, L, H)
        decoder_input: (B, 1, input_size)
        """
        outputs = []
        attention_maps = []

        h, c = hidden

        for _ in range(self.horizon):
            dec_out, (h, c) = self.lstm(decoder_input, (h, c))
            context, attn = self.attn(dec_out, encoder_outputs, encoder_outputs)

            pred = self.fc(context)

            outputs.append(pred)
            attention_maps.append(attn)

            decoder_input = pred

        outputs = torch.cat(outputs, dim=1)
        attention_maps = torch.cat(attention_maps, dim=2)

        return outputs, attention_maps


class Seq2SeqAttention(nn.Module):
    def __init__(self, input_size, hidden_size, horizon, num_layers=1, dropout=0.2, num_heads=4):
        super().__init__()

        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(input_size, hidden_size, input_size, horizon, num_layers, dropout, num_heads)

    def forward(self, x):
        encoder_outputs, hidden = self.encoder(x)

        decoder_input = x[:, -1:, :]
        preds, attn = self.decoder(encoder_outputs, hidden, decoder_input)

        return preds, attn

# src/baseline_lstm.py

import torch
import torch.nn as nn

class BaselineLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, horizon):
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        preds = []

        cur = last
        for _ in range(self.horizon):
            step = self.fc(cur).unsqueeze(1)
            preds.append(step)
            cur = out[:, -1, :]

        return torch.cat(preds, dim=1)

# src/evaluate.py

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(model, test_loader, device):
    model.eval()

    preds_all = []
    y_all = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds, _ = model(x)

            preds_all.append(preds.cpu().numpy())
            y_all.append(y.cpu().numpy())

    preds_all = np.concatenate(preds_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    rmse = np.sqrt(mean_squared_error(y_all.reshape(-1), preds_all.reshape(-1)))
    mae = mean_absolute_error(y_all.reshape(-1), preds_all.reshape(-1))

    return rmse, mae

# src/train.py

import torch
from tqdm import tqdm
import numpy as np

def train_model(model, train_loader, val_loader, epochs, lr, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    best_val_loss = float("inf")
    best_path = "outputs/models/best_seq2seq_attention.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} - Training"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            preds, _ = model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} - Validation"):
                x, y = x.to(device), y.to(device)
                preds, _ = model(x)
                loss = loss_fn(preds, y)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)

        print(f"\nEpoch {epoch}: Train Loss={avg_train:.4f} | Val Loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), best_path)
            print(f"✅ Best model saved: {best_path}")

    return best_path

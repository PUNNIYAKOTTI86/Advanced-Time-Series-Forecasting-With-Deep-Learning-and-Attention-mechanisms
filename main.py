#%pip install torch
#%pip install numpy
#%pip install pandas matplotlib scikit-learn
#%pip install tqdm

# main.py

import torch
from torch.utils.data import DataLoader

from src.config import CONFIG
from src.utils import set_seed, make_dirs
from src.dataset import (
    generate_multivariate_series,
    train_val_test_split,
    scale_data,
    TimeSeriesWindowDataset
)

from src.model_seq2seq_attention import Seq2SeqAttention
from src.train import train_model
from src.evaluate import evaluate_model
from src.visualize_attention import visualize_attention

from src.baseline_lstm import BaselineLSTM
import numpy as np

def run():
    make_dirs()
    set_seed(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
import numpy as np
import pandas as pd

def generate_multivariate_series(n_timesteps=1500, n_features=5, seed=42):
    np.random.seed(seed)

    t = np.arange(n_timesteps)

    base_trend = 0.005 * t
    season1 = 2 * np.sin(2 * np.pi * t / 50)
    season2 = 1.2 * np.sin(2 * np.pi * t / 120)

    data = []
    for i in range(n_features):
        noise = np.random.normal(0, 0.5 + i * 0.05, size=n_timesteps)

        series = (
            (1 + i * 0.1) * season1 +
            (0.8 - i * 0.05) * season2 +
            base_trend +
            noise
        )

        if i > 0:
            series += 0.3 * np.roll(data[i-1], 2)

        data.append(series)

    df = pd.DataFrame(np.array(data).T, columns=[f"feature_{i+1}" for i in range(n_features)])
    return df

df = generate_multivariate_series()
df.to_csv("multivariate_series.csv", index=False)

print("✅ CSV saved as multivariate_series.csv")
df.head()

import os

os.makedirs("data/raw", exist_ok=True)

df = generate_multivariate_series(CONFIG["n_timesteps"], CONFIG["n_features"], CONFIG["seed"])
df.to_csv("data/raw/multivariate_series.csv", index=False)

print("✅ Saved: data/raw/multivariate_series.csv")

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

    # 1) Generate Dataset
df = generate_multivariate_series(CONFIG["n_timesteps"], CONFIG["n_features"], CONFIG["seed"])
df.to_csv("data/raw/multivariate_series.csv", index=False)

    # 2) Split
train_df, val_df, test_df = train_val_test_split(df, CONFIG["train_ratio"], CONFIG["val_ratio"])

    # 3) Scale
train_scaled, val_scaled, test_scaled, scaler = scale_data(train_df, val_df, test_df)

    # 4) Window datasets
train_ds = TimeSeriesWindowDataset(train_scaled, CONFIG["lookback"], CONFIG["horizon"])
val_ds = TimeSeriesWindowDataset(val_scaled, CONFIG["lookback"], CONFIG["horizon"])
test_ds = TimeSeriesWindowDataset(test_scaled, CONFIG["lookback"], CONFIG["horizon"])

train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False)
test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False)

    # 5) Train Attention Seq2Seq
model = Seq2SeqAttention(
    input_size=CONFIG["n_features"],
        hidden_size=CONFIG["hidden_size"],
        horizon=CONFIG["horizon"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
        num_heads=4
    ).to(device)

print("\nTraining Seq2Seq + Attention...")
best_path = train_model(model, train_loader, val_loader, CONFIG["epochs"], CONFIG["lr"], device)

model.load_state_dict(torch.load(best_path, map_location=device))
rmse_attn, mae_attn = evaluate_model(model, test_loader, device)
print("\n✅ Attention Model Results:")
print("RMSE:", rmse_attn)
print("MAE :", mae_attn)

    # 6) Baseline LSTM
baseline = BaselineLSTM(CONFIG["n_features"], CONFIG["hidden_size"], CONFIG["horizon"]).to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(baseline.parameters(), lr=CONFIG["lr"])

print("\nTraining Baseline LSTM...")
for epoch in range(10):
        baseline.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = baseline(x)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1}/10 Loss={np.mean(losses):.4f}")

    # Evaluate baseline
baseline.eval()
preds_all = []
y_all = []
with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = baseline(x)
            preds_all.append(preds.cpu().numpy())
            y_all.append(y.cpu().numpy())

preds_all = np.concatenate(preds_all, axis=0)
y_all = np.concatenate(y_all, axis=0)

from sklearn.metrics import mean_squared_error, mean_absolute_error
rmse_base = np.sqrt(mean_squared_error(y_all.reshape(-1), preds_all.reshape(-1)))
mae_base = mean_absolute_error(y_all.reshape(-1), preds_all.reshape(-1))

print("\n✅ Baseline LSTM Results:")
print("RMSE:", rmse_base)
print("MAE :", mae_base)

    # Save metrics
with open("outputs/metrics/results.txt", "w") as f:
        f.write("Seq2Seq + Attention\n")
        f.write(f"RMSE: {rmse_attn}\n")
        f.write(f"MAE : {mae_attn}\n\n")

        f.write("Baseline LSTM\n")
        f.write(f"RMSE: {rmse_base}\n")
        f.write(f"MAE : {mae_base}\n")

print("\n✅ Metrics saved: outputs/metrics/results.txt")

    # 7) Visualize Attention
sample_x, _ = test_ds[0]
visualize_attention(model, sample_x, device)

if __name__ == "__main__":
    run()

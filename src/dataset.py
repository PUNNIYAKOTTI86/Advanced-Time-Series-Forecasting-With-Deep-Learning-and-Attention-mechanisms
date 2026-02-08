# src/dataset.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

def generate_multivariate_series(n_timesteps=1500, n_features=5, seed=42):
    np.random.seed(seed)

    t = np.arange(n_timesteps)

    base_trend = 0.005 * t
    season1 = 2 * np.sin(2 * np.pi * t / 50)
    season2 = 1.2 * np.sin(2 * np.pi * t / 120)

    data = []
    for i in range(n_features):
        noise = np.random.normal(0, 0.5 + i * 0.05, size=n_timesteps)

        # correlated signals
        series = (
            (1 + i * 0.1) * season1 +
            (0.8 - i * 0.05) * season2 +
            base_trend +
            noise
        )

        # add a lagged correlation between features
        if i > 0:
            series += 0.3 * np.roll(data[i-1], 2)

        data.append(series)

    df = pd.DataFrame(np.array(data).T, columns=[f"feature_{i+1}" for i in range(n_features)])
    return df


class TimeSeriesWindowDataset(Dataset):
    def __init__(self, data, lookback=30, horizon=5):
        """
        data: numpy array shape (T, features)
        """
        self.data = data
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.lookback - self.horizon

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.lookback]
        y = self.data[idx+self.lookback:idx+self.lookback+self.horizon]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def train_val_test_split(df, train_ratio=0.7, val_ratio=0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df


def scale_data(train_df, val_df, test_df):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df.values)
    val_scaled = scaler.transform(val_df.values)
    test_scaled = scaler.transform(test_df.values)

    return train_scaled, val_scaled, test_scaled, scaler

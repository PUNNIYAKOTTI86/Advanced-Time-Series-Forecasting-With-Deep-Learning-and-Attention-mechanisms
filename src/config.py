# src/config.py

CONFIG = {
    "seed": 42,

    # Dataset
    "n_timesteps": 1500,
    "n_features": 5,
    "train_ratio": 0.7,
    "val_ratio": 0.15,

    # Windowing
    "lookback": 30,
    "horizon": 5,

    # Training
    "batch_size": 64,
    "epochs": 25,
    "lr": 0.001,

    # Model
    "hidden_size": 64,
    "num_layers": 1,
    "dropout": 0.2
}

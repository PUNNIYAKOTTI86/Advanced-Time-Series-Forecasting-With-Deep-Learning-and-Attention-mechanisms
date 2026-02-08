# src/utils.py

import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_dirs():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)

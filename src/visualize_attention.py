# src/visualize_attention.py

import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_attention(model, sample_x, device, save_path="outputs/plots/attention_heatmap.png"):
    model.eval()
    sample_x = sample_x.unsqueeze(0).to(device)

    with torch.no_grad():
        _, attn = model(sample_x)

    # attn shape: (B, heads, horizon, lookback)
    attn = attn.squeeze(0).mean(dim=0).cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.imshow(attn, aspect="auto")
    plt.colorbar()
    plt.title("Attention Heatmap (Avg Heads)")
    plt.xlabel("Lookback Steps")
    plt.ylabel("Forecast Step")
    plt.savefig(save_path)
    plt.show()

    print(f"✅ Saved attention plot: {save_path}")

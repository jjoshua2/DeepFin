#!/usr/bin/env python3
"""Analyze winrate calibration from shard data by running inference with current model."""

import argparse
import numpy as np
import torch
import zarr

from chess_anti_engine.uci.model_loader import load_model_from_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=str, required=True, help="Path to .zarr shard")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on")
    args = parser.parse_args()

    print(f"Loading shard: {args.shard}")
    z = zarr.open(args.shard)
    x = z["x"][:]
    wdl_target = z["wdl_target"][:]

    print(f"Shard has {len(x)} positions")
    print(f"WDL target distribution: {np.bincount(wdl_target)}")

    print(f"\nLoading model from: {args.checkpoint}")
    device = torch.device(args.device)
    model = load_model_from_checkpoint(args.checkpoint, device=str(device))

    print("Running inference...")
    with torch.no_grad():
        x_tensor = torch.from_numpy(x).to(device)
        out = model(x_tensor)
        wdl_logits = out["wdl"]
        wdl_probs = torch.softmax(wdl_logits, dim=-1).cpu().numpy()

    # Analyze confidence distribution
    confidences = wdl_probs.max(axis=1)
    predictions = wdl_probs.argmax(axis=1)
    correct = (predictions == wdl_target).astype(np.int32)

    print("\n=== Confidence Distribution ===")
    bins = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0]
    for i in range(len(bins) - 1):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        n = mask.sum()
        if n > 0:
            acc = correct[mask].mean()
            avg_conf = confidences[mask].mean()
            print(f"[{bins[i]:.0%}-{bins[i+1]:.0%}): n={n:4d}  acc={acc:.3f}  avg_conf={avg_conf:.3f}")

    # Extreme confidence ranges
    print("\n=== Extreme Confidence (>95% / <5%) ===")
    high_conf_mask = confidences >= 0.95
    low_conf_mask = confidences < 0.05

    if high_conf_mask.sum() > 0:
        high_acc = correct[high_conf_mask].mean()
        high_n = high_conf_mask.sum()
        print(f">95% confidence: n={high_n}  accuracy={high_acc:.3f}")

    if low_conf_mask.sum() > 0:
        low_acc = correct[low_conf_mask].mean()
        low_n = low_conf_mask.sum()
        print(f"<5% confidence:  n={low_n}  accuracy={low_acc:.3f}")

    # Compute Brier score
    one_hot = np.zeros_like(wdl_probs)
    one_hot[np.arange(len(wdl_target)), wdl_target] = 1.0
    brier = ((wdl_probs - one_hot) ** 2).sum(axis=1).mean()
    print(f"\nBrier score: {brier:.4f}")


if __name__ == "__main__":
    main()

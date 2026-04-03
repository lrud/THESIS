#!/usr/bin/env python3
"""
Train all 4 LSTM models on ai-server with 512x3 architecture and 72h normalization.

Usage:
    CUDA_VISIBLE_DEVICES=1 /root/thesis/.venv/bin/python scripts/train_server_512x3_72h.py

Models: market_lags, jump_aware  market_jumps  market
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = "/root/thesis"
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "scripts", "modeling"))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from model import LSTM_DVOL, count_parameters
from data_loader_unified import (
    create_unified_dataloaders,
    FEATURE_SETS,
)

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "bitcoin_lstm_features_v1.6_final.csv"
)
RESULTS_DIR = os.path.join(BASE_DIR, "results", "server_training")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "results", "logs")

CONFIG = {
    "hidden_size": 512,
    "num_layers": 3,
    "dropout": 0.4,
    "learning_rate": 0.0001,
    "batch_size": 32,
    "epochs": 100,
    "patience": 15,
    "sequence_length": 24,
    "window_size": 72,
    "use_multi_gpu": False,
}

MODELS_TO_TRAIN = [
    ("market_lags", False),
    ("jump_aware", True),
    ("market_jumps", False),
    ("market", False),
]


def weighted_mse_loss(predictions, targets, weights):
    mse = (predictions - targets) ** 2
    return (mse * weights).mean()


def calculate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    actual_dir = np.sign(y_true[1:] - y_true[:-1])
    predicted_dir = np.sign(y_pred[:-1] - y_true[:-1])
    valid = actual_dir != 0
    dir_acc = (
        (actual_dir[valid] == predicted_dir[valid]).sum() / valid.sum() * 100
        if valid.sum() > 0
        else 50.0
    )

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE": float(mape),
        "R2": float(r2),
        "Direction_%": float(dir_acc),
    }


def evaluate_model(model, test_loader, test_dataset, use_weighting, device):
    model.eval()
    all_preds, all_targets, all_weights, all_stats = [], [], [], []

    with torch.no_grad():
        for X_batch, y_batch, w_batch, stats_batch in test_loader:
            X_batch = X_batch.to(device)
            predictions = model(X_batch)
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(y_batch.numpy())
            all_weights.append(w_batch.numpy())
            all_stats.append(stats_batch.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    weights = np.concatenate(all_weights)
    stats = np.concatenate(all_stats)

    preds_orig = test_dataset.inverse_transform_target(preds, stats)
    targets_orig = test_dataset.inverse_transform_target(targets, stats)

    is_jump = weights.flatten() > 1.0

    overall = calculate_metrics(targets_orig.flatten(), preds_orig.flatten())
    results = {
        "overall": overall,
        "jump_samples": int(is_jump.sum()),
        "normal_samples": int((~is_jump).sum()),
    }

    if is_jump.sum() > 0 and (~is_jump).sum() > 0:
        results["normal"] = calculate_metrics(
            targets_orig.flatten()[~is_jump], preds_orig.flatten()[~is_jump]
        )
        results["jump"] = calculate_metrics(
            targets_orig.flatten()[is_jump], preds_orig.flatten()[is_jump]
        )

    return results


def train_one_model(model_type, use_weighting, config, device):
    input_size = len(FEATURE_SETS[model_type])
    print(f"\n{'=' * 80}")
    print(f"TRAINING: {model_type.upper()} ({input_size} features)")
    print(f"{'=' * 80}")

    model = LSTM_DVOL(
        input_size=input_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = (
        create_unified_dataloaders(
            data_path=DATA_PATH,
            feature_set=model_type,
            sequence_length=config["sequence_length"],
            window_size=config["window_size"],
            batch_size=config["batch_size"],
        )
    )
    print(
        f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} sequences"
    )

    model.device = device
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "learning_rate": []}
    model_path = os.path.join(MODELS_DIR, f"server_{model_type}_512x3_72h_best.pth")

    start_time = time.time()

    for epoch in range(config["epochs"]):
        model.train()
        train_losses = []
        for X_batch, y_batch, w_batch, _ in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            if use_weighting:
                w_batch = w_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)

            if use_weighting:
                loss = weighted_mse_loss(predictions, y_batch, w_batch)
            else:
                loss = nn.MSELoss()(predictions, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch, w_batch, _ in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                if use_weighting:
                    w_batch = w_batch.to(device)
                predictions = model(X_batch)
                if use_weighting:
                    loss = weighted_mse_loss(predictions, y_batch, w_batch)
                else:
                    loss = nn.MSELoss()(predictions, y_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 2 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            eta = (config["epochs"] - epoch - 1) * (elapsed / (epoch + 1))
            msg = (
                f"{datetime.now().isoformat()} | {model_type} | "
                f"Epoch {epoch + 1:3d}/{config['epochs']} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"Patience: {patience_counter}/{config['patience']} | "
                f"Time: {elapsed / 60:.1f}m | ETA: {eta / 60:.1f}m"
            )
            print(msg)
            with open(os.path.join(LOG_DIR, "training.log"), "a") as f:
                f.write(msg + "\n")

        if patience_counter >= config["patience"]:
            print(f"[{model_type}] Early stopping at epoch {epoch + 1}")
            break

    training_time = time.time() - start_time
    model.load_state_dict(torch.load(model_path))
    evaluation = evaluate_model(model, test_loader, test_ds, use_weighting, device)

    print(f"\n{'=' * 80}")
    print(f"RESULTS: {model_type.upper()}")
    print(f"{'=' * 80}")
    print(f"Training time: {training_time / 60:.1f} minutes")
    print(f"Parameters: {param_count:,}")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"\nOverall:")
    for k, v in evaluation["overall"].items():
        print(f"  {k}: {v:.4f}")
    if "normal" in evaluation:
        print(f"\nNormal ({evaluation['normal_samples']} samples):")
        for k, v in evaluation["normal"].items():
            print(f"  {k}: {v:.4f}")
    if "jump" in evaluation:
        print(f"\nJump ({evaluation['jump_samples']} samples):")
        for k, v in evaluation["jump"].items():
            print(f"  {k}: {v:.4f}")

    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    results = {
        "model_type": model_type,
        "architecture": "512x3",
        "normalization_window": 72,
        "config": config,
        "training_time_minutes": training_time / 60,
        "best_val_loss": best_val_loss,
        "timestamp": datetime.now().isoformat(),
        "evaluation": evaluation,
        "history": history,
        "model_path": model_path,
        "parameters": param_count,
        "device": str(device),
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"{model_type}_512x3_72h_{ts}.json")
    with open(results_file, "w") as f:
        json.dump(convert_numpy(results), f, indent=2)

    print(f"\nResults: {results_file}")
    print(f"Model: {model_path}")
    return results


def main():
    for d in [RESULTS_DIR, MODELS_DIR, LOG_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")

    print(f"\nConfig: {json.dumps(CONFIG, indent=2)}")
    print(f"Data: {DATA_PATH}")
    print(f"\nModels to train: {[m[0] for m in MODELS_TO_TRAIN]}")

    all_results = {}
    for model_type, use_weighting in MODELS_TO_TRAIN:
        result = train_one_model(model_type, use_weighting, CONFIG, device)
        all_results[model_type] = {
            "r2": result["evaluation"]["overall"]["R2"],
            "rmse": result["evaluation"]["overall"]["RMSE"],
            "direction_%": result["evaluation"]["overall"]["Direction_%"],
            "parameters": result["parameters"],
            "training_time_min": result["training_time_minutes"],
        }

    print(f"\n\n{'=' * 80}")
    print("FINAL SUMMARY - ALL MODELS")
    print(f"{'=' * 80}")
    print(f"{'Model':<15} {'R2':>8} {'RMSE':>8} {'Dir%':>8} {'Params':>10} {'Time':>8}")
    print("-" * 60)
    for name, r in all_results.items():
        print(
            f"{name:<15} {r['r2']:>8.4f} {r['rmse']:>8.4f} {r['direction_%']:>8.1f} {r['parameters']:>10,} {r['training_time_min']:>7.1f}m"
        )

    summary_file = os.path.join(
        RESULTS_DIR,
        f"all_models_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSummary: {summary_file}")


if __name__ == "__main__":
    main()

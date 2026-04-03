#!/usr/bin/env python3
"""
Self-contained LSTM training for ai-server.
All dependencies inlined. No external imports from project.

CUDA_VISIBLE_DEVICES=1 python scripts/train_all_512x3_72h.py
"""

import sys, os, json, time, traceback
from datetime import datetime
from pathlib import Path
import torch, torch.nn as nn
import numpy as np, pandas as pd
from torch.utils.data import Dataset, DataLoader

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "bitcoin_lstm_features_v1.6_final.csv")
RESULTS_DIR = os.path.join(BASE, "results", "server_training")
MODELS_DIR = os.path.join(BASE, "models")
LOG_DIR = os.path.join(BASE, "results", "logs")

FEATURE_SETS = {
    "market_lags": [
        "dvol_lag_1d",
        "dvol_lag_7d",
        "dvol_lag_30d",
        "transaction_volume",
        "network_activity",
        "nvrv",
        "dvol_rv_spread",
    ],
    "jump_aware": [
        "dvol_lag_1d",
        "dvol_lag_7d",
        "dvol_lag_30d",
        "transaction_volume",
        "network_activity",
        "nvrv",
        "dvol_rv_spread",
        "lee_mykland_jump",
        "jump_magnitude",
        "days_since_jump",
        "jump_cluster_7d",
    ],
    "market_jumps": [
        "transaction_volume",
        "network_activity",
        "nvrv",
        "dvol_rv_spread",
        "lee_mykland_jump",
        "jump_magnitude",
        "days_since_jump",
        "jump_cluster_7d",
    ],
    "market": ["transaction_volume", "network_activity", "nvrv", "dvol_rv_spread"],
}
JUMP_INDICATORS = {"lee_mykland_jump", "jump_cluster_7d"}

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
}
MODELS = [
    ("market_lags", False),
    ("jump_aware", True),
    ("market_jumps", False),
    ("market", False),
]


class LSTM_DVOL(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)


class LSTMDataset(Dataset):
    def __init__(self, data, feature_cols, seq_len, window, use_weighting):
        self.data = data.copy()
        self.feature_cols = feature_cols
        self.seq_len = seq_len
        self.use_weighting = use_weighting
        target = "dvol"
        self.data = self.data.dropna(subset=feature_cols + [target])

        for col in feature_cols:
            if col in JUMP_INDICATORS:
                self.data[f"{col}_n"] = self.data[col]
                continue
            rm = self.data[col].rolling(window, min_periods=1).mean()
            rs = self.data[col].rolling(window, min_periods=1).std().replace(0, 1e-8)
            self.data[f"{col}_n"] = (self.data[col] - rm) / rs

        trm = self.data[target].rolling(window, min_periods=1).mean()
        trs = self.data[target].rolling(window, min_periods=1).std().replace(0, 1e-8)
        self.data["target_n"] = (self.data[target] - trm) / trs
        self.data["trm"] = trm
        self.data["trs"] = trs
        self.data = self.data[window:]

        norm_cols = [f"{c}_n" if c not in JUMP_INDICATORS else c for c in feature_cols]
        X, y, w, s = [], [], [], []
        for i in range(seq_len, len(self.data) - 1):
            X.append(self.data[norm_cols].iloc[i - seq_len : i].values)
            y.append(self.data["target_n"].iloc[i + 1])
            if use_weighting:
                w.append(2.0 if self.data["lee_mykland_jump"].iloc[i + 1] else 1.0)
            else:
                w.append(1.0)
            s.append([self.data["trm"].iloc[i + 1], self.data["trs"].iloc[i + 1]])

        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y, dtype=np.float32)
        self.w = np.array(w, dtype=np.float32)
        self.s = np.array(s, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (
            torch.FloatTensor(self.X[i]),
            torch.FloatTensor([self.y[i]]),
            torch.FloatTensor([self.w[i]]),
            torch.FloatTensor(self.s[i]),
        )

    def inverse(self, pred, stats):
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(stats, torch.Tensor):
            stats = stats.cpu().numpy()
        return pred * stats[:, 1:2] + stats[:, 0:1]


def make_loaders(feature_set, use_weighting, cfg):
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    t1, t2 = int(n * 0.6), int(n * 0.8)
    cols = FEATURE_SETS[feature_set]
    sl, ws, bs = cfg["sequence_length"], cfg["window_size"], cfg["batch_size"]

    ds_t = LSTMDataset(df.iloc[:t1], cols, sl, ws, use_weighting)
    ds_v = LSTMDataset(df.iloc[t1:t2], cols, sl, ws, use_weighting)
    ds_e = LSTMDataset(df.iloc[t2:], cols, sl, ws, use_weighting)
    return (
        DataLoader(ds_t, bs, True),
        DataLoader(ds_v, bs),
        DataLoader(ds_e, bs),
        ds_t,
        ds_v,
        ds_e,
    )


def metrics(yt, yp):
    mse = np.mean((yt - yp) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(yt - yp))
    mape = np.mean(np.abs((yt - yp) / yt)) * 100
    ss_r = np.sum((yt - yp) ** 2)
    ss_t = np.sum((yt - np.mean(yt)) ** 2)
    r2 = 1 - ss_r / ss_t if ss_t > 0 else 0
    ad = np.sign(yt[1:] - yt[:-1])
    pd_ = np.sign(yp[:-1] - yt[:-1])
    v = ad != 0
    da = float((ad[v] == pd_[v]).sum() / v.sum() * 100) if v.sum() > 0 else 50.0
    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE": float(mape),
        "R2": float(r2),
        "Direction_%": da,
    }


def evaluate(model, loader, dataset, use_w, device):
    model.eval()
    P, T, W, S = [], [], [], []
    with torch.no_grad():
        for xb, yb, wb, sb in loader:
            P.append(model(xb.to(device)).cpu().numpy())
            T.append(yb.numpy())
            W.append(wb.numpy())
            S.append(sb.numpy())
    P, T, W, S = (
        np.concatenate(P),
        np.concatenate(T),
        np.concatenate(W),
        np.concatenate(S),
    )
    po = dataset.inverse(P, S)
    to = dataset.inverse(T, S)
    isj = W.flatten() > 1.0
    r = {
        "overall": metrics(to.flatten(), po.flatten()),
        "jump_samples": int(isj.sum()),
        "normal_samples": int((~isj).sum()),
    }
    if isj.sum() > 0 and (~isj).sum() > 0:
        r["normal"] = metrics(to.flatten()[~isj], po.flatten()[~isj])
        r["jump"] = metrics(to.flatten()[isj], po.flatten()[isj])
    return r


def train_one(name, use_w, cfg, device):
    cols = FEATURE_SETS[name]
    print(f"\n{'=' * 80}\nTRAINING: {name.upper()} ({len(cols)} features)\n{'=' * 80}")

    model = LSTM_DVOL(
        len(cols), cfg["hidden_size"], cfg["num_layers"], cfg["dropout"]
    ).to(device)
    pc = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {pc:,}")

    tl, vl, el, td, vd, ed = make_loaders(name, use_w, cfg)
    print(f"Train: {len(td)} | Val: {len(vd)} | Test: {len(ed)}")

    opt = torch.optim.Adam(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-5
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5
    )
    bvl = float("inf")
    pat = 0
    hist = {"train_loss": [], "val_loss": [], "lr": []}
    mp = os.path.join(MODELS_DIR, f"server_{name}_512x3_72h_best.pth")
    t0 = time.time()

    for ep in range(cfg["epochs"]):
        model.train()
        tls = []
        for xb, yb, wb, _ in tl:
            xb, yb = xb.to(device), yb.to(device)
            if use_w:
                wb = wb.to(device)
            opt.zero_grad()
            pr = model(xb)
            lo = ((pr - yb) ** 2 * wb).mean() if use_w else nn.MSELoss()(pr, yb)
            lo.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tls.append(lo.item())

        model.eval()
        vls = []
        with torch.no_grad():
            for xb, yb, wb, _ in vl:
                xb, yb = xb.to(device), yb.to(device)
                if use_w:
                    wb = wb.to(device)
                pr = model(xb)
                lo = ((pr - yb) ** 2 * wb).mean() if use_w else nn.MSELoss()(pr, yb)
                vls.append(lo.item())

        tl_, vl_ = np.mean(tls), np.mean(vls)
        hist["train_loss"].append(tl_)
        hist["val_loss"].append(vl_)
        hist["lr"].append(opt.param_groups[0]["lr"])
        sched.step(vl_)

        if vl_ < bvl:
            bvl = vl_
            pat = 0
            torch.save(model.state_dict(), mp)
        else:
            pat += 1

        if (ep + 1) % 2 == 0 or ep == 0:
            el_ = time.time() - t0
            eta = (cfg["epochs"] - ep - 1) * (el_ / (ep + 1))
            msg = (
                f"{datetime.now().isoformat()} | {name} | Ep {ep + 1:3d}/{cfg['epochs']} | "
                f"T:{tl_:.6f} V:{vl_:.6f} LR:{opt.param_groups[0]['lr']:.2e} "
                f"P:{pat}/{cfg['patience']} | {el_ / 60:.1f}m ETA:{eta / 60:.1f}m"
            )
            print(msg)
            with open(os.path.join(LOG_DIR, "training.log"), "a") as f:
                f.write(msg + "\n")

        if pat >= cfg["patience"]:
            print(f"[{name}] Early stop epoch {ep + 1}")
            break

    tt = time.time() - t0
    model.load_state_dict(torch.load(mp, weights_only=True))
    ev = evaluate(model, el, ed, use_w, device)

    print(f"\nRESULTS: {name.upper()} | {tt / 60:.1f}m | {pc:,} params")
    for k, v in ev["overall"].items():
        print(f"  {k}: {v:.4f}")
    if "normal" in ev:
        print(
            f"  Normal ({ev['normal_samples']}):",
            " ".join(f"{k}={v:.4f}" for k, v in ev["normal"].items()),
        )
    if "jump" in ev:
        print(
            f"  Jump ({ev['jump_samples']}):",
            " ".join(f"{k}={v:.4f}" for k, v in ev["jump"].items()),
        )

    def cn(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, dict):
            return {k: cn(v) for k, v in o.items()}
        if isinstance(o, list):
            return [cn(i) for i in o]
        return o

    res = {
        "model_type": name,
        "architecture": "512x3",
        "window": 72,
        "config": cfg,
        "training_time_min": tt / 60,
        "best_val_loss": bvl,
        "timestamp": datetime.now().isoformat(),
        "evaluation": ev,
        "history": hist,
        "model_path": mp,
        "parameters": pc,
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rf = os.path.join(RESULTS_DIR, f"{name}_512x3_72h_{ts}.json")
    with open(rf, "w") as f:
        json.dump(cn(res), f, indent=2)
    print(f"Saved: {rf}")
    return res


def main():
    for d in [RESULTS_DIR, MODELS_DIR, LOG_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)} | Count: {torch.cuda.device_count()}"
        )
    print(f"Config: {json.dumps(CONFIG, indent=2)}")
    print(f"Data: {DATA_PATH}")
    print(f"Models: {[m[0] for m in MODELS]}")

    summary = {}
    for name, uw in MODELS:
        try:
            r = train_one(name, uw, CONFIG, device)
            summary[name] = {
                "r2": r["evaluation"]["overall"]["R2"],
                "rmse": r["evaluation"]["overall"]["RMSE"],
                "dir%": r["evaluation"]["overall"]["Direction_%"],
                "params": r["parameters"],
                "time_min": r["training_time_min"],
            }
        except Exception as e:
            print(f"FAILED {name}: {e}\n{traceback.format_exc()}")
            summary[name] = {"error": str(e)}

    print(f"\n{'=' * 80}\nFINAL SUMMARY\n{'=' * 80}")
    print(f"{'Model':<15} {'R2':>8} {'RMSE':>8} {'Dir%':>8} {'Params':>10} {'Time':>8}")
    print("-" * 60)
    for n, r in summary.items():
        if "error" in r:
            print(f"{n:<15} FAILED: {r['error']}")
        else:
            print(
                f"{n:<15} {r['r2']:>8.4f} {r['rmse']:>8.4f} {r['dir%']:>8.1f} {r['params']:>10,} {r['time_min']:>7.1f}m"
            )

    sf = os.path.join(
        RESULTS_DIR, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(sf, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary: {sf}")


if __name__ == "__main__":
    main()

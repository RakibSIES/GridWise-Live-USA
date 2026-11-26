import os

# Prevent macOS OpenMP / libomp threading conflict between PyTorch and XGBoost
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from datetime import datetime, timedelta, timezone
from io import StringIO

import joblib
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import xgboost as xgb

torch.set_num_threads(1)

REGIONS = {
    "CAISO (California - Los Angeles)": {
        "tag": "caiso",
        "lat": 34.0522,
        "lon": -118.2437,
        "base_mw": 28500.0,
        "temp_sens": 420.0,
    },
    "ERCOT (Texas - Houston)": {
        "tag": "ercot",
        "lat": 29.7604,
        "lon": -95.3698,
        "base_mw": 46000.0,
        "temp_sens": 580.0,
    },
}

FEATURES = [
    "temp_c",
    "humidity",
    "hour",
    "dayofweek",
    "is_weekend",
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "roll_24h_mean",
]


class ResidualLSTM(nn.Module):
    """Stage 2 PyTorch LSTM modeling temporal residuals from Stage 1 XGBoost."""

    def __init__(self, input_size: int = 1, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def fetch_caiso_live_demand() -> float | None:
    """Fetch latest valid MW reading from CAISO's live public demand CSV feed."""
    url = "https://www.caiso.com/outlook/SP/History/demand.csv"
    try:
        resp = requests.get(url, timeout=12)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        candidates = [c for c in df.columns if "current" in c.lower() and "demand" in c.lower()]
        candidates += [c for c in reversed(df.columns[1:]) if c not in candidates]
        for col in candidates:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            vals = vals[vals > 1000.0]
            if not vals.empty:
                latest_mw = float(vals.iloc[-1])
                print(f"  [CAISO Live Feed] Latest Current Demand ({col.strip()}): {latest_mw:,.1f} MW")
                return latest_mw
    except Exception as exc:
        print(f"  [CAISO Live Feed] Could not read live CSV ({exc}); using baseline calibration.")
    return None


def build_training_dataset(region_name: str, days: int = 180) -> pd.DataFrame:
    """Fetch 180 days of hourly weather from Open-Meteo Archive API and calibrate demand."""
    cfg = REGIONS[region_name]
    end_dt = datetime.now(timezone.utc).date() - timedelta(days=1)
    start_dt = end_dt - timedelta(days=days)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": cfg["lat"],
        "longitude": cfg["lon"],
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m",
        "timezone": "UTC",
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]

    df = (
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime(hourly["time"]),
                "temp_c": hourly["temperature_2m"],
                "humidity": hourly["relative_humidity_2m"],
            }
        )
        .dropna()
        .reset_index(drop=True)
    )

    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # Realistic diurnal + thermal cooling/heating + weekend + correlated autoregressive noise
    diurnal = (
        np.sin((df["hour"] - 6) * np.pi / 14.0) * 0.18
        + np.where(df["hour"].between(16, 21), 0.14, 0.0)
    )
    thermal_load = (
        np.maximum(0.0, df["temp_c"] - 22.0) * cfg["temp_sens"]
        + np.maximum(0.0, 12.0 - df["temp_c"]) * (cfg["temp_sens"] * 0.65)
    )
    weekend_adj = np.where(df["is_weekend"] == 1, -0.07 * cfg["base_mw"], 0.0)

    rng = np.random.default_rng(42 if cfg["tag"] == "caiso" else 99)
    ar_shock = np.zeros(len(df))
    raw_noise = rng.normal(0.0, cfg["base_mw"] * 0.012, size=len(df))
    for i in range(1, len(df)):
        ar_shock[i] = 0.65 * ar_shock[i - 1] + raw_noise[i]

    df["demand_mw"] = cfg["base_mw"] * (1.0 + diurnal) + thermal_load + weekend_adj + ar_shock

    # Calibrate demand against CAISO's live public feed
    live_caiso_mw = fetch_caiso_live_demand()
    if live_caiso_mw is not None:
        if cfg["tag"] == "caiso":
            scale_ratio = float(np.clip(live_caiso_mw / df["demand_mw"].iloc[-1], 0.80, 1.20))
            df["demand_mw"] *= scale_ratio
            print(f"  [Calibration] Applied CAISO live scale factor: {scale_ratio:.4f}")
        else:
            # Cross-calibrate ERCOT baseline drift gently from national grid conditions
            caiso_ref = 28500.0
            drift_ratio = float(np.clip(live_caiso_mw / caiso_ref, 0.90, 1.10))
            df["demand_mw"] *= 0.5 + 0.5 * drift_ratio

    df["lag_1h"] = df["demand_mw"].shift(1)
    df["lag_24h"] = df["demand_mw"].shift(24)
    df["lag_168h"] = df["demand_mw"].shift(168)
    df["roll_24h_mean"] = df["demand_mw"].rolling(24).mean()
    return df.dropna().reset_index(drop=True)


def train_models() -> None:
    os.makedirs("models", exist_ok=True)
    torch.manual_seed(42)

    for region_name, cfg in REGIONS.items():
        tag = cfg["tag"]
        print(f"\n=== Training Hybrid XGBoost + PyTorch LSTM for {region_name} ===")
        df = build_training_dataset(region_name, days=180)
        df.to_csv(f"models/{tag}_history.csv", index=False)

        X = df[FEATURES]
        y = df["demand_mw"]

        # Stage 1: XGBRegressor
        model_xgb = xgb.XGBRegressor(
            n_estimators=220,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=1,
            random_state=42,
        )
        model_xgb.fit(X, y)
        joblib.dump(model_xgb, f"models/{tag}_xgb.pkl")

        stage1_preds = model_xgb.predict(X)
        residuals = (y - stage1_preds).to_numpy(dtype=np.float32)
        res_std = float(np.std(residuals)) if float(np.std(residuals)) > 1e-6 else 1.0
        res_norm = residuals / res_std

        # Stage 2: PyTorch LSTM on 24-hour residual windows
        seq_len = 24
        X_seq, y_seq = [], []
        for i in range(len(res_norm) - seq_len):
            X_seq.append(res_norm[i : i + seq_len])
            y_seq.append(res_norm[i + seq_len])

        X_t = torch.tensor(np.array(X_seq, dtype=np.float32)).unsqueeze(-1)
        y_t = torch.tensor(np.array(y_seq, dtype=np.float32)).unsqueeze(-1)

        model_lstm = ResidualLSTM(input_size=1, hidden_size=32)
        optimizer = torch.optim.Adam(model_lstm.parameters(), lr=0.008)
        loss_fn = nn.MSELoss()

        model_lstm.train()
        batch_size = 256
        for epoch in range(20):
            perm = torch.randperm(X_t.size(0))
            for start in range(0, X_t.size(0), batch_size):
                idx = perm[start : start + batch_size]
                optimizer.zero_grad()
                pred_res = model_lstm(X_t[idx])
                loss = loss_fn(pred_res, y_t[idx])
                loss.backward()
                optimizer.step()

        model_lstm.eval()
        with torch.no_grad():
            lstm_res_preds = model_lstm(X_t).squeeze(-1).numpy() * res_std

        hybrid_preds = stage1_preds[seq_len:] + lstm_res_preds
        y_eval = y.to_numpy()[seq_len:]

        s1_mae = mean_absolute_error(y_eval, stage1_preds[seq_len:])
        hyb_mae = mean_absolute_error(y_eval, hybrid_preds)
        hyb_rmse = root_mean_squared_error(y_eval, hybrid_preds)

        torch.save(model_lstm.state_dict(), f"models/{tag}_lstm.pt")
        joblib.dump(
            {
                "res_std": res_std,
                "features": FEATURES,
                "seq_len": seq_len,
                "stage1_mae": float(s1_mae),
                "hybrid_mae": float(hyb_mae),
                "hybrid_rmse": float(hyb_rmse),
                "trained_at": datetime.now(timezone.utc).isoformat(),
            },
            f"models/{tag}_meta.pkl",
        )
        print(
            f"  Stage 1 MAE: {s1_mae:.1f} MW | Hybrid MAE: {hyb_mae:.1f} MW | Hybrid RMSE: {hyb_rmse:.1f} MW"
        )
        print(f"  Saved models/{tag}_xgb.pkl, models/{tag}_lstm.pt, and models/{tag}_meta.pkl")


if __name__ == "__main__":
    train_models()

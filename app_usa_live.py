import os

# Prevent macOS OpenMP / libomp threading conflict between PyTorch and XGBoost
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from datetime import datetime
from io import StringIO

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import shap
import streamlit as st
import torch
import torch.nn as nn
import xgboost  # noqa: F401 - required before unpickling XGBRegressor

torch.set_num_threads(1)

st.set_page_config(
    page_title="GridWise-Live USA | Real-Time Hybrid Forecast",
    page_icon="⚡",
    layout="wide",
)


class ResidualLSTM(nn.Module):
    """Stage 2 PyTorch LSTM modeling temporal residuals from Stage 1 XGBoost."""

    def __init__(self, input_size: int = 1, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


REGIONS = {
    "CAISO (California - Los Angeles)": {
        "tag": "caiso",
        "lat": 34.0522,
        "lon": -118.2437,
        "capacity_mw": 35500.0,
    },
    "ERCOT (Texas - Houston)": {
        "tag": "ercot",
        "lat": 29.7604,
        "lon": -95.3698,
        "capacity_mw": 58000.0,
    },
}

FEATURE_LABELS = {
    "temp_c": "Temperature (°C)",
    "humidity": "Relative Humidity (%)",
    "hour": "Hour of Day",
    "dayofweek": "Day of Week",
    "is_weekend": "Weekend Flag",
    "lag_1h": "1h Lag Demand (MW)",
    "lag_24h": "24h Lag Demand (MW)",
    "lag_168h": "7d Lag Demand (MW)",
    "roll_24h_mean": "24h Rolling Mean (MW)",
}


@st.cache_resource
def load_region_assets(tag: str):
    hist_df = pd.read_csv(f"models/{tag}_history.csv")
    default_features = [
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
    try:
        meta = joblib.load(f"models/{tag}_meta.pkl")
    except Exception:
        meta = {"features": default_features, "res_std": 300.0, "seq_len": 24}

    try:
        xgb_m = joblib.load(f"models/{tag}_xgb.pkl")
        xgb_m.set_params(n_jobs=1)
    except Exception:
        # Fallback if cloud Python/XGBoost version differs from local pickle version
        feats = meta.get("features", default_features)
        xgb_m = xgboost.XGBRegressor(
            n_estimators=220,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=1,
            random_state=42,
        )
        xgb_m.fit(hist_df[feats], hist_df["demand_mw"])

    lstm_m = ResidualLSTM(input_size=1, hidden_size=32)
    lstm_m.load_state_dict(torch.load(f"models/{tag}_lstm.pt", map_location="cpu", weights_only=True))
    lstm_m.eval()
    explainer = shap.TreeExplainer(xgb_m)
    return xgb_m, lstm_m, meta, hist_df, explainer


@st.cache_data(ttl=300)
def fetch_live_usa_telemetry(region_name: str):
    cfg = REGIONS[region_name]
    w_url = "https://api.open-meteo.com/v1/forecast"
    w_params = {
        "latitude": cfg["lat"],
        "longitude": cfg["lon"],
        "hourly": "temperature_2m,relative_humidity_2m",
        "current_weather": "true",
        "forecast_days": 3,
        "timezone": "auto",
    }
    w_resp = requests.get(w_url, params=w_params, timeout=12)
    w_resp.raise_for_status()
    w_data = w_resp.json()

    live_grid_mw = None
    grid_source = "Open-Meteo Live Weather Forecast API + Hybrid Grid Model"
    if cfg["tag"] == "caiso":
        try:
            c_resp = requests.get("https://www.caiso.com/outlook/SP/History/demand.csv", timeout=10)
            c_resp.raise_for_status()
            c_df = pd.read_csv(StringIO(c_resp.text))
            candidates = [c for c in c_df.columns if "current" in c.lower() and "demand" in c.lower()]
            candidates += [c for c in reversed(c_df.columns[1:]) if c not in candidates]
            for col in candidates:
                vals = pd.to_numeric(c_df[col], errors="coerce").dropna()
                vals = vals[vals > 1000.0]
                if not vals.empty:
                    live_grid_mw = float(vals.iloc[-1])
                    grid_source = "CAISO Official Live Feed (caiso.com) + Open-Meteo Forecast API"
                    break
        except Exception:
            pass

    return w_data, live_grid_mw, grid_source


st.sidebar.header("🇺🇸 GridWise-Live Control Panel")
region = st.sidebar.selectbox("Select US Grid Operator", list(REGIONS.keys()))
st.sidebar.subheader("Stress-Test Scenarios")
heat_boost = st.sidebar.checkbox("🔥 Simulate Heatwave (+5.0 °C)", value=False)
gen_outage = st.sidebar.checkbox("⚠️ Simulate Generator Outage (-12% Capacity)", value=False)
if st.sidebar.button("🔄 Refresh Live Telemetry"):
    st.cache_data.clear()
    st.rerun()

cfg = REGIONS[region]
required_files = [
    f"models/{cfg['tag']}_xgb.pkl",
    f"models/{cfg['tag']}_lstm.pt",
    f"models/{cfg['tag']}_meta.pkl",
    f"models/{cfg['tag']}_history.csv",
]
if not all(os.path.exists(p) for p in required_files):
    st.error(
        f"Model artifacts for **{region}** were not found in `models/`. "
        "Please run `.venv/bin/python train_usa_hybrid.py` first, then refresh this page."
    )
    st.stop()

xgb_m, lstm_m, meta, hist_df, explainer = load_region_assets(cfg["tag"])
features = meta["features"]
res_std = float(meta["res_std"])
seq_len = int(meta.get("seq_len", 24))

try:
    w_data, live_grid_mw, grid_source = fetch_live_usa_telemetry(region)
    all_times = pd.Series(pd.to_datetime(w_data["hourly"]["time"]))
    all_temps = np.array(w_data["hourly"]["temperature_2m"], dtype=float)
    all_hums = np.array(w_data["hourly"]["relative_humidity_2m"], dtype=float)
    curr_time = pd.to_datetime(w_data["current_weather"]["time"])

    future_mask = (all_times >= curr_time).to_numpy()
    if int(np.sum(future_mask)) >= 24:
        hourly_times = all_times[future_mask].iloc[:24].reset_index(drop=True)
        hourly_temps = all_temps[future_mask][:24]
        hourly_hums = all_hums[future_mask][:24]
    else:
        hourly_times = all_times.iloc[:24].reset_index(drop=True)
        hourly_temps = all_temps[:24]
        hourly_hums = all_hums[:24]

    curr_temp = float(w_data["current_weather"]["temperature"])
except Exception as exc:
    st.warning(f"Live Open-Meteo API temporarily unreachable ({exc}); using cached historical tail.")
    tail_24 = hist_df.tail(24).reset_index(drop=True)
    hourly_times = pd.to_datetime(tail_24["timestamp"])
    hourly_temps = tail_24["temp_c"].to_numpy(dtype=float)
    hourly_hums = tail_24["humidity"].to_numpy(dtype=float)
    curr_temp = float(hourly_temps[0])
    live_grid_mw = None
    grid_source = "Local Historical Fallback"

temp_offset = 5.0 if heat_boost else 0.0
curr_temp_adj = curr_temp + temp_offset
hourly_temps_adj = hourly_temps + temp_offset

st.sidebar.markdown("---")
st.sidebar.metric(
    "Current Local Temperature",
    f"{curr_temp_adj:.1f} °C ({curr_temp_adj * 9 / 5 + 32:.1f} °F)",
    "+5.0 °C Heatwave Override" if heat_boost else "Live Open-Meteo",
)
if live_grid_mw is not None:
    st.sidebar.metric("Live CAISO System Demand", f"{live_grid_mw:,.0f} MW", "Official Real-Time Feed")
if {"stage1_mae", "hybrid_mae", "hybrid_rmse"}.issubset(meta):
    st.sidebar.markdown("---")
    st.sidebar.subheader("Trained Model Metrics")
    st.sidebar.metric("Stage 1 XGBoost MAE", f"{meta['stage1_mae']:,.1f} MW")
    st.sidebar.metric(
        "Hybrid MAE (XGB + LSTM)",
        f"{meta['hybrid_mae']:,.1f} MW",
        f"{meta['hybrid_mae'] - meta['stage1_mae']:+.1f} MW vs Stage 1",
        delta_color="inverse",
    )
    st.sidebar.metric("Hybrid RMSE", f"{meta['hybrid_rmse']:,.1f} MW")
st.sidebar.caption(f"Refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {grid_source}")

# Seed rolling demand & residual state
recent_window = hist_df.tail(168).copy()
if live_grid_mw is not None:
    recent_window.iloc[-1, recent_window.columns.get_loc("demand_mw")] = live_grid_mw

recent_tail = recent_window.tail(seq_len)
res_buffer = list(
    ((recent_tail["demand_mw"].to_numpy() - xgb_m.predict(recent_tail[features])) / res_std).astype(np.float32)
)
rolling_demands = list(recent_window["demand_mw"].to_numpy(dtype=float))

stage1_preds, final_preds, lstm_corrections, feature_rows = [], [], [], []

for h in range(24):
    ts = pd.Timestamp(hourly_times.iloc[h])
    row = {
        "temp_c": float(hourly_temps_adj[h]),
        "humidity": float(hourly_hums[h]),
        "hour": int(ts.hour),
        "dayofweek": int(ts.dayofweek),
        "is_weekend": int(ts.dayofweek in [5, 6]),
        "lag_1h": float(rolling_demands[-1]),
        "lag_24h": float(rolling_demands[-24]),
        "lag_168h": float(rolling_demands[-168]),
        "roll_24h_mean": float(np.mean(rolling_demands[-24:])),
    }
    X_step = pd.DataFrame([row])[features]
    s1_mw = float(xgb_m.predict(X_step)[0])

    with torch.no_grad():
        seq_tensor = torch.tensor(res_buffer[-seq_len:], dtype=torch.float32).reshape(1, seq_len, 1)
        pred_res_norm = float(lstm_m(seq_tensor).item())

    lstm_mw = pred_res_norm * res_std
    s2_mw = s1_mw + lstm_mw

    stage1_preds.append(round(s1_mw, 1))
    final_preds.append(round(s2_mw, 1))
    lstm_corrections.append(round(lstm_mw, 1))
    feature_rows.append(row)

    rolling_demands.append(s2_mw)
    res_buffer.append(np.float32(pred_res_norm * 0.85))

avail_cap = cfg["capacity_mw"] * (0.88 if gen_outage else 1.0)
deficits = np.maximum(0.0, np.array(final_preds) - avail_cap)
peak_idx = int(np.argmax(final_preds))
peak_time_str = pd.Timestamp(hourly_times.iloc[peak_idx]).strftime("%b %d, %H:%M")

st.title(f"⚡ GridWise-Live USA: 24-Hour Hybrid Forecast — {region}")
st.caption(f"Two-Stage XGBoost + PyTorch Residual LSTM | Source: {grid_source}")

if float(np.max(deficits)) > 0:
    st.error(
        f"🚨 **Capacity Deficit Warning:** Forecast peak ({max(final_preds):,.0f} MW at {peak_time_str}) "
        f"exceeds available grid capacity ({avail_cap:,.0f} MW) by **{max(deficits):,.0f} MW**."
    )

c1, c2, c3, c4 = st.columns(4)
c1.metric("Peak 24h Forecast Load", f"{max(final_preds):,.0f} MW", f"At {peak_time_str}")
c2.metric(
    "Available Grid Capacity",
    f"{avail_cap:,.0f} MW",
    "-12% Outage Applied" if gen_outage else "100% Nominal Reserve",
    delta_color="inverse" if gen_outage else "normal",
)
c3.metric(
    "Peak Reserve Deficit",
    f"{max(deficits):,.0f} MW",
    "Load Shedding Risk" if max(deficits) > 0 else "Adequate Reserve Margin",
    delta_color="inverse" if max(deficits) > 0 else "off",
)
c4.metric(
    "Stage 2 LSTM Adjustment (Peak)",
    f"{lstm_corrections[peak_idx]:+.1f} MW",
    "PyTorch Residual Memory",
)

tab1, tab2 = st.tabs(
    [
        "📈 Live 24-Hour Demand & Capacity Forecast",
        "🧠 SHAP Feature Attribution",
    ]
)

with tab1:
    x_labels = [pd.Timestamp(t).strftime("%b %d %H:%M") for t in hourly_times]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=stage1_preds,
            name="Stage 1: XGBRegressor",
            line=dict(color="#636EFA", width=2, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=final_preds,
            name="Stage 2 Hybrid (XGBoost + PyTorch LSTM)",
            line=dict(color="#FF4B4B", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=[avail_cap] * 24,
            name="Available Generation Capacity",
            line=dict(color="#00CC96", width=2, dash="dash"),
        )
    )
    if float(np.max(deficits)) > 0:
        deficit_y = [p if p > avail_cap else avail_cap for p in final_preds]
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=deficit_y,
                name="Capacity Deficit Zone",
                fill="tonexty",
                fillcolor="rgba(255, 75, 75, 0.22)",
                line=dict(width=0),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        height=460,
        yaxis_title="Electric Load / Capacity (MW)",
        xaxis_title="Forecast Horizon (Local Grid Time, Next 24 Hours)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, width="stretch")

    forecast_table = pd.DataFrame(
        {
            "Timestamp": x_labels,
            "Temp (°C)": np.round(hourly_temps_adj, 1),
            "Humidity (%)": np.round(hourly_hums, 0),
            "Stage 1 XGBoost (MW)": stage1_preds,
            "Stage 2 LSTM Residual (MW)": lstm_corrections,
            "Final Hybrid Forecast (MW)": final_preds,
            "Available Capacity (MW)": np.round(avail_cap, 1),
            "Deficit (MW)": np.round(deficits, 1),
        }
    )
    with st.expander("View Hourly 24-Hour Telemetry & Forecast Table"):
        st.dataframe(forecast_table, width="stretch", hide_index=True)

with tab2:
    selected_hour = st.slider(
        "Inspect Forecast Hour (0 = Now, Default = Peak Hour)",
        min_value=0,
        max_value=23,
        value=peak_idx,
    )
    target_ts = pd.Timestamp(hourly_times.iloc[selected_hour]).strftime("%b %d, %H:%M")
    X_target = pd.DataFrame([feature_rows[selected_hour]])[features]
    sv = np.asarray(explainer.shap_values(X_target))[0]
    base_val = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    shap_df = pd.DataFrame(
        {
            "Feature": [FEATURE_LABELS.get(f, f) for f in features],
            "RawFeature": features,
            "Value": X_target.iloc[0].to_numpy(),
            "SHAP_MW": sv,
        }
    ).sort_values("SHAP_MW", ascending=True)

    fig_s = go.Figure(
        go.Bar(
            x=shap_df["SHAP_MW"],
            y=[f"{lbl} = {val:,.1f}" for lbl, val in zip(shap_df["Feature"], shap_df["Value"])],
            orientation="h",
            marker_color=["#FF4B4B" if x >= 0 else "#00CC96" for x in shap_df["SHAP_MW"]],
            text=[f"{x:+,.1f} MW" for x in shap_df["SHAP_MW"]],
            textposition="outside",
        )
    )
    fig_s.update_layout(
        template="plotly_dark",
        height=440,
        title=(
            f"SHAP Feature Attribution at {target_ts} | "
            f"Baseline: {base_val:,.0f} MW → Stage 1 Forecast: {stage1_preds[selected_hour]:,.0f} MW"
        ),
        xaxis_title="Marginal Contribution to Predicted Load (MW)",
        margin=dict(l=20, r=60, t=50, b=20),
    )
    st.plotly_chart(fig_s, width="stretch")

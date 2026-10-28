"""Reproducible synthetic time-series forecasting and honest holdout evaluation.

Demonstration only: never use for clinical or operational decisions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

LAGS = (0, 24, 48, 168)
FEATURES = ["hour", "dow", "weekend", "lag_0", "lag_24", "lag_48", "lag_168"]


def synthetic_arrivals(days: int = 150, seed: int = 2026) -> pd.DataFrame:
    """Simulate aggregate arrivals, including a deliberately shifted final regime."""
    if days < 35:
        raise ValueError("At least 35 days required")
    rng = np.random.default_rng(seed)
    hours = days * 24
    idx = np.arange(hours)
    hour = idx % 24
    dow = (idx // 24) % 7
    rate = (
        9 + 2.7 * np.sin(2 * np.pi * (hour - 8) / 24)
        + 1.3 * (dow < 5) + 0.8 * np.sin(2 * np.pi * idx / 168)
        + np.where(idx >= int(hours * 0.83), 3.1, 0)
    )
    counts = rng.poisson(np.clip(rate, 0.5, None))
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=hours, freq="h"),
        "arrivals": counts,
    })


def supervised(data: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    """Each origin uses only observations available by the forecasting origin."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    df = data.copy().sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].duplicated().any() or df["arrivals"].isna().any():
        raise ValueError("Duplicated timestamps or missing counts")
    if (df["arrivals"] < 0).any():
        raise ValueError("Negative arrivals")
    out = pd.DataFrame({
        "origin_time": df["timestamp"],
        "target_time": df["timestamp"].shift(-horizon),
        "target": df["arrivals"].shift(-horizon),
        "hour": df["timestamp"].dt.hour,
        "dow": df["timestamp"].dt.dayofweek,
        "weekend": (df["timestamp"].dt.dayofweek >= 5).astype(int),
    })
    for lag in LAGS:
        out[f"lag_{lag}"] = df["arrivals"].shift(lag)
    return out.dropna().reset_index(drop=True)


def chronological_split(frame: pd.DataFrame, train_fraction: float = .65,
                        calibration_fraction: float = .15, horizon: int = 24):
    """Purged chronological split prevents train labels overlapping future origins."""
    n = len(frame)
    if not 0 < train_fraction < 1 or not 0 < calibration_fraction < 1:
        raise ValueError("Invalid split fractions")
    first, second = int(n * train_fraction), int(n * (train_fraction + calibration_fraction))
    if first <= horizon or second - first <= horizon or n - second <= 0:
        raise ValueError("Insufficient rows")
    train = frame.iloc[:first - horizon].copy()
    cal = frame.iloc[first:second - horizon].copy()
    test = frame.iloc[second:].copy()
    if not train["target_time"].max() < cal["origin_time"].min():
        raise AssertionError("Train/calibration temporal leakage")
    if not cal["target_time"].max() < test["origin_time"].min():
        raise AssertionError("Calibration/test temporal leakage")
    return train, cal, test


def absolute_residual_radius(actual, predicted, alpha: float = .2) -> float:
    """Finite-sample split conformal radius under exchangeability assumptions."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    errors = np.sort(np.abs(np.asarray(actual) - np.asarray(predicted)))
    n = len(errors)
    if n == 0:
        raise ValueError("Empty calibration set")
    rank = min(n, int(np.ceil((n + 1) * (1 - alpha))))
    return float(errors[rank - 1])


def capacity_scenarios(actual, predicted, capacities=(6, 8, 10)):
    """Simple queue simulation; not a staffing decision model."""
    result = []
    for capacity in capacities:
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        backlog, peak = 0, 0
        for arrivals in actual:
            backlog = max(0, backlog + int(arrivals) - capacity)
            peak = max(peak, backlog)
        result.append({"capacity_per_hour": capacity, "final_backlog": backlog,
                       "peak_backlog": peak, "forecast_total": float(sum(predicted))})
    return result


def mae(y, pred):
    return float(mean_absolute_error(y, pred))


def run_experiment(output_dir: str | Path = "reports", seed: int = 2026, days: int = 150):
    """Train, calibrate, test and save independently reproducible artifacts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = synthetic_arrivals(days=days, seed=seed)
    data_path = output / "synthetic_hourly_arrivals.csv"
    data.to_csv(data_path, index=False)
    examples = supervised(data)
    train, cal, test = chronological_split(examples)
    model = HistGradientBoostingRegressor(
        loss="poisson", max_iter=130, max_leaf_nodes=15, l2_regularization=3.0,
        learning_rate=.045, random_state=seed,
    )
    model.fit(train[FEATURES], train["target"])
    cal_pred = model.predict(cal[FEATURES])
    radius = absolute_residual_radius(cal["target"], cal_pred)
    actual = test["target"].to_numpy()
    predicted = model.predict(test[FEATURES])
    lo, hi = np.maximum(0, predicted - radius), predicted + radius
    daily = test["lag_0"].to_numpy()
    weekly = test["lag_168"].to_numpy()
    pred = pd.DataFrame({
        "origin_time": test["origin_time"], "target_time": test["target_time"],
        "actual": actual, "forecast": predicted, "lower": lo, "upper": hi,
        "daily_naive": daily, "weekly_naive": weekly,
    })
    pred.to_csv(output / "predictions.csv", index=False)
    metrics = {
        "data": "Synthetic aggregate counts, not actual patient data",
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "seed": seed, "horizon_hours": 24,
        "split": {"train_examples": len(train), "calibration_examples": len(cal),
                  "test_examples": len(test)},
        "point_forecast": {
            "poisson_hist_gradient_boosting": {"mae": mae(actual, predicted),
                "rmse": float(np.sqrt(mean_squared_error(actual, predicted)))},
            "daily_naive": {"mae": mae(actual, daily)},
            "weekly_naive": {"mae": mae(actual, weekly)}
        },
        "interval": {"nominal_coverage": .8,
                     "empirical_coverage": float(np.mean((actual >= lo) & (actual <= hi))),
                     "calibration_radius": radius},
        "distribution_shift": {"train_target_mean": float(train["target"].mean()),
                               "test_target_mean": float(test["target"].mean())},
        "warnings": [
            "Synthetic data: does not establish clinical or real-world validity.",
            "Conformal exchangeability may fail with temporal drift.",
            "Capacity simulations are illustrative only."
        ],
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    scenarios = {"not_for": "clinical or staffing decisions", "scenarios":
                 capacity_scenarios(actual[:168], predicted[:168])}
    (output / "capacity_scenarios.json").write_text(json.dumps(scenarios, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(11, 4))
    small = pred.head(168)
    ax.plot(small["target_time"], small["actual"], label="Synthetic observations", alpha=.6)
    ax.plot(small["target_time"], small["forecast"], label="Forecast")
    ax.legend()
    ax.set(title="WardPulse: first 168 holdout hours", ylabel="Hourly arrivals", xlabel="Time")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / "forecast_preview.png", dpi=130)
    plt.close(fig)
    return metrics

"""
risk_model.py — ML risk scoring (FR-8, FR-9, FR-10)

Trains (or loads) a Gradient Boosting Regressor that maps conjunction event
features → 0–100 risk score.  Supports two data paths:

  1. PRIMARY: Real ESA Kelvins CDM data (train_data.csv / test_data.csv)
  2. FALLBACK: Physics-based synthetic training set (if real data is missing)

The model uses exactly 4 features (per TRD §6 and 05_BACKEND_SCHEMA.md §6):
  miss_distance_km, relative_velocity_kms, tca_hours_from_now, size_class

Column mapping from ESA CDM (per build prompt spec):
  miss_distance (m)   → miss_distance_km      (÷ 1000)
  relative_speed (m/s)→ relative_velocity_kms  (÷ 1000)
  time_to_tca (days)  → tca_hours_from_now     (× 24, clip ≥ 0)
  c_object_type       → size_class             ROCKET BODY→3, PAYLOAD→2, else→1
  risk (log10 Pc)     → risk_score             ((risk + 30) / 30) * 100, clip [0,100]

Data contract out (05_BACKEND_SCHEMA.md §5 — Scored Event Record):
  Conjunction Event columns + risk_score (0-100) + risk_category
"""

from __future__ import annotations
import os
import json
import warnings
import pathlib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# ─── Constants ────────────────────────────────────────────────────────────────

RANDOM_STATE = 42
MODEL_DIR = pathlib.Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "risk_model.joblib"
REPORT_PATH = MODEL_DIR / "training_report.json"

# Paths to ESA Kelvins CDM data
_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR = _PROJECT_ROOT / "dataSet"
TRAIN_CSV = DATA_DIR / "train_data.csv"
TEST_CSV = DATA_DIR / "test_data.csv"

# Risk score conversion constants (fixed per build spec)
RISK_FLOOR = -30.0
RISK_CEIL = 0.0  # Using round number; actual max ≈ -1.44

# Object type → size class mapping
OBJECT_TYPE_MAP = {
    "ROCKET BODY": 3,
    "PAYLOAD": 2,
    "DEBRIS": 1,
    "UNKNOWN": 1,
    "TBA": 1,
}

RISK_CATEGORIES = [
    (20, "Low"),
    (50, "Medium"),
    (80, "High"),
    (100, "Critical"),
]


# ─── Column mapping ──────────────────────────────────────────────────────────

def _map_cdm_to_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map real ESA CDM columns to the 4-feature schema."""
    out = pd.DataFrame(index=df.index)
    out["miss_distance_km"] = df["miss_distance"] / 1000.0
    out["relative_velocity_kms"] = df["relative_speed"] / 1000.0
    out["tca_hours_from_now"] = (df["time_to_tca"] * 24.0).clip(lower=0)
    out["size_class"] = df["c_object_type"].map(OBJECT_TYPE_MAP).fillna(1).astype(int)
    return out


def _map_cdm_risk_to_score(risk_log10: pd.Series) -> pd.Series:
    """Convert log10(Pc) → 0-100 risk score per build spec formula."""
    score = ((risk_log10 + 30) / 30) * 100
    return score.clip(0, 100)


def categorize_risk(score: float) -> str:
    """Map a 0-100 risk score to a category label."""
    for threshold, label in RISK_CATEGORIES:
        if score <= threshold:
            return label
    return "Critical"


# ─── Synthetic fallback ──────────────────────────────────────────────────────

def _generate_synthetic_training_set(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generate physics-based synthetic training data as documented in risk_model.py.
    Used ONLY if real ESA CDM data files are missing (per TRD extensibility req).

    Formula: closer + faster + sooner + bigger → higher risk (monotonic).
    """
    rng = np.random.RandomState(RANDOM_STATE)

    miss_dist = rng.exponential(scale=30.0, size=n_samples)     # km
    rel_vel = rng.uniform(0.1, 15.0, size=n_samples)            # km/s
    tca_hours = rng.uniform(0.5, 168.0, size=n_samples)         # hours
    size_class = rng.choice([1, 2, 3], size=n_samples, p=[0.6, 0.3, 0.1])

    # Physics-based synthetic risk label (documented formula)
    raw = (rel_vel * 10) / (miss_dist + 0.5) / np.sqrt(tca_hours) * size_class
    risk_score = np.clip(100 * raw / np.percentile(raw, 99), 0, 100)

    return pd.DataFrame({
        "miss_distance_km": miss_dist,
        "relative_velocity_kms": rel_vel,
        "tca_hours_from_now": tca_hours,
        "size_class": size_class,
        "risk_score": risk_score,
    })


# ─── Training ────────────────────────────────────────────────────────────────

def train_model(
    force_synthetic: bool = False,
) -> Tuple[GradientBoostingRegressor, Dict[str, Any]]:
    """
    Train the risk model on ESA CDM data (primary) or synthetic data (fallback).

    Uses GradientBoostingRegressor(n_estimators=200, max_depth=3, lr=0.05,
    random_state=42) per TRD §6.

    Returns
    -------
    (model, report_dict)
        model: trained GBR
        report_dict: MAE, R², data source, feature importances
    """
    features = ["miss_distance_km", "relative_velocity_kms",
                "tca_hours_from_now", "size_class"]

    use_real_data = (not force_synthetic
                     and TRAIN_CSV.exists()
                     and TEST_CSV.exists())

    if use_real_data:
        # ── Real ESA Kelvins CDM data ──
        print("Loading ESA Kelvins CDM training data...")
        train_df = pd.read_csv(TRAIN_CSV)
        test_df = pd.read_csv(TEST_CSV)

        X_train = _map_cdm_to_features(train_df)
        y_train = _map_cdm_risk_to_score(train_df["risk"])

        X_test = _map_cdm_to_features(test_df)
        y_test = _map_cdm_risk_to_score(test_df["risk"])

        data_source = "esa_kelvins_cdm"
        n_train = len(train_df)
        n_test = len(test_df)
    else:
        # ── Synthetic fallback ──
        warnings.warn("Real CDM data not found — training on synthetic data.")
        synth = _generate_synthetic_training_set(n_samples=8000)
        split = int(len(synth) * 0.8)

        X_all = synth[features]
        y_all = synth["risk_score"]

        X_train, X_test = X_all.iloc[:split], X_all.iloc[split:]
        y_train, y_test = y_all.iloc[:split], y_all.iloc[split:]

        data_source = "synthetic_physics"
        n_train = split
        n_test = len(synth) - split

    # ── Train ──
    print(f"Training GBR on {n_train} samples ({data_source})...")
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    # ── Evaluate ──
    pred_test = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, pred_test))

    # Feature importances (built-in GBR)
    importances = dict(zip(features, [float(x) for x in model.feature_importances_]))

    # Calibration data — predicted vs actual in buckets
    calibration = _compute_calibration(y_test.values, pred_test)

    report = {
        "data_source": data_source,
        "n_train": n_train,
        "n_test": n_test,
        "mae": round(mae, 2),
        "feature_importances": importances,
        "calibration": calibration,
        "model_params": {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.05,
            "random_state": RANDOM_STATE,
        },
        "note": (
            f"MAE of {mae:.1f} points is an honest result for this 4-feature "
            f"model on noisy Pc data. This is observed on the test split, "
            f"not a guarantee of operational accuracy."
        ),
    }

    # ── Save ──
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "report": report}, MODEL_PATH)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Model saved to {MODEL_PATH}")
    print(f"MAE on test set: {mae:.2f} (data source: {data_source})")

    return model, report


def _compute_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> list:
    """Compute calibration buckets: predicted vs observed mean per decile."""
    bins = np.linspace(0, 100, n_bins + 1)
    cal = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_pred >= lo) & (y_pred < hi)
        if mask.sum() > 0:
            cal.append({
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "mean_predicted": float(np.mean(y_pred[mask])),
                "mean_actual": float(np.mean(y_true[mask])),
                "count": int(mask.sum()),
            })
    return cal


# ─── Scoring ──────────────────────────────────────────────────────────────────

def load_model() -> Tuple[GradientBoostingRegressor, Dict[str, Any]]:
    """Load a pre-trained model, or train one if none exists."""
    if MODEL_PATH.exists():
        data = joblib.load(MODEL_PATH)
        return data["model"], data.get("report", {})
    else:
        return train_model()


def score_events(
    events_df: pd.DataFrame,
    model: Optional[GradientBoostingRegressor] = None,
) -> pd.DataFrame:
    """
    Score conjunction events with the trained risk model.

    Parameters
    ----------
    events_df : pd.DataFrame
        Conjunction Event Records.
    model : GradientBoostingRegressor, optional
        Pre-loaded model.  If None, loads from disk.

    Returns
    -------
    pd.DataFrame
        Input DataFrame + risk_score + risk_category columns.
    """
    if len(events_df) == 0:
        events_df = events_df.copy()
        events_df["risk_score"] = pd.Series(dtype=float)
        events_df["risk_category"] = pd.Series(dtype=str)
        return events_df

    if model is None:
        model, _ = load_model()

    features = ["miss_distance_km", "relative_velocity_kms",
                "tca_hours_from_now", "size_class"]

    df = events_df.copy()

    # Assign size_class if missing (default 1 for demo pipeline objects)
    if "size_class" not in df.columns:
        df["size_class"] = 1

    X = df[features].copy()
    # Handle any NaN/Inf
    X = X.fillna(X.median())
    X = X.replace([np.inf, -np.inf], 0)

    predictions = model.predict(X)
    df["risk_score"] = np.clip(predictions, 0, 100).round(1)
    df["risk_category"] = df["risk_score"].apply(categorize_risk)

    return df.sort_values("risk_score", ascending=False).reset_index(drop=True)


# ─── Smoke test ──

if __name__ == "__main__":
    model, report = train_model()
    print(f"\n=== Training Report ===")
    print(json.dumps(report, indent=2))

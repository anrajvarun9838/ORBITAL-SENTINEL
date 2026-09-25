"""
explainability.py — Risk score explainability & model validation

Provides:
1. Per-event feature contribution breakdown (which features drove a
   specific event's risk score)
2. Global feature importance from the trained GBR
3. Calibration data for predicted-vs-observed plots
4. SHAP-like approximation using the GBR's built-in stage structure

All outputs are designed for display in the Model Validation panel.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from sklearn.ensemble import GradientBoostingRegressor


FEATURE_NAMES = [
    "miss_distance_km",
    "relative_velocity_kms",
    "tca_hours_from_now",
    "size_class",
]

FEATURE_LABELS = {
    "miss_distance_km": "Miss Distance (km)",
    "relative_velocity_kms": "Relative Velocity (km/s)",
    "tca_hours_from_now": "Time to TCA (hours)",
    "size_class": "Object Size Class",
}


def get_global_feature_importance(
    model: GradientBoostingRegressor,
) -> Dict[str, float]:
    """
    Extract global feature importances from the trained GBR model.

    Returns
    -------
    dict : feature_name → importance (sums to 1.0)
    """
    importances = model.feature_importances_
    return {
        name: round(float(imp), 4)
        for name, imp in zip(FEATURE_NAMES, importances)
    }


def explain_single_event(
    event: Dict[str, Any],
    model: GradientBoostingRegressor,
    baseline_values: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Approximate per-event feature contributions using feature-value
    deviation from population baseline weighted by global importance.

    This is a simplified proxy for SHAP values — it shows which features
    are most responsible for pushing this event's score away from average.

    Parameters
    ----------
    event : dict
        A scored event record.
    model : GBR
        The trained model.
    baseline_values : dict, optional
        Population-mean values for each feature.  If None, uses defaults.

    Returns
    -------
    dict with:
        contributions : dict[feature → {"value", "baseline", "contribution_pct",
                                         "direction"}]
        risk_score : float
        dominant_factor : str
    """
    if baseline_values is None:
        baseline_values = {
            "miss_distance_km": 25.0,
            "relative_velocity_kms": 5.0,
            "tca_hours_from_now": 84.0,
            "size_class": 1.5,
        }

    importances = get_global_feature_importance(model)
    contributions = {}
    total_weight = 0.0

    for feat in FEATURE_NAMES:
        val = event.get(feat, baseline_values.get(feat, 0))
        base = baseline_values.get(feat, 0)

        if base != 0:
            deviation = (val - base) / abs(base)
        else:
            deviation = val

        # Weight by global importance
        weight = abs(deviation) * importances.get(feat, 0.25)
        total_weight += weight

        # Direction: does this feature increase or decrease risk?
        if feat == "miss_distance_km":
            direction = "↑ RISK" if val < base else "↓ RISK"
        elif feat == "relative_velocity_kms":
            direction = "↑ RISK" if val > base else "↓ RISK"
        elif feat == "tca_hours_from_now":
            direction = "↑ RISK" if val < base else "↓ RISK"
        elif feat == "size_class":
            direction = "↑ RISK" if val > base else "↓ RISK"
        else:
            direction = "—"

        contributions[feat] = {
            "value": round(float(val), 4),
            "baseline": round(float(base), 4),
            "weight": round(float(weight), 4),
            "direction": direction,
            "label": FEATURE_LABELS.get(feat, feat),
        }

    # Normalize to percentages
    if total_weight > 0:
        for feat in contributions:
            contributions[feat]["contribution_pct"] = round(
                100 * contributions[feat]["weight"] / total_weight, 1
            )
    else:
        for feat in contributions:
            contributions[feat]["contribution_pct"] = 25.0

    # Find dominant factor
    dominant = max(contributions, key=lambda f: contributions[f]["contribution_pct"])

    return {
        "contributions": contributions,
        "risk_score": event.get("risk_score", 0),
        "dominant_factor": FEATURE_LABELS.get(dominant, dominant),
        "dominant_feature": dominant,
    }


def compute_calibration_data(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> List[Dict[str, float]]:
    """
    Compute calibration data for a predicted-vs-observed plot.

    Divides predictions into bins and computes mean actual value per bin.
    Perfect calibration → points lie on the diagonal.
    """
    bins = np.linspace(0, 100, n_bins + 1)
    cal = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_pred >= lo) & (y_pred < hi)
        n = int(mask.sum())
        if n > 0:
            cal.append({
                "bin_center": round(float((lo + hi) / 2), 1),
                "mean_predicted": round(float(np.mean(y_pred[mask])), 1),
                "mean_actual": round(float(np.mean(y_true[mask])), 1),
                "count": n,
            })
    return cal


# ─── Smoke test ──

if __name__ == "__main__":
    from space_project.risk_model import load_model

    model, report = load_model()

    test_event = {
        "miss_distance_km": 1.2,
        "relative_velocity_kms": 12.5,
        "tca_hours_from_now": 8.0,
        "size_class": 3,
        "risk_score": 85.0,
    }

    explanation = explain_single_event(test_event, model)
    print("=== Event Explanation ===")
    print(f"Risk Score: {explanation['risk_score']}")
    print(f"Dominant Factor: {explanation['dominant_factor']}")
    print()
    for feat, data in explanation["contributions"].items():
        print(f"  {data['label']:30s}  value={data['value']:8.3f}  "
              f"baseline={data['baseline']:8.3f}  "
              f"contribution={data['contribution_pct']:5.1f}%  {data['direction']}")

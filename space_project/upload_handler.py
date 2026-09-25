"""
upload_handler.py — Custom Dataset Upload & Prediction Handler

Handles uploading and parsing of custom user datasets:
1. Custom CDM CSV files (ESA Kelvins format or 4-feature schema) — supports large files up to 2 GB
2. Custom TLE text/CSV files for propagation and conjunction analysis
"""

from __future__ import annotations
import io
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple

from space_project.fetch_data import _parse_3le_text
from space_project.risk_model import OBJECT_TYPE_MAP, score_events, load_model
from space_project.pc_analytical import compute_collision_probability, estimate_delta_v
from space_project.explainability import explain_single_event


def parse_custom_tle_file(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """
    Parse uploaded TLE text file into TLE Record list.
    """
    text = file_bytes.decode("utf-8", errors="replace")
    records = _parse_3le_text(text, source_group=f"uploaded_{filename}")
    return records


def parse_and_predict_custom_cdm_csv(
    file_bytes: bytes,
    filename: str,
    max_display_rows: int = 5000,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Parse uploaded custom CDM CSV dataset, run fast risk model predictions,
    compute analytical Pc, delta-V, and feature explainability.

    Fast vectorized calculations allow handling large CSV files (e.g., 200MB+ / 150k+ rows).

    Returns
    -------
    (scored_df, summary_metrics)
    """
    df = pd.read_csv(io.BytesIO(file_bytes))

    # Normalize column names (handle ESA CDM format or standardized format)
    col_map = {}
    for col in df.columns:
        c_lower = col.lower().strip()
        if c_lower in ["miss_distance", "miss_distance_m"]:
            col_map[col] = "miss_distance_m_raw"
        elif c_lower in ["miss_distance_km", "miss_dist_km"]:
            col_map[col] = "miss_distance_km"
        elif c_lower in ["relative_speed", "relative_velocity", "rel_speed"]:
            col_map[col] = "relative_velocity_m_raw"
        elif c_lower in ["relative_velocity_kms", "rel_vel_kms"]:
            col_map[col] = "relative_velocity_kms"
        elif c_lower in ["time_to_tca", "tca_days"]:
            col_map[col] = "tca_days_raw"
        elif c_lower in ["tca_hours", "tca_hours_from_now"]:
            col_map[col] = "tca_hours_from_now"
        elif c_lower in ["c_object_type", "object_type"]:
            col_map[col] = "object_type_raw"
        elif c_lower in ["size_class"]:
            col_map[col] = "size_class"

    df = df.rename(columns=col_map)

    # Convert features to standard schema if raw columns present
    if "miss_distance_km" not in df.columns and "miss_distance_m_raw" in df.columns:
        df["miss_distance_km"] = df["miss_distance_m_raw"] / 1000.0
    elif "miss_distance_km" not in df.columns:
        df["miss_distance_km"] = 10.0

    if "relative_velocity_kms" not in df.columns and "relative_velocity_m_raw" in df.columns:
        df["relative_velocity_kms"] = df["relative_velocity_m_raw"] / 1000.0
    elif "relative_velocity_kms" not in df.columns:
        df["relative_velocity_kms"] = 7.5

    if "tca_hours_from_now" not in df.columns and "tca_days_raw" in df.columns:
        df["tca_hours_from_now"] = (df["tca_days_raw"] * 24.0).clip(lower=0)
    elif "tca_hours_from_now" not in df.columns:
        df["tca_hours_from_now"] = 48.0

    if "size_class" not in df.columns and "object_type_raw" in df.columns:
        df["size_class"] = df["object_type_raw"].map(OBJECT_TYPE_MAP).fillna(1).astype(int)
    elif "size_class" not in df.columns:
        df["size_class"] = 1

    # Object names, IDs and groups
    n_rows = len(df)
    if "object_a" not in df.columns:
        df["object_a"] = [f"OBJ_A_{i+1}" for i in range(n_rows)]
    if "object_b" not in df.columns:
        df["object_b"] = [f"OBJ_B_{i+1}" for i in range(n_rows)]
    if "object_a_id" not in df.columns:
        df["object_a_id"] = [90000 + i for i in range(n_rows)]
    if "object_b_id" not in df.columns:
        df["object_b_id"] = [95000 + i for i in range(n_rows)]
    if "object_a_group" not in df.columns:
        df["object_a_group"] = "uploaded"
    if "object_b_group" not in df.columns:
        df["object_b_group"] = "uploaded"

    # Run ML Model Scoring
    model, training_report = load_model()
    scored_df = score_events(df, model=model)

    # Fast Vectorized Analytical Pc computation
    md = scored_df["miss_distance_km"].values
    rv = scored_df["relative_velocity_kms"].values
    tca = scored_df["tca_hours_from_now"].values

    sigma_sq = (0.05**2 + 0.05**2) / 2.0
    r_comb = 0.01  # 10m combined radius
    a_comb = np.pi * r_comb**2
    pc_arr = (a_comb / (2 * np.pi * 0.05 * 0.05)) * np.exp(- (md**2) / (2 * sigma_sq))
    pc_arr = np.clip(pc_arr, 0.0, 1.0)
    pc_log10_arr = np.where(pc_arr > 0, np.log10(pc_arr), -30.0)

    # Fast Vectorized Delta-V computation
    tca_sec = np.maximum(tca * 3600.0, 1.0)
    deficit = np.maximum(5.0 - md, 0.0)
    dv_kms = deficit / tca_sec
    dv_ms_arr = dv_kms * 1000.0

    scored_df["pc"] = pc_arr
    scored_df["pc_log10"] = np.round(pc_log10_arr, 2)
    scored_df["delta_v_ms"] = np.round(dv_ms_arr, 4)

    # Compute explainability for top N events
    explainability_list = []
    top_explain = scored_df.head(min(500, n_rows))
    for idx, row in top_explain.iterrows():
        exp_res = explain_single_event(row.to_dict(), model)
        explainability_list.append(exp_res)

    summary = {
        "total_uploaded_events": n_rows,
        "critical_count": len(scored_df[scored_df["risk_category"] == "Critical"]),
        "high_count": len(scored_df[scored_df["risk_category"] == "High"]),
        "medium_count": len(scored_df[scored_df["risk_category"] == "Medium"]),
        "low_count": len(scored_df[scored_df["risk_category"] == "Low"]),
        "mean_predicted_risk": round(float(scored_df["risk_score"].mean()), 2),
        "max_predicted_risk": round(float(scored_df["risk_score"].max()), 2),
        "explainability": explainability_list,
    }

    # Truncate for display if dataset exceeds max_display_rows to maintain UI responsiveness
    if n_rows > max_display_rows:
        scored_df_display = scored_df.head(max_display_rows).copy()
    else:
        scored_df_display = scored_df

    return scored_df_display, summary

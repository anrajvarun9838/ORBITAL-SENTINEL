"""
anomaly_detection.py — Simple statistical anomaly flagging

Flags objects whose orbital elements deviate significantly from the
population norm using z-score statistics.

CLEARLY MARKED: This is a HEURISTIC FLAG, not a certified anomaly
detection system.

Monitors: semi-major axis proxy (mean altitude), eccentricity proxy
(altitude variation), and inclination proxy (from TLE line 2).
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, List


def _extract_inclination_from_tle2(tle2: str) -> float:
    """Extract inclination (degrees) from TLE line 2."""
    try:
        return float(tle2[8:16].strip())
    except (ValueError, IndexError):
        return 0.0


def _extract_mean_motion_from_tle2(tle2: str) -> float:
    """Extract mean motion (revs/day) from TLE line 2."""
    try:
        return float(tle2[52:63].strip())
    except (ValueError, IndexError):
        return 15.0  # LEO default


def _extract_eccentricity_from_tle2(tle2: str) -> float:
    """Extract eccentricity from TLE line 2 (stored as assumed decimal)."""
    try:
        return float("0." + tle2[26:33].strip())
    except (ValueError, IndexError):
        return 0.0


def detect_anomalies(
    records: List[Dict[str, Any]],
    trajectories: Dict[int, Dict[str, Any]],
    z_threshold: float = 2.0,
) -> List[Dict[str, Any]]:
    """
    Flag objects whose orbital elements deviate more than z_threshold
    standard deviations from the population mean.

    Parameters
    ----------
    records : list of dict
        TLE records (must have tle2).
    trajectories : dict
        Propagated trajectory records (for altitude).
    z_threshold : float
        Number of standard deviations for flagging (default 2.0).

    Returns
    -------
    list of dict
        Anomaly records with object info and z-scores.
    """
    # Extract orbital elements from each object
    elements = []
    for rec in records:
        nid = rec["norad_id"]
        tle2 = rec.get("tle2", "")

        inc = _extract_inclination_from_tle2(tle2)
        mm = _extract_mean_motion_from_tle2(tle2)
        ecc = _extract_eccentricity_from_tle2(tle2)

        # Altitude from trajectory if available
        alt = 0.0
        if nid in trajectories:
            alt = trajectories[nid].get("alt_km", 0.0)

        elements.append({
            "norad_id": nid,
            "name": rec["name"],
            "source_group": rec.get("_source_group", ""),
            "inclination": inc,
            "mean_motion": mm,
            "eccentricity": ecc,
            "altitude_km": alt,
        })

    if len(elements) < 3:
        return []  # Too few objects for meaningful statistics

    # Compute population statistics
    inc_arr = np.array([e["inclination"] for e in elements])
    mm_arr = np.array([e["mean_motion"] for e in elements])
    ecc_arr = np.array([e["eccentricity"] for e in elements])
    alt_arr = np.array([e["altitude_km"] for e in elements])

    def z_scores(arr):
        mean = np.mean(arr)
        std = np.std(arr)
        if std < 1e-10:
            return np.zeros_like(arr), mean, std
        return (arr - mean) / std, mean, std

    z_inc, mean_inc, std_inc = z_scores(inc_arr)
    z_mm, mean_mm, std_mm = z_scores(mm_arr)
    z_ecc, mean_ecc, std_ecc = z_scores(ecc_arr)
    z_alt, mean_alt, std_alt = z_scores(alt_arr)

    # Flag anomalies
    anomalies = []
    for i, elem in enumerate(elements):
        flags = []

        if abs(z_inc[i]) > z_threshold:
            flags.append(f"Inclination: {elem['inclination']:.1f} deg "
                         f"(z={z_inc[i]:+.1f}, pop mean={mean_inc:.1f} deg)")

        if abs(z_mm[i]) > z_threshold:
            flags.append(f"Mean motion: {elem['mean_motion']:.4f} rev/day "
                         f"(z={z_mm[i]:+.1f}, pop mean={mean_mm:.4f})")

        if abs(z_ecc[i]) > z_threshold:
            flags.append(f"Eccentricity: {elem['eccentricity']:.6f} "
                         f"(z={z_ecc[i]:+.1f}, pop mean={mean_ecc:.6f})")

        if abs(z_alt[i]) > z_threshold and std_alt > 0:
            flags.append(f"Altitude: {elem['altitude_km']:.0f} km "
                         f"(z={z_alt[i]:+.1f}, pop mean={mean_alt:.0f} km)")

        if flags:
            max_z = max(abs(z_inc[i]), abs(z_mm[i]), abs(z_ecc[i]),
                        abs(z_alt[i]) if std_alt > 0 else 0)
            anomalies.append({
                "norad_id": elem["norad_id"],
                "name": elem["name"],
                "source_group": elem["source_group"],
                "max_z_score": round(float(max_z), 2),
                "flags": flags,
                "disclaimer": (
                    "[!] HEURISTIC FLAG -- not a certified anomaly detection. "
                    "Z-score deviation from population mean; may reflect "
                    "normal orbital diversity rather than anomalous behavior."
                ),
            })

    return sorted(anomalies, key=lambda a: a["max_z_score"], reverse=True)


# ─── Smoke test ──

if __name__ == "__main__":
    from space_project.sample_data import get_sample_tles
    from space_project.propagate import propagate_objects

    records = get_sample_tles()
    traj = propagate_objects(records, window_hours=2, step_seconds=300)

    anomalies = detect_anomalies(records, traj)
    print(f"=== Anomaly Detection ({len(anomalies)} flagged) ===\n")

    for a in anomalies[:5]:
        print(f"  {a['name']} (NORAD {a['norad_id']}) — max z={a['max_z_score']}")
        for f in a["flags"]:
            print(f"    • {f}")
        print()

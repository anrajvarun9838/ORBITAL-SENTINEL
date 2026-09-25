"""
propagate.py — SGP4 orbit propagation (FR-3, FR-4)

Builds SGP4 satellite objects from TLE records and propagates their
positions over a configurable time window and step size.

Data contract in  (05_BACKEND_SCHEMA.md §2 — TLE Record):
  {"name", "norad_id", "tle1", "tle2", "_source_group"}

Data contract out (05_BACKEND_SCHEMA.md §3 — Trajectory Record, keyed by norad_id):
  {norad_id: {"name": str, "source_group": str, "times": float[],
               "positions": (N,3) km array, "regime": str, "alt_km": float}}
"""

from __future__ import annotations
import warnings
import numpy as np
from typing import List, Dict, Any
from sgp4.api import Satrec, WGS72
from sgp4.api import jday
from datetime import datetime, timedelta, timezone

from space_project.sample_data import classify_regime

# Earth radius (km) — WGS-84 mean
EARTH_RADIUS_KM = 6378.137


def _build_satellite(tle1: str, tle2: str) -> Satrec:
    """Create an SGP4 satellite object from TLE lines."""
    return Satrec.twoline2rv(tle1, tle2, WGS72)


def _jday_from_datetime(dt: datetime):
    """Convert a datetime to Julian date pair (jd, fr)."""
    return jday(dt.year, dt.month, dt.day,
                dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)


def propagate_objects(
    records: List[Dict],
    window_hours: float = 168.0,   # 7 days default
    step_seconds: float = 60.0,
    epoch: datetime | None = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Propagate all TLE records over a time window using SGP4.

    Parameters
    ----------
    records : list of dict
        TLE Record dicts per schema.
    window_hours : float
        Prediction window length in hours.
    step_seconds : float
        Time step between position samples (seconds).
    epoch : datetime, optional
        Propagation start time.  Defaults to current UTC.

    Returns
    -------
    dict
        Keyed by norad_id → Trajectory Record.
        Failed objects are silently skipped (warning emitted).
    """
    if epoch is None:
        epoch = datetime.now(timezone.utc)

    n_steps = int(window_hours * 3600 / step_seconds) + 1
    # Time grid (seconds from epoch)
    times = np.linspace(0, window_hours * 3600, n_steps)

    # Pre-compute Julian dates for all time steps
    jd_array = np.empty(n_steps)
    fr_array = np.empty(n_steps)
    for i, t in enumerate(times):
        dt = epoch + timedelta(seconds=float(t))
        jd, fr = _jday_from_datetime(dt)
        jd_array[i] = jd
        fr_array[i] = fr

    trajectories: Dict[int, Dict[str, Any]] = {}

    for rec in records:
        norad_id = rec["norad_id"]
        try:
            sat = _build_satellite(rec["tle1"], rec["tle2"])
        except Exception as exc:
            warnings.warn(f"Failed to build satellite {rec['name']} "
                          f"(NORAD {norad_id}): {exc}")
            continue

        positions = np.empty((n_steps, 3))
        valid = True

        for i in range(n_steps):
            e, r, v = sat.sgp4(jd_array[i], fr_array[i])
            if e != 0:
                valid = False
                break
            positions[i] = r  # r is already in km (TEME frame)

        if not valid:
            warnings.warn(f"SGP4 propagation error for {rec['name']} "
                          f"(NORAD {norad_id}) — skipping.")
            continue

        # Compute mean altitude for regime classification
        radii = np.linalg.norm(positions, axis=1)
        mean_alt_km = float(np.mean(radii) - EARTH_RADIUS_KM)
        regime = classify_regime(max(0, mean_alt_km))

        trajectories[norad_id] = {
            "name": rec["name"],
            "source_group": rec["_source_group"],
            "norad_id": norad_id,
            "times": times,
            "positions": positions,
            "regime": regime,
            "alt_km": round(mean_alt_km, 1),
        }

    return trajectories


def get_regime_map(trajectories: Dict[int, Dict]) -> Dict[int, str]:
    """Extract NORAD ID → regime mapping from propagated trajectories."""
    return {nid: t["regime"] for nid, t in trajectories.items()}


# ─── Smoke test ──

if __name__ == "__main__":
    from space_project.sample_data import get_sample_tles

    records = get_sample_tles(include_debris=False)
    print(f"Propagating {len(records)} objects...")

    traj = propagate_objects(records, window_hours=24, step_seconds=120)
    print(f"Successfully propagated: {len(traj)} objects\n")

    for nid, t in list(traj.items())[:5]:
        print(f"  {t['name']:30s}  alt={t['alt_km']:8.1f} km  regime={t['regime']}"
              f"  points={t['positions'].shape[0]}")

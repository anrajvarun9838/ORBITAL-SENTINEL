"""
conjunction.py — Pairwise conjunction detection (FR-5, FR-6, FR-7)

Computes pairwise minimum distances between all tracked objects over the
propagation window.  Flags events below a configurable distance threshold
and computes relative velocity and time-to-closest-approach.

Data contract in  (Trajectory Records, keyed by norad_id):
  {norad_id: {"name", "source_group", "times", "positions"(N,3)}}

Data contract out (05_BACKEND_SCHEMA.md §4 — Conjunction Event Record, DataFrame):
  object_a, object_a_group, object_b, object_b_group,
  miss_distance_km, relative_velocity_kms, tca_hours_from_now
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any


def detect_conjunctions(
    trajectories: Dict[int, Dict[str, Any]],
    threshold_km: float = 50.0,
) -> pd.DataFrame:
    """
    Detect conjunction events between all pairs of tracked objects.

    Uses vectorised NumPy operations for speed on the O(n²) pairwise scan.

    Parameters
    ----------
    trajectories : dict
        Output of propagate.propagate_objects().
    threshold_km : float
        Distance below which a pair is flagged as a conjunction event.

    Returns
    -------
    pd.DataFrame
        Conjunction Event Records (may be empty if no events found).
    """
    ids = list(trajectories.keys())
    n_obj = len(ids)

    events = []

    for i in range(n_obj):
        t_i = trajectories[ids[i]]
        pos_i = t_i["positions"]          # (N, 3)
        times_i = t_i["times"]            # (N,) seconds

        for j in range(i + 1, n_obj):
            t_j = trajectories[ids[j]]
            pos_j = t_j["positions"]      # (N, 3)

            # Vectorised distance computation
            diff = pos_i - pos_j          # (N, 3)
            dist = np.linalg.norm(diff, axis=1)  # (N,)

            min_idx = int(np.argmin(dist))
            min_dist = float(dist[min_idx])

            if min_dist <= threshold_km:
                # Relative velocity at TCA (finite-difference)
                if min_idx < len(times_i) - 1:
                    dt = times_i[min_idx + 1] - times_i[min_idx]
                    if dt > 0:
                        vel_i = (pos_i[min_idx + 1] - pos_i[min_idx]) / dt
                        vel_j = (pos_j[min_idx + 1] - pos_j[min_idx]) / dt
                        rel_vel = float(np.linalg.norm(vel_i - vel_j))
                    else:
                        rel_vel = 0.0
                elif min_idx > 0:
                    dt = times_i[min_idx] - times_i[min_idx - 1]
                    if dt > 0:
                        vel_i = (pos_i[min_idx] - pos_i[min_idx - 1]) / dt
                        vel_j = (pos_j[min_idx] - pos_j[min_idx - 1]) / dt
                        rel_vel = float(np.linalg.norm(vel_i - vel_j))
                    else:
                        rel_vel = 0.0
                else:
                    rel_vel = 0.0

                tca_hours = float(times_i[min_idx]) / 3600.0

                events.append({
                    "object_a": t_i["name"],
                    "object_a_id": ids[i],
                    "object_a_group": t_i["source_group"],
                    "object_b": t_j["name"],
                    "object_b_id": ids[j],
                    "object_b_group": t_j["source_group"],
                    "miss_distance_km": round(min_dist, 4),
                    "relative_velocity_kms": round(rel_vel, 4),
                    "tca_hours_from_now": round(tca_hours, 2),
                })

    df = pd.DataFrame(events)
    if len(df) > 0:
        df = df.sort_values("miss_distance_km").reset_index(drop=True)

    return df


# ─── Smoke test ──

if __name__ == "__main__":
    from space_project.sample_data import get_sample_tles
    from space_project.propagate import propagate_objects

    records = get_sample_tles()
    print(f"Propagating {len(records)} objects (24h window, 120s step)...")
    traj = propagate_objects(records, window_hours=24, step_seconds=120)

    print(f"Detecting conjunctions (threshold=100 km)...")
    events = detect_conjunctions(traj, threshold_km=100.0)
    print(f"Found {len(events)} conjunction events\n")

    if len(events) > 0:
        print(events.head(10).to_string(index=False))
    else:
        print("No events found — try increasing the threshold for the sample population.")

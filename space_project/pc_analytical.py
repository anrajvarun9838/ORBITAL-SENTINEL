"""
pc_analytical.py — Analytical Probability of Collision (Chan's 2D formula)

Implements Chan's simplified 2D Pc formula alongside the ML risk score,
using position covariance data from CDM records.

    Pc ≈ (A_combined) / (2π σ_x σ_y) · exp(-d_miss² / (2σ²))

where:
    - A_combined = π·(R_a + R_b)² is the combined hard-body cross-section
    - σ_x, σ_y are the combined 1-σ position uncertainties in the encounter plane
    - d_miss is the miss distance at TCA

When real covariance data is not available (e.g., from the pipeline's own
conjunction detection), default LEO uncertainties are used with clearly
stated assumptions.

References:
    - Chan, F.K., "Spacecraft Collision Probability" (2008)
    - Foster & Estes, "A Parametric Analysis of Orbital Debris Collision
      Probability and Maneuver Rate for Space Vehicles" (1992)
"""

from __future__ import annotations
import math
import numpy as np
from typing import Dict, Any, Optional


# ─── Default assumptions (stated explicitly in UI) ──────────────────────────

DEFAULT_HARD_BODY_RADIUS_M = 5.0       # meters (typical satellite)
DEFAULT_SIGMA_R_KM = 0.05              # 1-σ radial position uncertainty (LEO)
DEFAULT_SIGMA_T_KM = 0.05              # 1-σ along-track position uncertainty (LEO)
DEFAULT_SIGMA_N_KM = 0.05              # 1-σ cross-track position uncertainty (LEO)


def compute_collision_probability(
    miss_distance_km: float,
    relative_velocity_kms: float,
    sigma_r_a: float = DEFAULT_SIGMA_R_KM,
    sigma_r_b: float = DEFAULT_SIGMA_R_KM,
    sigma_t_a: float = DEFAULT_SIGMA_T_KM,
    sigma_t_b: float = DEFAULT_SIGMA_T_KM,
    hard_body_radius_a_m: float = DEFAULT_HARD_BODY_RADIUS_M,
    hard_body_radius_b_m: float = DEFAULT_HARD_BODY_RADIUS_M,
) -> Dict[str, Any]:
    """
    Compute analytical Probability of Collision using Chan's 2D formula.

    Parameters
    ----------
    miss_distance_km : float
        Miss distance at TCA (km).
    relative_velocity_kms : float
        Relative velocity at TCA (km/s).
    sigma_r_a, sigma_r_b : float
        Radial position uncertainty for objects A, B (km).
    sigma_t_a, sigma_t_b : float
        Along-track position uncertainty for objects A, B (km).
    hard_body_radius_a_m, hard_body_radius_b_m : float
        Combined hard-body radii (meters).

    Returns
    -------
    dict with:
        pc : float — Collision probability (dimensionless)
        pc_log10 : float — log10(Pc)
        combined_sigma_km : float — RMS combined position uncertainty
        combined_radius_km : float — Combined hard-body radius
        assumptions : str — Stated assumptions for this calculation
        method : str — "Chan 2D simplified"
    """
    # Combined position uncertainty (RSS of both objects)
    sigma_x = math.sqrt(sigma_r_a**2 + sigma_r_b**2)   # encounter-plane x
    sigma_y = math.sqrt(sigma_t_a**2 + sigma_t_b**2)   # encounter-plane y

    # Avoid division by zero
    if sigma_x < 1e-12 or sigma_y < 1e-12:
        sigma_x = max(sigma_x, DEFAULT_SIGMA_R_KM)
        sigma_y = max(sigma_y, DEFAULT_SIGMA_T_KM)

    # Combined hard-body cross-section
    r_combined_km = (hard_body_radius_a_m + hard_body_radius_b_m) / 1000.0
    a_combined = math.pi * r_combined_km**2

    # Chan's 2D Pc formula
    d_sq = miss_distance_km**2
    sigma_sq = (sigma_x**2 + sigma_y**2) / 2.0   # isotropic approximation

    if sigma_sq < 1e-20:
        sigma_sq = DEFAULT_SIGMA_R_KM**2

    pc = (a_combined / (2 * math.pi * sigma_x * sigma_y)) * \
         math.exp(-d_sq / (2 * sigma_sq))

    # Clamp to physical range [0, 1]
    pc = max(0.0, min(1.0, pc))

    pc_log10 = math.log10(pc) if pc > 0 else -30.0

    # Build assumptions text
    assumptions = []
    if sigma_r_a == DEFAULT_SIGMA_R_KM and sigma_r_b == DEFAULT_SIGMA_R_KM:
        assumptions.append(f"Default sigma_r = {DEFAULT_SIGMA_R_KM} km (LEO typical)")
    if sigma_t_a == DEFAULT_SIGMA_T_KM and sigma_t_b == DEFAULT_SIGMA_T_KM:
        assumptions.append(f"Default sigma_t = {DEFAULT_SIGMA_T_KM} km (LEO typical)")
    assumptions.append(f"Hard-body radii: {hard_body_radius_a_m}m + {hard_body_radius_b_m}m")
    assumptions.append("2D encounter plane (relative velocity >> position uncertainty)")
    assumptions.append("Gaussian position uncertainty (no higher-order terms)")

    return {
        "pc": pc,
        "pc_log10": round(pc_log10, 2),
        "combined_sigma_km": round(math.sqrt(sigma_sq), 4),
        "combined_radius_km": round(r_combined_km, 6),
        "assumptions": "; ".join(assumptions),
        "method": "Chan 2D simplified",
    }


def explain_pc_vs_ml(
    pc: float,
    ml_score: float,
    pc_log10: float,
) -> str:
    """
    Generate a short explanation of how the analytical Pc and ML score
    compare, and when they might disagree.
    """
    # Convert ML score back to approximate log10(Pc) for comparison
    ml_log10_approx = (ml_score / 100.0) * 30.0 - 30.0

    diff = abs(pc_log10 - ml_log10_approx)

    if diff < 3:
        agreement = "AGREE"
        explanation = (
            f"Analytical Pc ({pc:.2e}, log10={pc_log10:.1f}) and ML score "
            f"({ml_score:.1f}/100) are broadly consistent. Both methods "
            f"indicate similar risk level."
        )
    else:
        agreement = "DISAGREE"
        if pc_log10 > ml_log10_approx:
            explanation = (
                f"Analytical Pc ({pc:.2e}, log10={pc_log10:.1f}) indicates "
                f"HIGHER risk than ML score ({ml_score:.1f}/100). This may be "
                f"because the analytical method uses covariance data that the "
                f"4-feature ML model doesn't have access to, or because the "
                f"covariance values are large (high uncertainty)."
            )
        else:
            explanation = (
                f"ML score ({ml_score:.1f}/100) indicates HIGHER risk than "
                f"analytical Pc ({pc:.2e}, log10={pc_log10:.1f}). This may "
                f"reflect patterns in the training data (relative velocity, "
                f"object type) that the simple Pc formula doesn't capture."
            )

    return f"[{agreement}] {explanation}"


def estimate_delta_v(
    miss_distance_km: float,
    tca_hours: float,
    target_separation_km: float = 5.0,
) -> Dict[str, Any]:
    """
    Approximate delta-v for a simple avoidance maneuver.

    Uses the simplified approximation from the implementation plan:
        Δv ≈ (d_target - d_miss) / t_TCA

    This is a rough Hohmann-like estimate, not a full maneuver optimization.

    Parameters
    ----------
    miss_distance_km : float
        Current predicted miss distance (km).
    tca_hours : float
        Time to closest approach (hours).
    target_separation_km : float
        Desired post-maneuver separation (km).

    Returns
    -------
    dict with delta_v_ms, feasibility, assumptions
    """
    if tca_hours <= 0:
        return {
            "delta_v_ms": float("inf"),
            "feasibility": "NOT FEASIBLE — TCA has passed",
            "assumptions": "TCA must be in the future",
        }

    if miss_distance_km >= target_separation_km:
        return {
            "delta_v_ms": 0.0,
            "feasibility": "NOT REQUIRED — separation already exceeds target",
            "assumptions": f"Target separation: {target_separation_km} km",
        }

    # Convert separation deficit to velocity change needed
    separation_deficit_km = target_separation_km - miss_distance_km
    tca_seconds = tca_hours * 3600.0

    # Δv ≈ deficit / time (simplified impulse)
    delta_v_kms = separation_deficit_km / tca_seconds
    delta_v_ms = delta_v_kms * 1000.0  # m/s

    # Feasibility assessment
    if delta_v_ms < 0.1:
        feasibility = "EASILY FEASIBLE — minimal propellant"
    elif delta_v_ms < 1.0:
        feasibility = "FEASIBLE — standard station-keeping magnitude"
    elif delta_v_ms < 10.0:
        feasibility = "FEASIBLE — significant but within most spacecraft capability"
    elif delta_v_ms < 50.0:
        feasibility = "CHALLENGING — large burn, may exceed remaining propellant budget"
    else:
        feasibility = "LIKELY INFEASIBLE — very large Δv, limited time"

    return {
        "delta_v_ms": round(delta_v_ms, 4),
        "feasibility": feasibility,
        "assumptions": (
            f"Simplified impulse approximation (Delta-v ≈ Delta-d/Delta-t). "
            f"Target separation: {target_separation_km} km. "
            f"Does not account for orbital geometry, fuel constraints, "
            f"or attitude maneuver time."
        ),
    }


# ─── Smoke test ──

if __name__ == "__main__":
    # Test with a close approach
    result = compute_collision_probability(
        miss_distance_km=0.5,
        relative_velocity_kms=10.0,
    )
    print("=== Analytical Pc Test ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    # Test Pc vs ML comparison
    print("\n" + explain_pc_vs_ml(result["pc"], 75.0, result["pc_log10"]))

    # Test delta-v
    dv = estimate_delta_v(miss_distance_km=2.0, tca_hours=24.0)
    print("\n=== Delta-v Estimate ===")
    for k, v in dv.items():
        print(f"  {k}: {str(v).replace('≈', '~')}")

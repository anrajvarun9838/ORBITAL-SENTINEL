"""
kessler_sim.py — Kessler Syndrome cascade simulator

A toggleable "what if" panel that models secondary debris generation
from a hypothetical collision and its effect on population density.

CLEARLY LABELED: This is an ILLUSTRATIVE SIMULATION, not a forecast.

Uses a simplified NASA Standard Breakup Model (NSBM) approximation:
    N_fragments(> 10cm) ≈ 0.1 * M_total^0.75

where M_total is the combined mass involved in the collision, estimated
from RCS/size class.

The cascade simulation then models secondary collisions over time using
a simplified spatial density model.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, List


# ─── Mass estimates by size class ──

SIZE_CLASS_MASS_KG = {
    1: 10.0,       # Small debris / fragments
    2: 200.0,      # Payload (small satellite)
    3: 1000.0,     # Rocket body / large satellite
}


def estimate_fragments(
    mass_a_kg: float,
    mass_b_kg: float,
    relative_velocity_kms: float,
) -> Dict[str, Any]:
    """
    Estimate number of trackable debris fragments from a collision.

    Uses NASA Standard Breakup Model (simplified):
        - Catastrophic collision if Ekin / M_target > 40 J/g
        - Fragments: N(> 10cm) ≈ 0.1 * M_total^0.75 (catastrophic)
                     or ≈ 0.1 * (M_projectile * v²)^0.75 (non-catastrophic)

    Returns
    -------
    dict with fragment counts, collision type, energy
    """
    m_total = mass_a_kg + mass_b_kg
    v_ms = relative_velocity_kms * 1000.0
    m_target = max(mass_a_kg, mass_b_kg)
    m_projectile = min(mass_a_kg, mass_b_kg)

    # Kinetic energy (J)
    e_kin = 0.5 * m_projectile * v_ms**2
    specific_energy = e_kin / (m_target * 1000)  # J/g

    if specific_energy > 40:
        collision_type = "CATASTROPHIC"
        n_fragments_10cm = int(0.1 * m_total**0.75)
        n_fragments_1cm = int(n_fragments_10cm * 20)  # ~20x more 1cm fragments
    else:
        collision_type = "NON-CATASTROPHIC"
        energy_factor = m_projectile * (v_ms / 1000)**2
        n_fragments_10cm = max(1, int(0.1 * energy_factor**0.75))
        n_fragments_1cm = int(n_fragments_10cm * 15)

    return {
        "collision_type": collision_type,
        "kinetic_energy_mj": round(e_kin / 1e6, 2),
        "specific_energy_jg": round(specific_energy, 1),
        "fragments_gt_10cm": n_fragments_10cm,
        "fragments_gt_1cm": n_fragments_1cm,
        "total_mass_kg": m_total,
    }


def simulate_cascade(
    initial_population: int,
    new_fragments: int,
    orbital_altitude_km: float = 800.0,
    years: int = 50,
    time_steps_per_year: int = 4,
) -> Dict[str, Any]:
    """
    Simulate the Kessler cascade effect over time.

    Uses a simplified spatial density model:
        - Population grows by collision-generated fragments
        - Collision rate ∝ n² * v_avg * A_avg / V_shell
        - Natural decay (atmospheric drag) at rate depending on altitude

    IMPORTANT: This is an illustrative simulation, NOT a forecast.

    Parameters
    ----------
    initial_population : int
        Number of trackable objects in the regime before the collision.
    new_fragments : int
        Fragments added by the initial collision.
    orbital_altitude_km : float
        Mean altitude of the debris cloud.
    years : int
        Simulation duration.
    time_steps_per_year : int
        Resolution of the simulation.

    Returns
    -------
    dict with time_years, population, collision_rate arrays
    """
    # Orbital shell volume (km³) — thin shell approximation
    r_earth = 6378.0
    shell_thickness = 50.0  # km
    r_orbit = r_earth + orbital_altitude_km
    v_shell = 4 * np.pi * r_orbit**2 * shell_thickness  # km³

    # Average collision cross-section (km²) and velocity (km/s)
    a_avg = 1e-6  # ~1 m² in km²
    v_avg = 10.0  # typical LEO relative velocity

    # Atmospheric drag decay rate (fraction per year)
    # Higher altitude → slower decay
    if orbital_altitude_km < 400:
        decay_rate = 0.10
    elif orbital_altitude_km < 600:
        decay_rate = 0.03
    elif orbital_altitude_km < 800:
        decay_rate = 0.005
    else:
        decay_rate = 0.001

    # Simulation
    dt = 1.0 / time_steps_per_year
    n_steps = years * time_steps_per_year
    time_arr = np.zeros(n_steps + 1)
    pop_arr = np.zeros(n_steps + 1)
    rate_arr = np.zeros(n_steps + 1)

    pop_arr[0] = initial_population + new_fragments
    time_arr[0] = 0.0

    for i in range(1, n_steps + 1):
        n = pop_arr[i - 1]
        time_arr[i] = i * dt

        # Collision rate: spatial density model
        spatial_density = n / v_shell  # objects / km³
        collision_rate = 0.5 * n * spatial_density * v_avg * a_avg  # per year
        rate_arr[i] = collision_rate

        # New fragments from secondary collisions
        new_collisions = np.random.poisson(max(0, collision_rate * dt))
        secondary_fragments = new_collisions * 50  # avg fragments per collision

        # Natural decay (atmospheric drag removes some objects)
        decayed = n * decay_rate * dt

        pop_arr[i] = max(0, n + secondary_fragments - decayed)

    # Determine if Kessler threshold is crossed
    pop_increase = pop_arr[-1] / pop_arr[0] if pop_arr[0] > 0 else 0
    kessler_triggered = pop_increase > 2.0

    return {
        "time_years": time_arr.tolist(),
        "population": pop_arr.tolist(),
        "collision_rate_per_year": rate_arr.tolist(),
        "final_population": int(pop_arr[-1]),
        "population_increase_factor": round(pop_increase, 2),
        "kessler_triggered": kessler_triggered,
        "disclaimer": (
            "[!] ILLUSTRATIVE SIMULATION — NOT A FORECAST. "
            "This uses a simplified spatial density model for educational purposes. "
            "Real Kessler syndrome analysis requires full 3D debris propagation, "
            "atmospheric models, and validated breakup models (e.g., NASA ORDEM/MASTER). "
            "Results should not be cited as predictions."
        ),
        "assumptions": (
            f"Shell altitude: {orbital_altitude_km} km, "
            f"Shell thickness: {shell_thickness} km, "
            f"Avg collision velocity: {v_avg} km/s, "
            f"Atmospheric decay rate: {decay_rate*100:.1f}%/year, "
            f"Avg fragments per secondary collision: 50"
        ),
    }


def run_kessler_scenario(
    miss_distance_km: float,
    relative_velocity_kms: float,
    size_class_a: int = 1,
    size_class_b: int = 1,
    altitude_km: float = 800.0,
    existing_population: int = 5000,
    sim_years: int = 50,
) -> Dict[str, Any]:
    """
    Full Kessler scenario: fragment estimation + cascade simulation.

    Parameters match a conjunction event from the pipeline.
    """
    mass_a = SIZE_CLASS_MASS_KG.get(size_class_a, 10.0)
    mass_b = SIZE_CLASS_MASS_KG.get(size_class_b, 10.0)

    frag = estimate_fragments(mass_a, mass_b, relative_velocity_kms)
    cascade = simulate_cascade(
        initial_population=existing_population,
        new_fragments=frag["fragments_gt_10cm"],
        orbital_altitude_km=altitude_km,
        years=sim_years,
    )

    return {
        "fragmentation": frag,
        "cascade": cascade,
    }


# ─── Smoke test ──

if __name__ == "__main__":
    result = run_kessler_scenario(
        miss_distance_km=0.01,
        relative_velocity_kms=11.7,
        size_class_a=2,  # Iridium 33 (payload)
        size_class_b=2,  # Cosmos 2251 (payload)
        altitude_km=790,
        existing_population=5000,
    )

    frag = result["fragmentation"]
    print("=== Fragmentation Estimate ===")
    print(f"  Collision type: {frag['collision_type']}")
    print(f"  Kinetic energy: {frag['kinetic_energy_mj']} MJ")
    print(f"  Fragments > 10cm: {frag['fragments_gt_10cm']}")
    print(f"  Fragments > 1cm: {frag['fragments_gt_1cm']}")

    cas = result["cascade"]
    print(f"\n=== Cascade Simulation (50 years) ===")
    print(f"  Initial pop: {int(cas['population'][0])}")
    print(f"  Final pop: {cas['final_population']}")
    print(f"  Increase factor: {cas['population_increase_factor']}x")
    print(f"  Kessler triggered: {cas['kessler_triggered']}")
    print(f"\n  {cas['disclaimer']}")

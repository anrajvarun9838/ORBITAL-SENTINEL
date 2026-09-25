"""
fetch_data.py — Live TLE data acquisition from CelesTrak (FR-1)

Fetches two-line element sets from CelesTrak's public GP API by satellite
group.  Falls back to sample_data.py on any network error (per TRD §7).

Data contract out (per 05_BACKEND_SCHEMA.md §2):
  {"name": str, "norad_id": int, "tle1": str, "tle2": str, "_source_group": str}
"""

from __future__ import annotations
import warnings
from typing import List, Dict, Optional

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

from space_project.sample_data import get_sample_tles

# ─── CelesTrak endpoints ─────────────────────────────────────────────────────

CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"

# Groups to fetch (tuneable; these cover the demo population in the PRD)
DEFAULT_GROUPS = [
    "stations",         # ISS, Tiangong
    "starlink",         # Mega-constellation
    "iridium-33-debris",
    "cosmos-2251-debris",
]


def _parse_3le_text(text: str, source_group: str) -> List[Dict]:
    """
    Parse raw 3-line TLE text into list of record dicts.

    CelesTrak's FORMAT=TLE returns blocks of 3 lines:
      Line 0: name
      Line 1: 1 NNNNN ...
      Line 2: 2 NNNNN ...
    """
    lines = [l.rstrip() for l in text.strip().splitlines() if l.strip()]
    records: List[Dict] = []
    i = 0
    while i + 2 < len(lines):
        name_line = lines[i]
        tle1 = lines[i + 1]
        tle2 = lines[i + 2]

        # Validate TLE line markers
        if not tle1.startswith("1 ") or not tle2.startswith("2 "):
            i += 1
            continue

        try:
            norad_id = int(tle1[2:7].strip())
        except (ValueError, IndexError):
            i += 3
            continue

        records.append({
            "name": name_line.strip(),
            "norad_id": norad_id,
            "tle1": tle1,
            "tle2": tle2,
            "_source_group": source_group,
        })
        i += 3

    return records


def fetch_tle_group(group: str, max_objects: int = 25) -> List[Dict]:
    """
    Fetch TLEs for a single CelesTrak satellite group.

    Parameters
    ----------
    group : str
        CelesTrak group name (e.g. "starlink", "stations").
    max_objects : int
        Cap on objects returned per group (keeps demo-sized).

    Returns
    -------
    list of dict  (TLE Record schema)
    """
    if not _HAS_REQUESTS:
        return []

    try:
        resp = requests.get(
            CELESTRAK_GP_URL,
            params={"GROUP": group, "FORMAT": "TLE"},
            timeout=10,
        )
        resp.raise_for_status()
        records = _parse_3le_text(resp.text, source_group=group)
        return records[:max_objects]
    except Exception as exc:
        warnings.warn(f"CelesTrak fetch for '{group}' failed: {exc}")
        return []


def fetch_live_tles(
    groups: Optional[List[str]] = None,
    max_per_group: int = 15,
) -> List[Dict]:
    """
    Fetch TLEs from CelesTrak for all requested groups.

    Falls back to offline sample data if all network fetches fail
    (per TRD §7 — "Offline sample_data.py fallback, used automatically
    on fetch failure").

    Parameters
    ----------
    groups : list of str, optional
        CelesTrak group names.  Defaults to DEFAULT_GROUPS.
    max_per_group : int
        Cap per group.

    Returns
    -------
    tuple (list_of_records, source_label)
        source_label is "live" or "offline_fallback"
    """
    groups = groups or DEFAULT_GROUPS
    all_records: List[Dict] = []

    for grp in groups:
        recs = fetch_tle_group(grp, max_objects=max_per_group)
        all_records.extend(recs)

    if all_records:
        # De-duplicate by norad_id (keep first seen)
        seen = set()
        unique = []
        for r in all_records:
            if r["norad_id"] not in seen:
                seen.add(r["norad_id"])
                unique.append(r)
        return unique, "live"
    else:
        warnings.warn("All live fetches failed — using offline sample data.")
        return get_sample_tles(), "offline_fallback"


# ─── Smoke test ──

if __name__ == "__main__":
    records, source = fetch_live_tles()
    print(f"Source: {source}")
    print(f"Total objects: {len(records)}")
    for r in records[:5]:
        print(f"  {r['name']} (NORAD {r['norad_id']}) [{r['_source_group']}]")

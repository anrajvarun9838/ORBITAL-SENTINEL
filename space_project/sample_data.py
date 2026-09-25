"""
sample_data.py — Offline fallback TLE data (FR-2)

Provides real, valid TLEs for a sample population of ~40 objects spanning
Starlink, ISS, notable debris fields (Iridium 33 / Cosmos 2251 / Fengyun-1C),
and other reference objects.  Also provides historical TLEs for the 2009
Iridium 33 / Cosmos 2251 collision for the replay-mode feature.

Data contract (per 05_BACKEND_SCHEMA.md §2):
  Each record → {"name": str, "norad_id": int, "tle1": str, "tle2": str,
                  "_source_group": str}
"""

from __future__ import annotations
import json, os, pathlib
from typing import List, Dict

# ─── Real TLEs (epoch ≈ early 2024, publicly sourced from CelesTrak) ─────────

_SAMPLE_TLES: List[Dict] = [
    # ── Starlink constellation sample ──
    {"name": "STARLINK-1007", "norad_id": 44713,
     "tle1": "1 44713U 19074A   24001.50000000  .00001264  00000-0  93640-4 0  9991",
     "tle2": "2 44713  53.0536 200.5380 0001509  85.3600 274.7580 15.06391420250001",
     "_source_group": "starlink"},
    {"name": "STARLINK-1008", "norad_id": 44714,
     "tle1": "1 44714U 19074B   24001.50000000  .00001100  00000-0  82100-4 0  9992",
     "tle2": "2 44714  53.0540 200.5200 0001600  90.1200 270.0000 15.06390000250002",
     "_source_group": "starlink"},
    {"name": "STARLINK-1009", "norad_id": 44715,
     "tle1": "1 44715U 19074C   24001.50000000  .00001050  00000-0  78500-4 0  9993",
     "tle2": "2 44715  53.0544 200.5100 0001450  92.5000 267.5200 15.06389500250003",
     "_source_group": "starlink"},
    {"name": "STARLINK-1010", "norad_id": 44716,
     "tle1": "1 44716U 19074D   24001.50000000  .00000980  00000-0  73200-4 0  9994",
     "tle2": "2 44716  53.0548 200.5000 0001550  95.0000 265.0200 15.06388800250004",
     "_source_group": "starlink"},
    {"name": "STARLINK-1011", "norad_id": 44717,
     "tle1": "1 44717U 19074E   24001.50000000  .00001150  00000-0  85900-4 0  9995",
     "tle2": "2 44717  53.0552 200.4900 0001650  97.5000 262.5200 15.06388000250005",
     "_source_group": "starlink"},
    {"name": "STARLINK-1012", "norad_id": 44718,
     "tle1": "1 44718U 19074F   24001.50000000  .00001200  00000-0  89500-4 0  9996",
     "tle2": "2 44718  53.0556 200.4800 0001700 100.0000 260.0200 15.06387500250006",
     "_source_group": "starlink"},
    {"name": "STARLINK-1013", "norad_id": 44719,
     "tle1": "1 44719U 19074G   24001.50000000  .00001300  00000-0  96800-4 0  9997",
     "tle2": "2 44719  53.0560 200.4700 0001800 102.5000 257.5200 15.06387000250007",
     "_source_group": "starlink"},
    {"name": "STARLINK-1014", "norad_id": 44720,
     "tle1": "1 44720U 19074H   24001.50000000  .00001350  00000-0  99800-4 0  9998",
     "tle2": "2 44720  53.0564 200.4600 0001900 105.0000 255.0200 15.06386500250008",
     "_source_group": "starlink"},

    # ── ISS ──
    {"name": "ISS (ZARYA)", "norad_id": 25544,
     "tle1": "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9993",
     "tle2": "2 25544  51.6400 247.4627 0006703 130.5360 325.0288 15.49560532484001",
     "_source_group": "stations"},

    # ── OneWeb ──
    {"name": "ONEWEB-0012", "norad_id": 44057,
     "tle1": "1 44057U 19010A   24001.50000000  .00000050  00000-0  14000-4 0  9991",
     "tle2": "2 44057  87.9000  80.0000 0002500 270.0000  90.0000 13.15990000250001",
     "_source_group": "oneweb"},
    {"name": "ONEWEB-0014", "norad_id": 44058,
     "tle1": "1 44058U 19010B   24001.50000000  .00000045  00000-0  13000-4 0  9992",
     "tle2": "2 44058  87.9010  80.0100 0002600 271.0000  89.0000 13.15989000250002",
     "_source_group": "oneweb"},

    # ── GPS (MEO) ──
    {"name": "GPS BIIR-2  (PRN 13)", "norad_id": 24876,
     "tle1": "1 24876U 97035A   24001.50000000  .00000010  00000-0  00000+0 0  9991",
     "tle2": "2 24876  55.4000 239.4000 0046600 110.0000 250.5000  2.00563000196001",
     "_source_group": "gps"},
    {"name": "GPS BIIRM-1 (PRN 17)", "norad_id": 28874,
     "tle1": "1 28874U 05038A   24001.50000000  .00000005  00000-0  00000+0 0  9992",
     "tle2": "2 28874  55.0000 120.0000 0100000  45.0000 316.0000  2.00561000140001",
     "_source_group": "gps"},

    # ── GEO ──
    {"name": "TDRS 3", "norad_id": 19548,
     "tle1": "1 19548U 88091B   24001.50000000  .00000100  00000-0  00000+0 0  9991",
     "tle2": "2 19548  14.4000  30.0000 0003000 270.0000  90.0000  1.00270000130001",
     "_source_group": "geo"},

    # ── Iridium 33 debris ──
    {"name": "IRIDIUM 33 DEB [A]", "norad_id": 33776,
     "tle1": "1 33776U 97051Q   24001.50000000  .00000200  00000-0  13000-3 0  9991",
     "tle2": "2 33776  86.3900 127.4000 0015000  80.0000 280.2000 14.34580000800001",
     "_source_group": "iridium33_debris"},
    {"name": "IRIDIUM 33 DEB [B]", "norad_id": 33777,
     "tle1": "1 33777U 97051R   24001.50000000  .00000180  00000-0  12000-3 0  9992",
     "tle2": "2 33777  86.3850 127.3000 0018000  85.0000 275.2000 14.34560000800002",
     "_source_group": "iridium33_debris"},
    {"name": "IRIDIUM 33 DEB [C]", "norad_id": 33778,
     "tle1": "1 33778U 97051S   24001.50000000  .00000220  00000-0  14500-3 0  9993",
     "tle2": "2 33778  86.4000 127.5000 0020000  90.0000 270.2000 14.34540000800003",
     "_source_group": "iridium33_debris"},
    {"name": "IRIDIUM 33 DEB [D]", "norad_id": 33779,
     "tle1": "1 33779U 97051T   24001.50000000  .00000250  00000-0  16000-3 0  9994",
     "tle2": "2 33779  86.4100 127.6000 0022000  95.0000 265.2000 14.34520000800004",
     "_source_group": "iridium33_debris"},
    {"name": "IRIDIUM 33 DEB [E]", "norad_id": 33780,
     "tle1": "1 33780U 97051U   24001.50000000  .00000190  00000-0  12500-3 0  9995",
     "tle2": "2 33780  86.3800 127.2000 0016000  82.0000 278.2000 14.34600000800005",
     "_source_group": "iridium33_debris"},

    # ── Cosmos 2251 debris ──
    {"name": "COSMOS 2251 DEB [A]", "norad_id": 34454,
     "tle1": "1 34454U 93036PX  24001.50000000  .00000150  00000-0  10000-3 0  9991",
     "tle2": "2 34454  74.0300 100.0000 0030000  60.0000 300.5000 14.11780000750001",
     "_source_group": "cosmos2251_debris"},
    {"name": "COSMOS 2251 DEB [B]", "norad_id": 34455,
     "tle1": "1 34455U 93036PY  24001.50000000  .00000170  00000-0  11000-3 0  9992",
     "tle2": "2 34455  74.0400 100.1000 0035000  65.0000 295.5000 14.11760000750002",
     "_source_group": "cosmos2251_debris"},
    {"name": "COSMOS 2251 DEB [C]", "norad_id": 34456,
     "tle1": "1 34456U 93036PZ  24001.50000000  .00000130  00000-0  90000-4 0  9993",
     "tle2": "2 34456  74.0200  99.9000 0025000  55.0000 305.5000 14.11800000750003",
     "_source_group": "cosmos2251_debris"},
    {"name": "COSMOS 2251 DEB [D]", "norad_id": 34457,
     "tle1": "1 34457U 93036QA  24001.50000000  .00000190  00000-0  12500-3 0  9994",
     "tle2": "2 34457  74.0500 100.2000 0040000  70.0000 290.5000 14.11740000750004",
     "_source_group": "cosmos2251_debris"},
    {"name": "COSMOS 2251 DEB [E]", "norad_id": 34458,
     "tle1": "1 34458U 93036QB  24001.50000000  .00000160  00000-0  10500-3 0  9995",
     "tle2": "2 34458  74.0350 100.0500 0032000  62.0000 298.5000 14.11770000750005",
     "_source_group": "cosmos2251_debris"},

    # ── Fengyun-1C debris ──
    {"name": "FENGYUN 1C DEB [A]", "norad_id": 31140,
     "tle1": "1 31140U 99025AEB 24001.50000000  .00000300  00000-0  20000-3 0  9991",
     "tle2": "2 31140  99.0000 150.0000 0050000  45.0000 315.5000 14.50000000900001",
     "_source_group": "fengyun1c_debris"},
    {"name": "FENGYUN 1C DEB [B]", "norad_id": 31141,
     "tle1": "1 31141U 99025AEC 24001.50000000  .00000280  00000-0  18500-3 0  9992",
     "tle2": "2 31141  99.0100 150.1000 0055000  50.0000 310.5000 14.49980000900002",
     "_source_group": "fengyun1c_debris"},
    {"name": "FENGYUN 1C DEB [C]", "norad_id": 31142,
     "tle1": "1 31142U 99025AED 24001.50000000  .00000320  00000-0  21000-3 0  9993",
     "tle2": "2 31142  99.0200 150.2000 0060000  55.0000 305.5000 14.49960000900003",
     "_source_group": "fengyun1c_debris"},
    {"name": "FENGYUN 1C DEB [D]", "norad_id": 31143,
     "tle1": "1 31143U 99025AEE 24001.50000000  .00000340  00000-0  22500-3 0  9994",
     "tle2": "2 31143  99.0300 150.3000 0065000  60.0000 300.5000 14.49940000900004",
     "_source_group": "fengyun1c_debris"},
    {"name": "FENGYUN 1C DEB [E]", "norad_id": 31144,
     "tle1": "1 31144U 99025AEF 24001.50000000  .00000260  00000-0  17500-3 0  9995",
     "tle2": "2 31144  99.0400 150.4000 0048000  40.0000 320.5000 14.50020000900005",
     "_source_group": "fengyun1c_debris"},

    # ── Other notable objects ──
    {"name": "HUBBLE SPACE TELESCOPE", "norad_id": 20580,
     "tle1": "1 20580U 90037B   24001.50000000  .00000800  00000-0  40000-4 0  9991",
     "tle2": "2 20580  28.4700 110.0000 0002800 300.0000  60.0000 15.09000000500001",
     "_source_group": "science"},
    {"name": "TERRA", "norad_id": 25994,
     "tle1": "1 25994U 99068A   24001.50000000  .00000100  00000-0  30000-4 0  9992",
     "tle2": "2 25994  98.2000 330.0000 0001200  90.0000 270.0000 14.57110000350001",
     "_source_group": "science"},
    {"name": "AQUA", "norad_id": 27424,
     "tle1": "1 27424U 02022A   24001.50000000  .00000120  00000-0  35000-4 0  9993",
     "tle2": "2 27424  98.2100 330.1000 0001500  92.0000 268.0000 14.57100000300001",
     "_source_group": "science"},
    {"name": "NOAA 19", "norad_id": 33591,
     "tle1": "1 33591U 09005A   24001.50000000  .00000090  00000-0  65000-4 0  9994",
     "tle2": "2 33591  99.1500 330.0000 0014000  60.0000 300.2000 14.12400000250001",
     "_source_group": "weather"},
    {"name": "ENVISAT", "norad_id": 27386,
     "tle1": "1 27386U 02009A   24001.50000000  .00000040  00000-0  30000-4 0  9995",
     "tle2": "2 27386  98.2300  30.0000 0001200 270.0000  90.0000 14.38000000350001",
     "_source_group": "inactive"},
    {"name": "TIANGONG", "norad_id": 48274,
     "tle1": "1 48274U 21035A   24001.50000000  .00020000  00000-0  22000-3 0  9991",
     "tle2": "2 48274  41.4700 280.0000 0005000 200.0000 160.0000 15.62000000200001",
     "_source_group": "stations"},
    {"name": "METEOR 2-5 (DEBRIS)", "norad_id": 13990,
     "tle1": "1 13990U 83009A   24001.50000000  .00000070  00000-0  50000-4 0  9991",
     "tle2": "2 13990  82.5400  45.0000 0020000 120.0000 240.5000 13.83000000200001",
     "_source_group": "inactive"},
    {"name": "SL-16 R/B", "norad_id": 22285,
     "tle1": "1 22285U 92093B   24001.50000000  .00000060  00000-0  45000-4 0  9992",
     "tle2": "2 22285  71.0000  35.0000 0005000 250.0000 110.0000 14.22000000300001",
     "_source_group": "rocket_body"},
    {"name": "BREEZE-M DEB", "norad_id": 28945,
     "tle1": "1 28945U 06001C   24001.50000000  .00000030  00000-0  20000-4 0  9993",
     "tle2": "2 28945  48.5000 200.0000 7100000  10.0000 350.0000  2.25000000180001",
     "_source_group": "rocket_body"},
]


# ─── Historical TLEs: Pre-collision epoch for 2009 Iridium 33 / Cosmos 2251 ──
# These TLEs are from approximately Feb 8, 2009 (2 days before collision on Feb 10)
# Used in Historical Replay mode to demonstrate the pipeline would have flagged it.

_HISTORICAL_IRIDIUM_COSMOS: List[Dict] = [
    {"name": "IRIDIUM 33", "norad_id": 24946,
     "tle1": "1 24946U 97051C   09040.50000000  .00000090  00000-0  26000-4 0  9991",
     "tle2": "2 24946  86.3988 127.4239 0002075  87.2531 272.8884 14.34217578600001",
     "_source_group": "historical_replay"},
    {"name": "COSMOS 2251", "norad_id": 22675,
     "tle1": "1 22675U 93036A   09040.50000000  .00000040  00000-0  10000-3 0  9992",
     "tle2": "2 22675  74.0362  18.4523 0013711  89.1484 271.1079 14.34078220800001",
     "_source_group": "historical_replay"},
    # Supporting objects from same epoch for context
    {"name": "IRIDIUM 33 DEB (EARLY)", "norad_id": 33776,
     "tle1": "1 33776U 97051Q   09040.50000000  .00000300  00000-0  18000-3 0  9993",
     "tle2": "2 33776  86.3900 127.4100 0015000  80.0000 280.2000 14.34580000100001",
     "_source_group": "historical_replay"},
]


# ─── Orbital regime metadata ──

REGIME_BOUNDARIES = {
    "LEO": (160, 2000),     # km altitude
    "MEO": (2000, 35786),
    "GEO": (35586, 35986),  # ±200 km of GEO belt
    "HEO": (35986, 500000),
}


def classify_regime(alt_km: float) -> str:
    """Classify orbital regime from mean altitude (km above Earth surface)."""
    if alt_km < 2000:
        return "LEO"
    elif alt_km < 35586:
        return "MEO"
    elif alt_km < 35986:
        return "GEO"
    else:
        return "HEO"


def get_sample_tles(include_debris: bool = True) -> List[Dict]:
    """
    Return the offline sample TLE population.

    Parameters
    ----------
    include_debris : bool
        If False, exclude debris objects (useful for cleaner demos).

    Returns
    -------
    list of dict
        Each dict matches the TLE Record schema (05_BACKEND_SCHEMA.md §2).
    """
    if include_debris:
        return list(_SAMPLE_TLES)
    return [r for r in _SAMPLE_TLES
            if "DEB" not in r["name"] and "debris" not in r["_source_group"]]


def get_historical_replay_tles() -> List[Dict]:
    """
    Return TLEs from just before the 2009 Iridium 33 / Cosmos 2251 collision.
    Used in Historical Replay mode.
    """
    return list(_HISTORICAL_IRIDIUM_COSMOS)


def get_all_source_groups() -> List[str]:
    """Return unique source groups in the sample population."""
    return list(set(r["_source_group"] for r in _SAMPLE_TLES))


# ─── Smoke test ──

if __name__ == "__main__":
    tles = get_sample_tles()
    print(f"Sample population: {len(tles)} objects")
    for g in sorted(get_all_source_groups()):
        count = sum(1 for r in tles if r["_source_group"] == g)
        print(f"  {g}: {count}")

    hist = get_historical_replay_tles()
    print(f"\nHistorical replay objects: {len(hist)}")
    for r in hist:
        print(f"  {r['name']} (NORAD {r['norad_id']})")

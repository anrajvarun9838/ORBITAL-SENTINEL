"""
export_report.py — Exportable mission briefing (PDF/HTML)

One-click export of the top-N events with their AI advisories,
risk scores, uncertainty metrics, and analytical Pc values.

Uses fpdf2 for PDF generation (lightweight, no external dependencies).
"""

from __future__ import annotations
import io
import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import pandas as pd

try:
    from fpdf import FPDF
    _HAS_FPDF = True
except ImportError:
    _HAS_FPDF = False


def _sanitize(text: str) -> str:
    """Remove characters that cause encoding issues in PDF."""
    # Replace unicode arrows and special chars with ASCII equivalents
    replacements = {
        "↔": "<->",
        "→": "->",
        "←": "<-",
        "↑": "^",
        "↓": "v",
        "•": "*",
        "╔": "+",
        "╗": "+",
        "╚": "+",
        "╝": "+",
        "═": "=",
        "║": "|",
        "─": "-",
        "⚠️": "[!]",
        "🛰️": "[SAT]",
        "📋": "[DOC]",
        "🌐": "[GLOBE]",
        "🌍": "[EARTH]",
        "📡": "[ANT]",
        "🤖": "[AI]",
        "🚀": "[ROCKET]",
        "₁₀": "10",
        "₂": "2",
        "\u2248": "~",
        "\u0394": "Delta-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Remove any remaining non-latin1 characters
    return text.encode("latin-1", errors="replace").decode("latin-1")


def generate_pdf_report(
    events_df: pd.DataFrame,
    reports: Dict[int, str],
    training_report: Dict[str, Any],
    top_n: int = 10,
) -> bytes:
    """
    Generate a PDF mission briefing for the top-N events.

    Parameters
    ----------
    events_df : pd.DataFrame
        Scored Event Records.
    reports : dict
        Mapping of event index → AI advisory text.
    training_report : dict
        Model training metrics (MAE, etc.).
    top_n : int
        Number of events to include.

    Returns
    -------
    bytes — PDF content.
    """
    if not _HAS_FPDF:
        raise ImportError("fpdf2 is required for PDF export: pip install fpdf2")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Cover page ──
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 20, _sanitize("SPACE SITUATIONAL AWARENESS"), ln=True, align="C")
    pdf.cell(0, 15, _sanitize("MISSION BRIEFING"), ln=True, align="C")

    pdf.set_font("Helvetica", "", 12)
    pdf.ln(10)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    pdf.cell(0, 8, f"Generated: {now}", ln=True, align="C")
    pdf.cell(0, 8, f"Events covered: Top {min(top_n, len(events_df))}", ln=True, align="C")

    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(0, 5, _sanitize(
        "ADVISORY ONLY - NOT AN OPERATIONAL DIRECTIVE. "
        "All risk scores are statistical estimates observed on sample/test data. "
        "This system does not provide guaranteed accuracy or certified operational status."
    ))

    # ── Model metrics ──
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Model Validation Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)

    mae = training_report.get("mae", "N/A")
    source = training_report.get("data_source", "N/A")
    n_train = training_report.get("n_train", "N/A")
    pdf.cell(0, 6, f"Data source: {source}", ln=True)
    pdf.cell(0, 6, f"Training samples: {n_train}", ln=True)
    pdf.cell(0, 6, f"Test MAE: {mae} points (on 0-100 scale)", ln=True)
    pdf.cell(0, 6, _sanitize(
        "Note: MAE ~25 points is an honest result for this 4-feature model on noisy Pc data."
    ), ln=True)

    # ── Event details ──
    top_events = events_df.head(top_n)

    for idx, (_, row) in enumerate(top_events.iterrows()):
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, f"Event #{idx + 1}: {_sanitize(str(row.get('object_a', '?')))} "
                        f"<-> {_sanitize(str(row.get('object_b', '?')))}", ln=True)

        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"Risk Score: {row.get('risk_score', 0):.1f}/100 "
                       f"[{row.get('risk_category', '?')}]", ln=True)
        pdf.cell(0, 6, f"Miss Distance: {row.get('miss_distance_km', 0):.3f} km", ln=True)
        pdf.cell(0, 6, f"Relative Velocity: {row.get('relative_velocity_kms', 0):.2f} km/s",
                 ln=True)
        pdf.cell(0, 6, f"Time to TCA: {row.get('tca_hours_from_now', 0):.1f} hours", ln=True)

        if "pc" in row:
            pdf.cell(0, 6, f"Analytical Pc: {row.get('pc', 0):.2e} "
                           f"(Chan 2D method)", ln=True)

        if "delta_v_ms" in row:
            pdf.cell(0, 6, f"Estimated Delta-v: {row.get('delta_v_ms', 0):.4f} m/s", ln=True)

        # AI Advisory
        if idx in reports:
            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 8, "AI Advisory:", ln=True)
            pdf.set_font("Courier", "", 8)
            advisory_text = _sanitize(reports[idx])
            for line in advisory_text.split("\n"):
                pdf.cell(0, 4, line[:120], ln=True)

    return pdf.output()


def generate_html_report(
    events_df: pd.DataFrame,
    reports: Dict[int, str],
    training_report: Dict[str, Any],
    top_n: int = 10,
) -> str:
    """Generate an HTML mission briefing."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    top_events = events_df.head(top_n)

    mae = training_report.get("mae", "N/A")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SSA Mission Briefing — {now}</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; background: #0a0e1a; color: #e0e0e0;
       max-width: 900px; margin: 0 auto; padding: 20px; }}
h1 {{ color: #0A84FF; text-align: center; }}
h2 {{ color: #0A84FF; border-bottom: 1px solid #1a2040; padding-bottom: 5px; }}
.event {{ background: #111827; border: 1px solid #1a2040; border-radius: 8px;
          padding: 15px; margin: 15px 0; }}
.critical {{ border-left: 4px solid #ff3b30; }}
.high {{ border-left: 4px solid #ff9500; }}
.medium {{ border-left: 4px solid #ffcc00; }}
.low {{ border-left: 4px solid #30d158; }}
.advisory {{ background: #0d1117; padding: 10px; border-radius: 4px;
             font-family: monospace; font-size: 12px; white-space: pre-wrap; }}
.disclaimer {{ background: #1c1c1e; padding: 10px; border-radius: 4px;
               font-style: italic; color: #aaa; margin: 10px 0; }}
.metric {{ display: inline-block; margin: 5px 15px 5px 0; }}
.metric-label {{ color: #888; font-size: 11px; }}
.metric-value {{ color: #0A84FF; font-size: 18px; font-weight: bold; }}
</style>
</head>
<body>
<h1>SPACE SITUATIONAL AWARENESS<br>MISSION BRIEFING</h1>
<p style="text-align:center">Generated: {now}</p>
<div class="disclaimer">
ADVISORY ONLY — NOT AN OPERATIONAL DIRECTIVE. All risk scores are statistical
estimates observed on sample/test data. This system does not provide guaranteed
accuracy or certified operational status.
</div>

<h2>Model Validation</h2>
<p>Data source: {training_report.get('data_source', 'N/A')} |
   Training samples: {training_report.get('n_train', 'N/A')} |
   Test MAE: {mae} points</p>

<h2>Top Events</h2>
"""

    for idx, (_, row) in enumerate(top_events.iterrows()):
        cat = row.get("risk_category", "Low").lower()
        html += f"""
<div class="event {cat}">
<h3>#{idx+1}: {row.get('object_a', '?')} ↔ {row.get('object_b', '?')}</h3>
<div class="metric"><span class="metric-label">Risk Score</span><br>
<span class="metric-value">{row.get('risk_score', 0):.1f}</span>/100 [{row.get('risk_category', '?')}]</div>
<div class="metric"><span class="metric-label">Miss Distance</span><br>
<span class="metric-value">{row.get('miss_distance_km', 0):.3f}</span> km</div>
<div class="metric"><span class="metric-label">Relative Velocity</span><br>
<span class="metric-value">{row.get('relative_velocity_kms', 0):.2f}</span> km/s</div>
<div class="metric"><span class="metric-label">TCA</span><br>
<span class="metric-value">{row.get('tca_hours_from_now', 0):.1f}</span> hours</div>
"""
        if idx in reports:
            html += f'<div class="advisory">{reports[idx]}</div>'
        html += "</div>"

    html += """
<div class="disclaimer">
All figures reported as observed on sample/test data — not operational guarantees.
</div>
</body></html>"""

    return html


# ─── Smoke test ──

if __name__ == "__main__":
    test_events = pd.DataFrame([
        {"object_a": "STARLINK-1007", "object_b": "COSMOS DEB",
         "miss_distance_km": 2.5, "relative_velocity_kms": 7.8,
         "tca_hours_from_now": 36.5, "risk_score": 72.3, "risk_category": "High"},
    ])
    test_reports = {0: "Test advisory report for export testing."}
    test_training = {"mae": 25.1, "data_source": "esa_kelvins_cdm", "n_train": 162634}

    html = generate_html_report(test_events, test_reports, test_training)
    print(f"HTML report generated: {len(html)} chars")

    if _HAS_FPDF:
        pdf_bytes = generate_pdf_report(test_events, test_reports, test_training)
        print(f"PDF report generated: {len(pdf_bytes)} bytes")

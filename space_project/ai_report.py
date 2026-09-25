"""
ai_report.py — Generative AI risk advisory reports (FR-11, FR-12)

Generates natural-language risk advisories for scored conjunction events.

Two paths:
  1. LLM (Google Gemini) — when GEMINI_API_KEY is set
  2. Template fallback — deterministic, same output format, zero network

Every report is labeled:
"ADVISORY ONLY — NOT AN OPERATIONAL DIRECTIVE"

Data contract in (Scored Event Record dict):
  {
      object_a,
      object_b,
      miss_distance_km,
      relative_velocity_kms,
      tca_hours_from_now,
      risk_score,
      risk_category,
      ...
  }

Data contract out:
  Plain-text structured risk advisory.
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, Any

from dotenv import load_dotenv


# ============================================================
# LOAD .ENV
# ============================================================

# Loads GEMINI_API_KEY from the project's .env file
load_dotenv()


# ============================================================
# TEMPLATE FALLBACK
# ============================================================

def _template_report(event: Dict[str, Any]) -> str:
    """
    Deterministic template-based report.

    Used when:
      - GEMINI_API_KEY is missing
      - Gemini API fails
      - Gemini returns an empty response

    This keeps the application working offline.
    """

    name_a = event.get("object_a", "Unknown-A")
    name_b = event.get("object_b", "Unknown-B")

    miss = event.get("miss_distance_km", 0)
    vel = event.get("relative_velocity_kms", 0)
    tca = event.get("tca_hours_from_now", 0)

    score = event.get("risk_score", 0)
    category = event.get("risk_category", "Unknown")

    pc = event.get("pc", None)
    delta_v = event.get("delta_v_ms", None)

    # --------------------------------------------------------
    # SUMMARY + ACTION BASED ON RISK CATEGORY
    # --------------------------------------------------------

    if category == "Critical":

        summary = (
            f"CRITICAL conjunction detected between {name_a} and {name_b}. "
            f"Predicted miss distance of {miss:.3f} km with closing velocity "
            f"of {vel:.2f} km/s. Time to closest approach: {tca:.1f} hours. "
            f"This event requires immediate operator attention and potential "
            f"avoidance maneuver assessment."
        )

        action = (
            "IMMEDIATE ACTION RECOMMENDED: Assess avoidance maneuver feasibility. "
            "Verify latest tracking data. Coordinate with space surveillance network "
            "for updated conjunction data message (CDM). "
            "Prepare contingency plan if maneuver is not executed."
        )

    elif category == "High":

        summary = (
            f"High-risk conjunction identified between {name_a} and {name_b}. "
            f"Predicted miss distance: {miss:.3f} km at relative velocity "
            f"{vel:.2f} km/s. TCA in {tca:.1f} hours. "
            f"Event warrants close monitoring and maneuver readiness."
        )

        action = (
            "MONITOR CLOSELY: Request updated tracking data. Prepare preliminary "
            "avoidance maneuver profile. Re-evaluate risk at next CDM update. "
            "Escalate if miss distance decreases or uncertainty grows."
        )

    elif category == "Medium":

        summary = (
            f"Medium-risk conjunction between {name_a} and {name_b}. "
            f"Miss distance: {miss:.3f} km, relative velocity: {vel:.2f} km/s, "
            f"TCA: {tca:.1f} hours. Event is within monitoring threshold but "
            f"does not currently require intervention."
        )

        action = (
            "ROUTINE MONITORING: Continue tracking. No immediate action required. "
            "Re-evaluate at next tracking update or if risk score trends upward."
        )

    else:

        summary = (
            f"Low-risk conjunction between {name_a} and {name_b}. "
            f"Miss distance: {miss:.3f} km, relative velocity: {vel:.2f} km/s, "
            f"TCA: {tca:.1f} hours. Conjunction is within normal operational "
            f"bounds for the current orbital regime."
        )

        action = "NO ACTION REQUIRED: Event logged for record-keeping."

    # --------------------------------------------------------
    # BUILD REPORT
    # --------------------------------------------------------

    lines = [
        "+==============================================================+",
        "|                      RISK ADVISORY                           |",
        "|         [!] ADVISORY ONLY -- NOT AN OPERATIONAL DIRECTIVE   |",
        "+==============================================================+",
        "",
        f"Object: {name_a}  |  Object: {name_b}",
        f"Time to TCA: {tca:.1f} hours",
        f"Risk Score: {score:.1f}/100  [{category}]",
    ]

    if pc is not None:
        lines.append(
            f"Analytical Pc: {pc:.2e} (Chan 2D method)"
        )

    lines += [
        "",
        "SUMMARY:",
        summary,
        "",
        "RECOMMENDED ACTION:",
        action,
        "",
        f"PRIORITY: {category.upper()}",
    ]

    if delta_v is not None and delta_v > 0:

        lines += [
            "",
            f"ESTIMATED AVOIDANCE Delta-v: {delta_v:.4f} m/s",
            "(Simplified impulse approximation -- see assumptions in Threat Assessment tab)",
        ]

    lines += [
        "",
        "--- Method & Uncertainty ---",
        "Risk score: ML model (GBR, 4 features). Observed MAE ~25 pts on ESA CDM test data.",
        "This score is a statistical estimate, not a certainty.",
        "All figures are observed on sample/test data -- not operational guarantees.",
        "",
        "Generated by: template engine (offline mode)",
    ]

    return "\n".join(lines)


# ============================================================
# GEMINI LLM REPORT
# ============================================================

def _llm_report(event: Dict[str, Any]) -> str:
    """
    Generate a report via Google Gemini API.

    Falls back to template report on any error.
    """

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    # --------------------------------------------------------
    # NO API KEY
    # --------------------------------------------------------

    if not api_key:
        return _template_report(event)

    try:

        from google import genai

        # Create Gemini client
        client = genai.Client(api_key=api_key)

        # ----------------------------------------------------
        # PROMPT
        # ----------------------------------------------------

        prompt = f"""
You are a Space Situational Awareness (SSA) analyst assistant.

Generate a structured risk advisory for the conjunction event below.

IMPORTANT RULES:

1. Use ONLY the provided event data.
2. Do not invent orbital data.
3. Do not invent risk scores.
4. Do not change numerical values.
5. Treat the ML risk score as a statistical estimate.
6. Do not claim that a collision is certain.
7. Do not provide autonomous spacecraft control commands.
8. The report is advisory only.
9. Clearly mention uncertainty.
10. Keep the answer professional and concise.

EVENT DATA:

Object A:
{event.get("object_a", "Unknown")}

Object B:
{event.get("object_b", "Unknown")}

Miss distance:
{event.get("miss_distance_km", 0):.3f} km

Relative velocity:
{event.get("relative_velocity_kms", 0):.2f} km/s

Time to TCA:
{event.get("tca_hours_from_now", 0):.1f} hours

ML Risk Score:
{event.get("risk_score", 0):.1f}/100

Risk Category:
{event.get("risk_category", "Unknown")}

Analytical Probability of Collision:
{event.get("pc", "N/A")}

Delta-v estimate:
{event.get("delta_v_ms", "N/A")} m/s


FORMAT YOUR RESPONSE EXACTLY IN THIS STRUCTURE:

RISK ADVISORY

⚠️ ADVISORY ONLY — NOT AN OPERATIONAL DIRECTIVE

Object: [Object A] | Object: [Object B]
Time to TCA: [hours] hours
Risk Score: [score]/100 [category]

SUMMARY:
[2-3 concise sentences explaining the event using only the supplied data.]

RECOMMENDED ACTION:
[General monitoring/review recommendation based only on the supplied risk information.
Do not provide spacecraft control commands.]

PRIORITY: [LOW|MEDIUM|HIGH|CRITICAL]

IMPORTANT:
State that all figures are statistical estimates observed on sample/test data and
are not operational guarantees.

End the response with exactly:

Generated by: Gemini LLM (advisory use only)
"""

        # ----------------------------------------------------
        # GEMINI API CALL
        # ----------------------------------------------------

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        # ----------------------------------------------------
        # RESPONSE VALIDATION
        # ----------------------------------------------------

        if response is not None and getattr(response, "text", None):

            return response.text.strip()

        # Empty response → fallback
        warnings.warn(
            "Gemini returned an empty response — using template report."
        )

        return _template_report(event)

    except Exception as exc:

        warnings.warn(
            f"Gemini LLM report generation failed: {exc} "
            f"— using template."
        )

        return _template_report(event)


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def generate_report(
    event: Dict[str, Any],
    force_template: bool = False,
) -> str:
    """
    Generate a risk advisory report for a scored conjunction event.

    Parameters
    ----------
    event : dict
        Scored Event Record.

    force_template : bool
        If True, skip Gemini even if API key is available.

    Returns
    -------
    str
        Formatted risk advisory text.
    """

    if force_template:

        return _template_report(event)

    return _llm_report(event)


# ============================================================
# REPORT MODE
# ============================================================

def get_report_mode() -> str:
    """
    Return:
        'llm'       → Gemini API key available
        'template'  → no Gemini API key
    """

    return (
        "llm"
        if os.environ.get("GEMINI_API_KEY", "").strip()
        else "template"
    )


# ============================================================
# SMOKE TEST
# ============================================================

if __name__ == "__main__":

    test_event = {
        "object_a": "STARLINK-1007",
        "object_b": "COSMOS 2251 DEB [A]",
        "miss_distance_km": 2.5,
        "relative_velocity_kms": 7.8,
        "tca_hours_from_now": 36.5,
        "risk_score": 72.3,
        "risk_category": "High",
        "pc": 1.2e-6,
        "delta_v_ms": 0.019,
    }

    print(f"Report mode: {get_report_mode()}")
    print()

    print(
        generate_report(
            test_event,
            force_template=False
        )
    )
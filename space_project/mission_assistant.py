"""
mission_assistant.py
Orbital Sentinel - Current Scenario AI Mission Assistant

Answers ONLY from the current analysis session.
Compatible with existing app.py:

    answer_question(question, session_data, force_template=False)
"""

from __future__ import annotations

import os
import re
import warnings
from typing import Dict, Any, List, Optional

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ============================================================
# SAFE HELPERS
# ============================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    return str(value)


def _normalize_question(question: str) -> str:
    q = _safe_text(question).lower().strip()
    q = re.sub(r"[^a-zA-Z0-9_\-\s]", " ", q)
    q = re.sub(r"\s+", " ", q)
    return q.strip()


def _contains_any(text: str, words: List[str]) -> bool:
    return any(word in text for word in words)


def _get_events(session_data: Dict[str, Any]) -> pd.DataFrame:
    events = session_data.get("events", pd.DataFrame())

    if isinstance(events, pd.DataFrame):
        return events.copy()

    try:
        return pd.DataFrame(events)
    except Exception:
        return pd.DataFrame()


def _get_records(session_data: Dict[str, Any]) -> List[Any]:
    records = session_data.get("records", [])

    if records is None:
        return []

    if isinstance(records, list):
        return records

    try:
        return list(records)
    except Exception:
        return []


# ============================================================
# SORTING
# ============================================================

def _sort_by_risk(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "risk_score" not in df.columns:
        return df.copy()

    result = df.copy()

    result["_risk_value"] = pd.to_numeric(
        result["risk_score"],
        errors="coerce"
    ).fillna(0)

    return result.sort_values(
        "_risk_value",
        ascending=False
    )


def _sort_by_tca(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "tca_hours_from_now" not in df.columns:
        return df.copy()

    result = df.copy()

    result["_tca_value"] = pd.to_numeric(
        result["tca_hours_from_now"],
        errors="coerce"
    ).fillna(float("inf"))

    return result.sort_values(
        "_tca_value",
        ascending=True
    )


def _sort_by_distance(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "miss_distance_km" not in df.columns:
        return df.copy()

    result = df.copy()

    result["_distance_value"] = pd.to_numeric(
        result["miss_distance_km"],
        errors="coerce"
    ).fillna(float("inf"))

    return result.sort_values(
        "_distance_value",
        ascending=True
    )


# ============================================================
# EVENT FORMATTER
# ============================================================

def _format_event(row: pd.Series) -> str:
    a = _safe_text(
        row.get("object_a"),
        "Unknown"
    )

    b = _safe_text(
        row.get("object_b"),
        "Unknown"
    )

    risk = _safe_float(
        row.get("risk_score")
    )

    category = _safe_text(
        row.get("risk_category"),
        "Unknown"
    )

    distance = _safe_float(
        row.get("miss_distance_km")
    )

    velocity = _safe_float(
        row.get("relative_velocity_kms")
    )

    tca = _safe_float(
        row.get("tca_hours_from_now")
    )

    return (
        f"{a} ↔ {b} | "
        f"Risk: {risk:.1f}/100 ({category}) | "
        f"Miss distance: {distance:.3f} km | "
        f"Relative velocity: {velocity:.2f} km/s | "
        f"TCA: {tca:.1f} hours"
    )


# ============================================================
# RISK COUNTS
# ============================================================

def _count_risk(
    events: pd.DataFrame,
    category: str
) -> int:

    if events.empty:
        return 0

    if "risk_category" not in events.columns:
        return 0

    values = (
        events["risk_category"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return int(
        (values == category.lower()).sum()
    )


# ============================================================
# FIND OBJECT NAME FROM CURRENT EVENTS
# ============================================================

def _extract_object_name(
    question: str,
    events: pd.DataFrame
) -> Optional[str]:

    if events.empty:
        return None

    names = set()

    for column in ["object_a", "object_b"]:

        if column not in events.columns:
            continue

        for value in events[column].dropna():

            name = str(value).strip()

            if name:
                names.add(name)

    # Longest first so specific names match first.
    for name in sorted(
        names,
        key=len,
        reverse=True
    ):

        if name.lower() in question.lower():
            return name

    return None


# ============================================================
# OBJECT ANALYSIS
# ============================================================

def _analyze_object(
    object_name: str,
    events: pd.DataFrame
) -> str:

    if events.empty:
        return (
            f"No conjunction data is available for "
            f"{object_name} in the current scenario."
        )

    mask = pd.Series(
        False,
        index=events.index
    )

    if "object_a" in events.columns:

        mask = mask | (
            events["object_a"]
            .astype(str)
            .str.lower()
            == object_name.lower()
        )

    if "object_b" in events.columns:

        mask = mask | (
            events["object_b"]
            .astype(str)
            .str.lower()
            == object_name.lower()
        )

    object_events = events[mask]

    if object_events.empty:

        return (
            f"I couldn't find any current conjunction "
            f"involving {object_name}."
        )

    object_events = _sort_by_risk(
        object_events
    )

    top = object_events.iloc[0]

    risk = _safe_float(
        top.get("risk_score")
    )

    category = _safe_text(
        top.get("risk_category"),
        "Unknown"
    )

    distance = _safe_float(
        top.get("miss_distance_km")
    )

    tca = _safe_float(
        top.get("tca_hours_from_now")
    )

    if (
        str(top.get("object_a", "")).lower()
        == object_name.lower()
    ):
        other = _safe_text(
            top.get("object_b"),
            "Unknown"
        )
    else:
        other = _safe_text(
            top.get("object_a"),
            "Unknown"
        )

    return (
        f"{object_name} is involved in "
        f"{len(object_events)} current conjunction event(s). "
        f"Its highest-risk event is with {other}. "
        f"Risk: {risk:.1f}/100 ({category}). "
        f"Miss distance: {distance:.3f} km. "
        f"TCA: {tca:.1f} hours."
    )


# ============================================================
# MOST DANGEROUS OBJECT
# ============================================================

def _highest_risk_event(
    events: pd.DataFrame
) -> Optional[pd.Series]:

    if events.empty:
        return None

    ranked = _sort_by_risk(events)

    if ranked.empty:
        return None

    return ranked.iloc[0]


def _most_dangerous_object(
    events: pd.DataFrame
) -> Optional[str]:

    top = _highest_risk_event(events)

    if top is None:
        return None

    a = _safe_text(
        top.get("object_a"),
        "Unknown"
    )

    b = _safe_text(
        top.get("object_b"),
        "Unknown"
    )

    return f"{a} ↔ {b}"


# ============================================================
# STARLINK ANALYSIS
# ============================================================

def _analyze_starlink(
    events: pd.DataFrame
) -> str:

    if events.empty:
        return (
            "No conjunction data is available in the "
            "current scenario."
        )

    mask = pd.Series(
        False,
        index=events.index
    )

    for column in ["object_a", "object_b"]:

        if column in events.columns:

            mask = mask | (
                events[column]
                .astype(str)
                .str.contains(
                    "starlink",
                    case=False,
                    na=False
                )
            )

    starlink_events = events[mask]

    if starlink_events.empty:

        return (
            "No Starlink-related conjunction event was "
            "detected in the current scenario."
        )

    starlink_events = _sort_by_risk(
        starlink_events
    )

    top = starlink_events.iloc[0]

    return (
        f"The current scenario has "
        f"{len(starlink_events)} Starlink-related "
        f"conjunction event(s). "
        f"The highest-risk one is "
        f"{_safe_text(top.get('object_a'))} ↔ "
        f"{_safe_text(top.get('object_b'))}, "
        f"with risk "
        f"{_safe_float(top.get('risk_score')):.1f}/100 "
        f"({_safe_text(top.get('risk_category'), 'Unknown')}) "
        f"and miss distance "
        f"{_safe_float(top.get('miss_distance_km')):.3f} km."
    )


# ============================================================
# LOCAL AI ASSISTANT
# ============================================================

def _rule_based_answer(
    question: str,
    session_data: Dict[str, Any]
) -> str:

    q = _normalize_question(question)

    events = _get_events(session_data)

    records = _get_records(session_data)

    # ========================================================
    # EMPTY
    # ========================================================

    if not q:

        return (
            "I'm ready. Ask me about the current orbital "
            "scenario, collision risk, TCA, miss distance, "
            "or a specific object."
        )

    # ========================================================
    # GREETINGS
    # ========================================================

    if q in {
        "hi",
        "hii",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
    }:

        return (
            f"Hello! I'm the Orbital Sentinel mission "
            f"assistant. This current run has "
            f"{len(records)} tracked objects and "
            f"{len(events)} conjunction events. "
            f"Ask me which object is most dangerous, "
            f"the highest-risk collision, next TCA, "
            f"critical events, or any object name."
        )

    # ========================================================
    # NO EVENTS
    # ========================================================

    if events.empty:

        return (
            f"The current scenario contains "
            f"{len(records)} tracked objects, but no "
            f"conjunction events are currently available "
            f"for analysis."
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    if _contains_any(
        q,
        [
            "summary",
            "summarize",
            "overview",
            "overall situation",
            "overall status",
            "situation",
            "status",
            "mission status",
            "give me report",
            "give me overview",
            "what is happening",
            "whats happening",
        ]
    ):

        critical = _count_risk(
            events,
            "Critical"
        )

        high = _count_risk(
            events,
            "High"
        )

        medium = _count_risk(
            events,
            "Medium"
        )

        low = _count_risk(
            events,
            "Low"
        )

        top = _highest_risk_event(events)

        return (
            f"Current scenario summary: "
            f"{len(records)} objects are tracked and "
            f"{len(events)} conjunction events are detected. "
            f"Risk distribution: "
            f"{critical} Critical, "
            f"{high} High, "
            f"{medium} Medium, "
            f"{low} Low. "
            f"The highest-risk pair is "
            f"{_safe_text(top.get('object_a'))} ↔ "
            f"{_safe_text(top.get('object_b'))} "
            f"with a risk score of "
            f"{_safe_float(top.get('risk_score')):.1f}/100."
        )

    # ========================================================
    # MOST DANGEROUS / HIGHEST RISK
    # ========================================================

    if _contains_any(
        q,
        [
            "highest risk",
            "highest-risk",
            "highest collision risk",
            "highest collision",
            "most risky",
            "most risk",
            "most dangerous",
            "dangerous object",
            "dangerous satellite",
            "most dangerous satellite",
            "worst collision",
            "worst object",
            "worst satellite",
            "maximum risk",
            "max risk",
            "greatest risk",
            "greatest threat",
            "biggest threat",
            "primary threat",
            "main threat",
            "highest threat",
            "most high collision",
            "high collision object",
            "which object is dangerous",
            "which satellite is dangerous",
            "which object has highest risk",
            "which satellite has highest risk",
        ]
    ):

        top = _highest_risk_event(events)

        if top is None:
            return (
                "There is no risk event available "
                "in the current scenario."
            )

        return (
            f"The highest-risk collision in the current "
            f"scenario is "
            f"{_safe_text(top.get('object_a'))} ↔ "
            f"{_safe_text(top.get('object_b'))}. "
            f"Risk score: "
            f"{_safe_float(top.get('risk_score')):.1f}/100 "
            f"({_safe_text(top.get('risk_category'), 'Unknown')}). "
            f"Miss distance: "
            f"{_safe_float(top.get('miss_distance_km')):.3f} km. "
            f"Relative velocity: "
            f"{_safe_float(top.get('relative_velocity_kms')):.2f} km/s. "
            f"TCA: "
            f"{_safe_float(top.get('tca_hours_from_now')):.1f} hours."
        )

    # ========================================================
    # CRITICAL
    # ========================================================

    if "critical" in q:

        count = _count_risk(
            events,
            "Critical"
        )

        if _contains_any(
            q,
            [
                "how many",
                "count",
                "number",
                "total",
            ]
        ):

            return (
                f"There are {count} Critical "
                f"conjunction event(s) in the current scenario."
            )

        critical_events = events[
            events["risk_category"]
            .astype(str)
            .str.lower()
            == "critical"
        ] if "risk_category" in events.columns else pd.DataFrame()

        if critical_events.empty:

            return (
                "There are no Critical conjunction "
                "events in the current scenario."
            )

        critical_events = _sort_by_risk(
            critical_events
        )

        result = [
            f"I found {len(critical_events)} Critical "
            f"event(s) in the current scenario:"
        ]

        for _, row in critical_events.head(5).iterrows():

            result.append(
                f"• {_format_event(row)}"
            )

        return "\n".join(result)

    # ========================================================
    # HIGH RISK
    # ========================================================

    if _contains_any(
        q,
        [
            "high risk",
            "high-risk",
            "high collision",
            "high threat",
        ]
    ):

        count = _count_risk(
            events,
            "High"
        )

        if _contains_any(
            q,
            [
                "how many",
                "count",
                "number",
                "total",
            ]
        ):

            return (
                f"There are {count} High-risk "
                f"conjunction event(s) in the current scenario."
            )

        high_events = events[
            events["risk_category"]
            .astype(str)
            .str.lower()
            == "high"
        ] if "risk_category" in events.columns else pd.DataFrame()

        if high_events.empty:

            return (
                "There are no High-risk conjunction "
                "events in the current scenario."
            )

        high_events = _sort_by_risk(
            high_events
        )

        result = [
            f"There are {len(high_events)} High-risk "
            f"event(s). The top current events are:"
        ]

        for _, row in high_events.head(5).iterrows():

            result.append(
                f"• {_format_event(row)}"
            )

        return "\n".join(result)

    # ========================================================
    # NEXT COLLISION / NEXT TCA
    # ========================================================

    if _contains_any(
        q,
        [
            "next collision",
            "next conjunction",
            "next tca",
            "next event",
            "soonest collision",
            "soonest conjunction",
            "soonest event",
            "earliest collision",
            "earliest conjunction",
            "upcoming collision",
            "upcoming conjunction",
            "what happens next",
            "when is the next",
        ]
    ):

        next_event = (
            _sort_by_tca(events)
            .iloc[0]
        )

        return (
            f"The next conjunction in the current "
            f"scenario is "
            f"{_safe_text(next_event.get('object_a'))} ↔ "
            f"{_safe_text(next_event.get('object_b'))}. "
            f"TCA is in "
            f"{_safe_float(next_event.get('tca_hours_from_now')):.1f} "
            f"hours. "
            f"Miss distance: "
            f"{_safe_float(next_event.get('miss_distance_km')):.3f} km. "
            f"Risk: "
            f"{_safe_float(next_event.get('risk_score')):.1f}/100 "
            f"({_safe_text(next_event.get('risk_category'), 'Unknown')})."
        )

    # ========================================================
    # CLOSEST COLLISION
    # ========================================================

    if _contains_any(
        q,
        [
            "closest collision",
            "closest approach",
            "closest object",
            "closest pair",
            "minimum distance",
            "smallest distance",
            "smallest miss",
            "minimum miss",
            "nearest collision",
            "nearest objects",
        ]
    ):

        closest = (
            _sort_by_distance(events)
            .iloc[0]
        )

        return (
            f"The closest current conjunction is "
            f"{_safe_text(closest.get('object_a'))} ↔ "
            f"{_safe_text(closest.get('object_b'))}. "
            f"Miss distance: "
            f"{_safe_float(closest.get('miss_distance_km')):.3f} km. "
            f"Risk score: "
            f"{_safe_float(closest.get('risk_score')):.1f}/100 "
            f"({_safe_text(closest.get('risk_category'), 'Unknown')}). "
            f"TCA: "
            f"{_safe_float(closest.get('tca_hours_from_now')):.1f} hours."
        )

    # ========================================================
    # HIGHEST VELOCITY
    # ========================================================

    if _contains_any(
        q,
        [
            "highest velocity",
            "highest relative velocity",
            "highest speed",
            "fastest collision",
            "fastest object",
            "maximum velocity",
            "max velocity",
        ]
    ):

        if "relative_velocity_kms" not in events.columns:

            return (
                "Relative velocity is not available "
                "in the current scenario."
            )

        df = events.copy()

        df["_velocity"] = pd.to_numeric(
            df["relative_velocity_kms"],
            errors="coerce"
        ).fillna(-1)

        fastest = (
            df.sort_values(
                "_velocity",
                ascending=False
            )
            .iloc[0]
        )

        return (
            f"The highest relative velocity is "
            f"{_safe_float(fastest.get('relative_velocity_kms')):.2f} "
            f"km/s for "
            f"{_safe_text(fastest.get('object_a'))} ↔ "
            f"{_safe_text(fastest.get('object_b'))}. "
            f"Risk score: "
            f"{_safe_float(fastest.get('risk_score')):.1f}/100."
        )

    # ========================================================
    # STARLINK
    # ========================================================

    if "starlink" in q:

        return _analyze_starlink(
            events
        )

    # ========================================================
    # NUMBER OF EVENTS
    # ========================================================

    if _contains_any(
        q,
        [
            "how many events",
            "how many collision",
            "how many collisions",
            "how many conjunction",
            "how many conjunctions",
            "total events",
            "total collisions",
            "total conjunctions",
            "number of events",
            "number of collisions",
            "number of conjunctions",
        ]
    ):

        return (
            f"The current scenario contains "
            f"{len(events)} conjunction event(s) "
            f"among {len(records)} tracked objects."
        )

    # ========================================================
    # NUMBER OF OBJECTS
    # ========================================================

    if _contains_any(
        q,
        [
            "how many objects",
            "how many satellites",
            "number of objects",
            "number of satellites",
            "objects tracked",
            "satellites tracked",
        ]
    ):

        return (
            f"The current analysis is tracking "
            f"{len(records)} objects and has detected "
            f"{len(events)} conjunction event(s)."
        )

    # ========================================================
    # RISK EXPLANATION
    # ========================================================

    if _contains_any(
        q,
        [
            "why risky",
            "why is it risky",
            "why dangerous",
            "why is it dangerous",
            "explain risk",
            "explain the risk",
            "why collision",
            "why is collision dangerous",
            "what makes it risky",
        ]
    ):

        top = _highest_risk_event(events)

        return (
            f"The highest-risk current event is "
            f"{_safe_text(top.get('object_a'))} ↔ "
            f"{_safe_text(top.get('object_b'))}. "
            f"The current model assigns it a risk score of "
            f"{_safe_float(top.get('risk_score')):.1f}/100. "
            f"Key current values are miss distance "
            f"{_safe_float(top.get('miss_distance_km')):.3f} km, "
            f"relative velocity "
            f"{_safe_float(top.get('relative_velocity_kms')):.2f} km/s, "
            f"and TCA "
            f"{_safe_float(top.get('tca_hours_from_now')):.1f} hours."
        )

    # ========================================================
    # OBJECT-SPECIFIC QUERY
    # ========================================================

    object_name = _extract_object_name(
        q,
        events
    )

    if object_name:

        return _analyze_object(
            object_name,
            events
        )

    # ========================================================
    # RISK SCORE
    # ========================================================

    if _contains_any(
        q,
        [
            "risk score",
            "risk level",
            "risk value",
            "collision score",
        ]
    ):

        top = _highest_risk_event(events)

        return (
            f"The highest current risk score is "
            f"{_safe_float(top.get('risk_score')):.1f}/100 "
            f"for "
            f"{_safe_text(top.get('object_a'))} ↔ "
            f"{_safe_text(top.get('object_b'))}. "
            f"Risk category: "
            f"{_safe_text(top.get('risk_category'), 'Unknown')}."
        )

    # ========================================================
    # MISS DISTANCE
    # ========================================================

    if _contains_any(
        q,
        [
            "miss distance",
            "distance between",
            "separation distance",
        ]
    ):

        top = _highest_risk_event(events)

        return (
            f"For the highest-risk current conjunction, "
            f"the predicted miss distance is "
            f"{_safe_float(top.get('miss_distance_km')):.3f} km."
        )

    # ========================================================
    # HELP
    # ========================================================

    if _contains_any(
        q,
        [
            "help",
            "what can you ask",
            "what can i ask",
            "what can you do",
            "commands",
        ]
    ):

        return (
            "You can ask me natural questions such as: "
            "'Which object is most dangerous?', "
            "'What is the highest-risk collision?', "
            "'When is the next TCA?', "
            "'How many critical events?', "
            "'Which collision is closest?', "
            "'What about Starlink?', or "
            "'Tell me about [object name]'."
        )

    # ========================================================
    # SMART DEFAULT
    # ========================================================

    top = _highest_risk_event(events)

    return (
        f"I can only answer from the current scenario data. "
        f"Right now there are {len(records)} tracked objects "
        f"and {len(events)} conjunction events. "
        f"The highest-risk current pair is "
        f"{_safe_text(top.get('object_a'))} ↔ "
        f"{_safe_text(top.get('object_b'))} "
        f"with risk "
        f"{_safe_float(top.get('risk_score')):.1f}/100 "
        f"({_safe_text(top.get('risk_category'), 'Unknown')}). "
        f"Try asking about the highest risk, next TCA, "
        f"closest collision, or a specific object."
    )


# ============================================================
# GEMINI AI
# ============================================================

def _build_session_context(
    session_data: Dict[str, Any]
) -> str:

    events = _get_events(
        session_data
    )

    records = _get_records(
        session_data
    )

    lines = [
        f"Objects tracked: {len(records)}",
        f"Conjunction events: {len(events)}",
    ]

    if events.empty:
        return "\n".join(lines)

    critical = _count_risk(
        events,
        "Critical"
    )

    high = _count_risk(
        events,
        "High"
    )

    medium = _count_risk(
        events,
        "Medium"
    )

    low = _count_risk(
        events,
        "Low"
    )

    lines.append(
        f"Risk counts: Critical={critical}, "
        f"High={high}, Medium={medium}, Low={low}"
    )

    lines.append(
        "\nCurrent conjunction events:"
    )

    # Give Gemini current events, not external data.
    for _, row in (
        _sort_by_risk(events)
        .head(20)
        .iterrows()
    ):

        lines.append(
            _format_event(row)
        )

    return "\n".join(lines)


def _gemini_answer(
    question: str,
    session_data: Dict[str, Any],
    api_key: str
) -> str:

    try:

        from google import genai

    except ImportError:

        raise RuntimeError(
            "google-genai package is not installed."
        )

    client = genai.Client(
        api_key=api_key
    )

    context = _build_session_context(
        session_data
    )

    prompt = f"""
You are Orbital Sentinel's AI Mission Assistant.

You are a SESSION-GROUNDED assistant.

You must answer ONLY from the current scenario data
provided below.

CURRENT SCENARIO
================
{context}

USER QUESTION
=============
{question}

STRICT RULES
============

1. Never invent data.

2. Never use outside satellite information.

3. Never invent a collision.

4. Never invent risk scores.

5. Never invent TCA.

6. Never invent distances.

7. Never invent orbital information.

8. If the user asks:
   "most high collision object",
   interpret this as:
   "Which object/pair has the highest collision risk
   in the current scenario?"

9. If the user asks for the most dangerous object,
   identify the object participating in the highest-risk
   current event.

10. If the user asks for the next collision,
    select the event with the smallest TCA.

11. If the user asks about a specific object,
    answer only using events containing that object.

12. If the requested information is unavailable,
    clearly say that it is unavailable.

13. Be conversational like a mission assistant.

14. Keep answers short and clear.

15. Never provide autonomous spacecraft maneuver commands.

16. This system is advisory only.

Answer using 2-5 sentences unless a list is necessary.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    text = getattr(
        response,
        "text",
        None
    )

    if not text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return text.strip()


# ============================================================
# MAIN PUBLIC FUNCTION
# ============================================================

def answer_question(
    question: str,
    session_data: Dict[str, Any],
    force_template: bool = False
) -> str:
    """
    Main function used by app.py.

    Existing call remains valid:

        answer_question(
            user_query,
            res,
            force_template=False
        )
    """

    question = _safe_text(
        question
    ).strip()

    if not question:

        return (
            "Ask me something about the current "
            "orbital scenario."
        )

    # --------------------------------------------------------
    # FORCE LOCAL ASSISTANT
    # --------------------------------------------------------

    if force_template:

        return _rule_based_answer(
            question,
            session_data
        )

    # --------------------------------------------------------
    # GEMINI API
    # --------------------------------------------------------

    api_key = os.environ.get(
        "GEMINI_API_KEY",
        ""
    ).strip()

    # If API key does not exist, use local AI.
    if not api_key:

        return _rule_based_answer(
            question,
            session_data
        )

    # Try Gemini.
    try:

        return _gemini_answer(
            question,
            session_data,
            api_key
        )

    except Exception as exc:

        warnings.warn(
            f"Gemini unavailable: {exc}. "
            f"Using local session-grounded assistant."
        )

        # Never show API errors to the user.
        # Always provide a useful current-session answer.
        return _rule_based_answer(
            question,
            session_data
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_data = {

        "records": [
            {"name": "STARLINK-1007"},
            {"name": "COSMOS-2251-DEB"},
            {"name": "ISS"},
        ],

        "events": pd.DataFrame([
            {
                "object_a": "STARLINK-1007",
                "object_b": "COSMOS-2251-DEB",
                "miss_distance_km": 2.5,
                "relative_velocity_kms": 7.8,
                "tca_hours_from_now": 36.5,
                "risk_score": 82.4,
                "risk_category": "Critical",
            },
            {
                "object_a": "ISS",
                "object_b": "STARLINK-1007",
                "miss_distance_km": 18.2,
                "relative_velocity_kms": 3.1,
                "tca_hours_from_now": 90.0,
                "risk_score": 41.5,
                "risk_category": "Medium",
            },
        ])
    }

    questions = [
        "hii",
        "most high collision object",
        "which object is most dangerous",
        "which satellite has highest risk",
        "what is the highest collision risk",
        "when is the next collision",
        "what is the next TCA",
        "which collision is closest",
        "how many critical events",
        "how many high risk events",
        "give me a summary",
        "what about Starlink",
        "tell me about STARLINK-1007",
        "why is the collision risky",
    ]

    print("=" * 70)
    print("ORBITAL SENTINEL AI ASSISTANT")
    print("=" * 70)

    for question in questions:

        print()
        print("USER:", question)

        answer = answer_question(
            question,
            test_data,
            force_template=True
        )

        print("AI:", answer)
        print("-" * 70)
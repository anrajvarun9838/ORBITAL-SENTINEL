"""
app.py — Orbital Sentinel | AI-Based Space Situational Awareness System

Main Streamlit application implementing the 5-tab NASA Orbital Sentinel UI.
Builds on the 7-stage pipeline in space_project/ with high physical fidelity,
analytical Pc, Kessler simulation, 3D orbital theater, historical replay,
AI mission assistant, and exportable briefings.
"""

import sys
import os
import time
import json
import pathlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timezone

# Ensure local space_project package is in path
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from space_project.fetch_data import fetch_live_tles
from space_project.sample_data import get_sample_tles, get_historical_replay_tles, classify_regime
from space_project.propagate import propagate_objects, get_regime_map
from space_project.conjunction import detect_conjunctions
from space_project.risk_model import load_model, score_events, train_model
from space_project.pc_analytical import compute_collision_probability, explain_pc_vs_ml, estimate_delta_v
from space_project.ai_report import generate_report, get_report_mode
from space_project.explainability import get_global_feature_importance, explain_single_event
from space_project.kessler_sim import run_kessler_scenario
from space_project.mission_assistant import answer_question
from space_project.anomaly_detection import detect_anomalies
from space_project.export_report import generate_pdf_report, generate_html_report, _HAS_FPDF
from space_project.upload_handler import parse_custom_tle_file, parse_and_predict_custom_cdm_csv

# ─── Page Configuration ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="Orbital Sentinel",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Custom Styling (NASA Orbital Sentinel Theme) ──────────────────────────

CSS_THEME = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root { --os-bg:#050b18; --os-blue:#38a8ff; --os-cyan:#56e0ff; --os-text:#e8f4ff; --os-muted:#8fa8c2; --os-border:rgba(59,160,255,.20); }
.stApp { background:radial-gradient(circle at 8% 0%,rgba(33,121,255,.16),transparent 28%),radial-gradient(circle at 92% 8%,rgba(0,214,255,.10),transparent 25%),linear-gradient(135deg,#030712 0%,#071226 48%,#040a16 100%); color:var(--os-text); font-family:'Inter',sans-serif; }
.stApp::before { content:""; position:fixed; inset:0; pointer-events:none; background-image:linear-gradient(rgba(80,170,255,.025) 1px,transparent 1px),linear-gradient(90deg,rgba(80,170,255,.025) 1px,transparent 1px); background-size:42px 42px; mask-image:linear-gradient(to bottom,black,transparent 90%); }
h1,h2,h3,h4,.orbitron-text { font-family:'Space Grotesk',sans-serif !important; color:var(--os-text) !important; letter-spacing:-.02em; }
h1 { font-weight:800 !important; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,rgba(6,17,35,.98),rgba(3,10,23,.98)) !important; border-right:1px solid var(--os-border); }
section[data-testid="stSidebar"] > div { padding-top:1.5rem; }
.stButton > button { border:1px solid rgba(72,177,255,.45); border-radius:12px; background:linear-gradient(135deg,#0b66c3,#148fd8); color:#fff; font-weight:700; box-shadow:0 8px 28px rgba(24,126,220,.20); transition:.2s ease; }
.stButton > button:hover { border-color:var(--os-cyan); transform:translateY(-1px); box-shadow:0 10px 34px rgba(42,173,255,.32); }
div[data-baseweb="select"] > div,.stTextInput input { background:rgba(8,23,45,.90) !important; border-color:var(--os-border) !important; color:var(--os-text) !important; border-radius:10px !important; }
.hud-card { background:linear-gradient(145deg,rgba(16,43,79,.74),rgba(5,17,35,.88)); border:1px solid var(--os-border); border-radius:16px; padding:18px; margin-bottom:10px; box-shadow:inset 0 1px 0 rgba(255,255,255,.04),0 12px 34px rgba(0,0,0,.22); position:relative; overflow:hidden; }
.hud-card::after { content:""; position:absolute; top:0; left:0; width:38%; height:2px; background:linear-gradient(90deg,var(--os-blue),transparent); }
.hud-title { font-family:'Space Grotesk',sans-serif; font-size:.72rem; color:var(--os-muted); text-transform:uppercase; letter-spacing:.10em; }
.hud-value { font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:800; color:#69c4ff; margin-top:4px; }
.advisory-box { background:linear-gradient(145deg,rgba(9,31,61,.86),rgba(5,15,30,.94)); border:1px solid rgba(56,168,255,.18); border-left:3px solid var(--os-blue); padding:17px; margin:10px 0; border-radius:12px; line-height:1.6; }
.telemetry-feed { background:linear-gradient(180deg,rgba(3,15,31,.96),rgba(2,10,22,.96)); border:1px solid rgba(56,168,255,.30); border-radius:12px; padding:16px; font-family:'Space Grotesk',monospace; font-size:.82rem; color:#70d7ff; max-height:400px; overflow-y:auto; box-shadow:inset 0 0 28px rgba(31,132,220,.06); }
.stTabs [data-baseweb="tab-list"] { gap:6px; background:rgba(6,17,35,.74); padding:7px; border-radius:14px; border:1px solid var(--os-border); }
.stTabs [data-baseweb="tab"] { height:44px; font-family:'Space Grotesk',sans-serif; font-size:.82rem; font-weight:600; color:#89a4bd; border-radius:10px; padding:0 16px; }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg,#0b65bd,#1498dc) !important; color:#fff !important; box-shadow:0 5px 20px rgba(26,140,225,.22); }
[data-testid="stDataFrame"] { border:1px solid var(--os-border); border-radius:12px; overflow:hidden; }
div[data-testid="stStatusWidget"] { border:1px solid var(--os-border); border-radius:12px; }
hr { border-color:rgba(75,145,205,.14) !important; }
a { color:#69c4ff !important; }
</style>
"""


st.markdown(CSS_THEME, unsafe_allow_html=True)


# ─── Helper: Initialize Session State ───────────────────────────────────────

if "results" not in st.session_state:
    st.session_state.results = None

if "assistant_history" not in st.session_state:
    st.session_state.assistant_history = []


# ─── Helper: Execute Pipeline ───────────────────────────────────────────────

def run_pipeline(
    data_source: str,
    window_hours: float,
    step_seconds: float,
    threshold_km: float,
    uploaded_file = None,
):
    """Executes the full 7-stage pipeline with status updates."""
    with st.status("Executing Orbital Sentinel analysis pipeline...", expanded=True) as status:
        # Check if CSV prediction upload
        if data_source == "Upload Custom Dataset (CSV / TLE)" and uploaded_file is not None and uploaded_file.name.endswith(".csv"):
            status.update(label=f"Stage 1/5: Loading & parsing custom CSV dataset ({uploaded_file.name})...")
            file_bytes = uploaded_file.read()
            scored_events, summary = parse_and_predict_custom_cdm_csv(file_bytes, uploaded_file.name)
            
            # Construct mock records for overview
            records = [{"name": f"Obj-{i+1}", "norad_id": 90000+i, "tle1": "", "tle2": "", "_source_group": "uploaded"} for i in range(len(scored_events))]
            trajectories = {}
            regime_map = {}
            anomalies = []
            source_used = f"uploaded_csv_{uploaded_file.name}"
            training_report = load_model()[1]
            
            status.update(label="Custom CSV predictions complete!", state="complete")
            
            st.session_state.results = {
                "records": records,
                "trajectories": trajectories,
                "events": scored_events,
                "training_report": training_report,
                "anomalies": anomalies,
                "regime_map": regime_map,
                "source_used": source_used,
                "explainability": summary["explainability"],
                "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
                "config": {
                    "window_hours": window_hours,
                    "step_seconds": step_seconds,
                    "threshold_km": threshold_km,
                }
            }
            return

        # Standard TLE propagation pipeline
        status.update(label="Stage 1/5: Sourcing orbital element sets (TLEs)...")
        if data_source == "Live (CelesTrak)":
            records, source_used = fetch_live_tles()
        elif data_source == "Historical Replay (2009 Iridium/Cosmos)":
            records = get_historical_replay_tles()
            source_used = "historical_replay"
        elif data_source == "Upload Custom Dataset (CSV / TLE)" and uploaded_file is not None:
            file_bytes = uploaded_file.read()
            records = parse_custom_tle_file(file_bytes, uploaded_file.name)
            source_used = f"uploaded_tle_{uploaded_file.name}"
            if not records:
                st.error("Could not parse valid TLE lines from uploaded file. Falling back to sample data.")
                records = get_sample_tles()
                source_used = "offline_sample"
        else:
            records = get_sample_tles()
            source_used = "offline_sample"

        time.sleep(0.2)

        # Stage 2: Propagation
        status.update(label=f"Stage 2/5: Propagating SGP4 orbits for {len(records)} objects ({window_hours}h window)...")
        trajectories = propagate_objects(records, window_hours=window_hours, step_seconds=step_seconds)
        regime_map = get_regime_map(trajectories)
        time.sleep(0.2)

        # Stage 3: Conjunction Detection
        status.update(label=f"Stage 3/5: Detecting pairwise close approaches (<{threshold_km} km threshold)...")
        events_df = detect_conjunctions(trajectories, threshold_km=threshold_km)
        time.sleep(0.2)

        # Stage 4: Risk Scoring & Analytical Physics
        status.update(label="Stage 4/5: Scoring risk with ML model & Chan analytical 2D Pc...")
        model, training_report = load_model()
        scored_events = score_events(events_df, _model=model)

        # Add Analytical Pc, Delta-V, and Explainability to each event
        pcs = []
        pc_logs = []
        delta_vs = []
        explainability_data = []

        for idx, row in scored_events.iterrows():
            pc_res = compute_collision_probability(
                miss_distance_km=row["miss_distance_km"],
                relative_velocity_kms=row["relative_velocity_kms"],
            )
            pcs.append(pc_res["pc"])
            pc_logs.append(pc_res["pc_log10"])

            dv_res = estimate_delta_v(
                miss_distance_km=row["miss_distance_km"],
                tca_hours=row["tca_hours_from_now"],
            )
            delta_vs.append(dv_res["delta_v_ms"])

            exp_res = explain_single_event(row.to_dict(), model)
            explainability_data.append(exp_res)

        scored_events["pc"] = pcs
        scored_events["pc_log10"] = pc_logs
        scored_events["delta_v_ms"] = delta_vs

        # Stage 5: Anomaly Detection
        status.update(label="Stage 5/5: Flagging statistical orbital anomalies...")
        anomalies = detect_anomalies(records, trajectories)

        status.update(label="Orbital Sentinel analysis complete!", state="complete")

    # Store results in session state
    st.session_state.results = {
        "records": records,
        "trajectories": trajectories,
        "events": scored_events,
        "training_report": training_report,
        "anomalies": anomalies,
        "regime_map": regime_map,
        "source_used": source_used,
        "explainability": explainability_data,
        "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
        "config": {
            "window_hours": window_hours,
            "step_seconds": step_seconds,
            "threshold_km": threshold_km,
        }
    }


# ─── Sidebar Configuration Controls ────────────────────────────────────────

with st.sidebar:
    st.image("https://raw.githubusercontent.com/streamlit/streamlit/main/docs/static/logo.png", width=120)
    st.markdown("### 🛰️ ORBITAL SENTINEL")
    st.markdown("---")

    data_source = st.radio(
        "Data Source",
        ["Offline Sample (40 objects)", "Live (CelesTrak)", "Historical Replay (2009 Iridium/Cosmos)", "Upload Custom Dataset (CSV / TLE)"],
        help="Select offline baseline sample, live satellite elements, historical collision data, or upload your own CSV/TLE dataset."
    )

    uploaded_file = None
    if data_source == "Upload Custom Dataset (CSV / TLE)":
        uploaded_file = st.file_uploader(
            "Upload Dataset (.csv or .txt / .tle)",
            type=["csv", "txt", "tle"],
            help="Upload a CSV with CDM features (miss_distance, relative_speed, time_to_tca, c_object_type) or a text file containing TLE lines."
        )

    window_days = st.slider("Prediction Window (days)", 1.0, 7.0, 7.0, 0.5)
    window_hours = window_days * 24.0

    step_seconds = st.select_slider(
        "Propagation Step (seconds)",
        options=[30, 60, 120, 300],
        value=60,
        help="Smaller step increases fidelity but requires slightly more computation."
    )

    threshold_km = st.slider(
        "Conjunction Threshold (km)",
        5.0, 150.0, 50.0, 5.0,
        help="Distance threshold for flagging close approaches. Higher threshold captures more candidate events."
    )

    st.markdown("---")

    if st.button("🚀 RUN ANALYSIS", use_container_width=True, type="primary"):
        run_pipeline(data_source, window_hours, step_seconds, threshold_km, uploaded_file)

    st.markdown("---")
    st.caption("ℹ️ **ORBITAL SENTINEL ADVISORY**: Collision probability (Pc) & ML risk are statistical estimates bounded by tracking uncertainty.")


# ─── Header: UTC Clock & Status Bar ─────────────────────────────────────────

utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("ORBITAL SENTINEL")
    st.caption("AN AI-BASED SPACE SITUATIONAL AWARENESS SYSTEM | ORBITAL OBJECT MONITORING & COLLISION RISK ANALYSIS")

with col_head2:
    st.markdown(f"""
    <div style="background-color: #030d1a; border: 1px solid #0A84FF; padding: 10px; border-radius: 6px; text-align: center;">
        <div style="font-size: 0.7rem; color: #8b949e; font-family: 'Orbitron';">SYSTEM TIME</div>
        <div style="font-size: 1.1rem; color: #30d158; font-family: 'Share Tech Mono'; font-weight: bold;">{utc_now}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# ─── Persistent Metric HUD (Header Metrics) ─────────────────────────────────

if st.session_state.results is not None:
    res = st.session_state.results
    records = res["records"]
    events_df = res["events"]
    anomalies = res["anomalies"]

    crit_count = len(events_df[events_df["risk_category"] == "Critical"]) if len(events_df) > 0 else 0
    high_count = len(events_df[events_df["risk_category"] == "High"]) if len(events_df) > 0 else 0
    med_count = len(events_df[events_df["risk_category"] == "Medium"]) if len(events_df) > 0 else 0

    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

    with col_m1:
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-title">Tracked Objects</div>
            <div class="hud-value">{len(records)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-title">Conjunction Events</div>
            <div class="hud-value">{len(events_df)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m3:
        crit_color = "#ff453a" if crit_count > 0 else "#0A84FF"
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-title">Critical Threats</div>
            <div class="hud-value" style="color: {crit_color};">{crit_count}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m4:
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-title">High Threats</div>
            <div class="hud-value" style="color: #ff9f0a;">{high_count}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m5:
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-title">Anomalies Flagged</div>
            <div class="hud-value" style="color: #ffd60a;">{len(anomalies)}</div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("👋 Welcome to Orbital Sentinel. Select options in the sidebar and click **🚀 RUN ANALYSIS** to initialize orbit propagation and collision prediction.")
    # Show pre-run initial state placeholder
    st.stop()


# ─── Main 5-Tab Dashboard Layout ────────────────────────────────────────────

tab_mission, tab_3d, tab_threat, tab_telemetry, tab_ai = st.tabs([
    "🌐 ORBITAL OVERVIEW",
    "🌍 3D ORBITAL VIEW",
    "⚠️ THREAT ANALYSIS",
    "📡 TELEMETRY",
    "🤖 AI ORBITAL ADVISOR"
])


# =============================================================================
# TAB 1: 🌐 ORBITAL OVERVIEW
# =============================================================================
with tab_mission:
    st.subheader("OVERVIEW & CONSTELLATION HEALTH")

    col_t1_left, col_t1_right = st.columns([2, 1])

    with col_t1_left:
        st.markdown("#### 🛰️ Orbital Risk Exposure")

        # Group events by source group
        events_df = res["events"]
        if len(events_df) > 0 and "object_a_group" in events_df.columns and "object_b_group" in events_df.columns:
            group_risk = []
            for group in ["starlink", "stations", "iridium33_debris", "cosmos2251_debris", "fengyun1c_debris", "gps", "oneweb", "uploaded"]:
                group_events = events_df[
                    (events_df["object_a_group"] == group) | (events_df["object_b_group"] == group)
                ]
                if len(group_events) > 0:
                    max_risk = group_events["risk_score"].max()
                    avg_risk = group_events["risk_score"].mean()
                    group_risk.append({
                        "Orbital Group": group.upper(),
                        "Total Events": len(group_events),
                        "Max Risk Score": round(max_risk, 1),
                        "Avg Risk Score": round(avg_risk, 1),
                    })

            if group_risk:
                st.dataframe(pd.DataFrame(group_risk), use_container_width=True)
            else:
                st.write("No major constellation groups flagged in events.")
        elif len(events_df) > 0:
            st.write(f"Processed {len(events_df)} custom dataset events.")
        else:
            st.write("No conjunction events detected in the current run.")

        st.markdown("---")
        st.markdown("#### ⚠️ Flagged Orbital Anomalies (Statistical Z-Score)")
        anomalies = res["anomalies"]
        if anomalies:
            for a in anomalies:
                with st.expander(f"⚠️ {a['name']} (NORAD {a['norad_id']}) — Max Z-Score: {a['max_z_score']}"):
                    st.write(f"**Source Group:** `{a['source_group']}`")
                    for flag in a["flags"]:
                        st.markdown(f"• `{flag}`")
                    st.caption(a["disclaimer"])
        else:
            st.success("Zero orbital anomalies flagged (all objects within 2.0σ of population norm).")

    with col_t1_right:
        st.markdown("#### 🌍 Orbital Regime Breakdown")
        regime_map = res["regime_map"]
        if regime_map:
            counts = pd.Series(regime_map.values()).value_counts().reset_index()
            counts.columns = ["Regime", "Count"]

            fig_regime = go.Figure(data=[go.Pie(
                labels=counts["Regime"],
                values=counts["Count"],
                hole=.4,
                marker=dict(colors=["#0A84FF", "#ff9f0a", "#30d158", "#bf5af2"])
            )])
            fig_regime.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c9d1d9"),
                margin=dict(l=10, r=10, t=10, b=10),
                height=250
            )
            st.plotly_chart(fig_regime, use_container_width=True)

        st.markdown("#### 📌 Quick System Status")
        st.write(f"• **Data Mode:** `{res['source_used']}`")
        st.write(f"• **Run Timestamp:** `{res['timestamp']}`")
        st.write(f"• **Prediction Window:** `{res['config']['window_hours']} hours`")
        st.write(f"• **Conjunction Threshold:** `{res['config']['threshold_km']} km`")


# =============================================================================
# TAB 2: 🌍 3D ORBITAL VIEW
# =============================================================================
with tab_3d:
    st.subheader("3D ORBIT TRAJECTORY & ENCOUNTER THEATER")

    trajectories = res["trajectories"]
    events_df = res["events"]

    col_3d_controls, col_3d_view = st.columns([1, 3])

    with col_3d_controls:
        st.markdown("#### Visualization Controls")
        show_all_orbits = st.checkbox("Show All Orbit Trails", value=True)
        show_earth_sphere = st.checkbox("Render 3D Earth", value=True)
        highlight_event = st.selectbox(
            "Focus on Close-Approach Pair",
            options=["None"] + [f"#{i+1}: {row['object_a']} ↔ {row['object_b']}" for i, row in events_df.iterrows()] if len(events_df) > 0 else ["None"]
        )

    with col_3d_view:
        fig_3d = go.Figure()

        # 1. Add Earth Sphere
        if show_earth_sphere:
            r_earth = 6378.137
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 30)
            x_earth = r_earth * np.outer(np.cos(u), np.sin(v))
            y_earth = r_earth * np.outer(np.sin(u), np.sin(v))
            z_earth = r_earth * np.outer(np.ones(np.size(u)), np.cos(v))

            fig_3d.add_trace(go.Surface(
                x=x_earth, y=y_earth, z=z_earth,
                colorscale=[[0, '#041c32'], [0.5, '#062c54'], [1, '#0A84FF']],
                showscale=False,
                opacity=0.8,
                hoverinfo='skip',
                name="Earth"
            ))

        # 2. Add Orbit Trails
        if show_all_orbits:
            for nid, traj in trajectories.items():
                pos = traj["positions"]
                fig_3d.add_trace(go.Scatter3d(
                    x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
                    mode='lines',
                    line=dict(width=2, color='#0A84FF'),
                    name=traj['name'],
                    hoverinfo='text',
                    text=f"Object: {traj['name']}<br>Regime: {traj['regime']}<br>Alt: {traj['alt_km']} km"
                ))

        # 3. Highlight Conjunction Lines
        if len(events_df) > 0:
            for idx, row in events_df.iterrows():
                id_a = row.get("object_a_id", None)
                id_b = row.get("object_b_id", None)

                if id_a is not None and id_b is not None and id_a in trajectories and id_b in trajectories:
                    pos_a = trajectories[id_a]["positions"]
                    pos_b = trajectories[id_b]["positions"]

                    # Draw red line between closest approach points
                    min_idx = np.argmin(np.linalg.norm(pos_a - pos_b, axis=1))

                    line_color = "#ff3b30" if row["risk_category"] == "Critical" else "#ff9500" if row["risk_category"] == "High" else "#ffcc00"

                    fig_3d.add_trace(go.Scatter3d(
                        x=[pos_a[min_idx, 0], pos_b[min_idx, 0]],
                        y=[pos_a[min_idx, 1], pos_b[min_idx, 1]],
                        z=[pos_a[min_idx, 2], pos_b[min_idx, 2]],
                        mode='lines+markers',
                        line=dict(color=line_color, width=6),
                        marker=dict(size=4, color=line_color),
                        name=f"Threat: {row['object_a']} ↔ {row['object_b']}",
                        text=f"Threat Pair: {row['object_a']} ↔ {row['object_b']}<br>Miss Distance: {row['miss_distance_km']} km"
                    ))

        fig_3d.update_layout(
            scene=dict(
                xaxis=dict(title="X (km)", backgroundcolor="#000811", gridcolor="#1a2942"),
                yaxis=dict(title="Y (km)", backgroundcolor="#000811", gridcolor="#1a2942"),
                zaxis=dict(title="Z (km)", backgroundcolor="#000811", gridcolor="#1a2942"),
                aspectmode='data'
            ),
            paper_bgcolor="#000811",
            plot_bgcolor="#000811",
            margin=dict(l=0, r=0, t=0, b=0),
            height=600,
            showlegend=False
        )

        st.plotly_chart(fig_3d, use_container_width=True)


# =============================================================================
# TAB 3: ⚠️ THREAT ANALYSIS
# =============================================================================
with tab_threat:
    st.subheader("CONJUNCTION RISK TABLE & MODEL VALIDATION")

    events_df = res["events"]

    if len(events_df) > 0:
        st.markdown("#### 📊 Scored Close-Approach Events")

        # Display table with formatting
        display_df = events_df.copy()
        display_df["Risk Category"] = display_df["risk_category"]
        display_df["ML Risk (0-100)"] = display_df["risk_score"]
        display_df["Analytical Pc (log10)"] = display_df["pc_log10"]
        display_df["Miss Distance (km)"] = display_df["miss_distance_km"]
        display_df["Rel. Vel (km/s)"] = display_df["relative_velocity_kms"]
        display_df["TCA (hours)"] = display_df["tca_hours_from_now"]
        display_df["Maneuver Δv (m/s)"] = display_df["delta_v_ms"]

        cols_to_show = [
            "object_a", "object_b", "Risk Category", "ML Risk (0-100)",
            "Analytical Pc (log10)", "Miss Distance (km)", "Rel. Vel (km/s)",
            "TCA (hours)", "Maneuver Δv (m/s)"
        ]

        st.dataframe(display_df[cols_to_show], use_container_width=True)

        st.markdown("---")

        # Explainability drilldown
        st.markdown("#### 🔍 Explainable Risk Breakdown & Analytical Pc vs ML")
        selected_event_idx = st.selectbox("Select Event for Deep Breakdown", range(len(events_df)), format_func=lambda i: f"#{i+1}: {events_df.iloc[i]['object_a']} ↔ {events_df.iloc[i]['object_b']}")

        selected_row = events_df.iloc[selected_event_idx]
        exp_data = res["explainability"][selected_event_idx]

        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            st.markdown(f"**Feature Contributions for Event #{selected_event_idx+1}**")
            cont_df = pd.DataFrame([
                {
                    "Feature": d["label"],
                    "Value": d["value"],
                    "Baseline": d["baseline"],
                    "Contribution %": f"{d['contribution_pct']}%",
                    "Direction": d["direction"]
                } for d in exp_data["contributions"].values()
            ])
            st.table(cont_df)
            st.caption(f"**Dominant Risk Factor:** `{exp_data['dominant_factor']}`")

        with col_exp2:
            st.markdown("**Analytical Pc vs ML Score Disagreement Check**")
            disagreement_explanation = explain_pc_vs_ml(
                pc=selected_row["pc"],
                ml_score=selected_row["risk_score"],
                pc_log10=selected_row["pc_log10"]
            )
            st.info(disagreement_explanation)

            st.markdown("**Avoidance Burn Estimate (Δv)**")
            st.write(f"• **Delta-V required:** `{selected_row['delta_v_ms']:.4f} m/s`")
            st.caption("Simplified impulse approximation for 5 km safe target separation.")

    else:
        st.warning("No conjunction events flagged below the distance threshold.")

    st.markdown("---")

    # ── Model Validation Panel ──────────────────────────────────────────────
    st.markdown("### 📈 Model Validation & Honest Error Reporting")
    tr = res["training_report"]

    val_col1, val_col2, val_col3 = st.columns(3)

    with val_col1:
        st.metric("Test Set MAE", f"{tr.get('mae', 'N/A')} pts", help="Observed Mean Absolute Error on held-out test data.")
        st.caption("MAE ~25 points is an honest, expected result for a 4-feature model on noisy Pc data.")

    with val_col2:
        st.metric("Training Samples", f"{tr.get('n_train', 'N/A')}")
        st.caption(f"Source: `{tr.get('data_source', 'N/A')}`")

    with val_col3:
        st.metric("Test Samples", f"{tr.get('n_test', 'N/A')}")
        st.caption("80/20 train/test split")

    if "feature_importances" in tr:
        fi_df = pd.DataFrame(list(tr["feature_importances"].items()), columns=["Feature", "Importance"])
        fig_fi = go.Figure(go.Bar(x=fi_df["Importance"], y=fi_df["Feature"], orientation='h', marker_color='#0A84FF'))
        fig_fi.update_layout(
            title="Global Feature Importances (GradientBoostingRegressor)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c9d1d9"), height=250, margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig_fi, use_container_width=True)


# =============================================================================
# TAB 4: 📡 TELEMETRY
# =============================================================================
with tab_telemetry:
    st.subheader("ORBITAL TELEMETRY & KESSLER SCENARIO SIMULATOR")

    col_tel_left, col_tel_right = st.columns([1, 1])

    with col_tel_left:
        st.markdown("#### 📡 Propagated State Vector Stream")
        trajectories = res["trajectories"]

        telemetry_lines = []
        for nid, traj in list(trajectories.items())[:5]:
            pos = traj["positions"][0]  # Initial state
            telemetry_lines.append(f"[{traj['name']:20s}] POS X:{pos[0]:10.2f} Y:{pos[1]:10.2f} Z:{pos[2]:10.2f} km | ALT: {traj['alt_km']} km [{traj['regime']}]")

        telemetry_text = "\n".join(telemetry_lines)
        st.markdown(f'<div class="telemetry-feed">{telemetry_text}</div>', unsafe_allow_html=True)

    with col_tel_right:
        st.markdown("#### 💥 Kessler Syndrome Cascade Simulator")
        st.caption("Toggleable 'What If' panel modeling secondary debris generation from hypothetical impact.")

        sim_years = st.slider("Cascade Simulation Horizon (Years)", 10, 100, 50)

        if len(events_df) > 0:
            top_event = events_df.iloc[0]
            kess_res = run_kessler_scenario(
                miss_distance_km=top_event["miss_distance_km"],
                relative_velocity_kms=top_event["relative_velocity_kms"],
                sim_years=sim_years
            )

            frag = kess_res["fragmentation"]
            cas = kess_res["cascade"]

            st.write(f"• **Collision Type:** `{frag['collision_type']}`")
            st.write(f"• **Kinetic Energy:** `{frag['kinetic_energy_mj']} MJ`")
            st.write(f"• **Fragments (>10cm):** `{frag['fragments_gt_10cm']}`")
            st.write(f"• **50-Year Pop Increase:** `{cas['population_increase_factor']}x`")

            st.warning(cas["disclaimer"])


# =============================================================================
# TAB 5: 🤖 AI ORBITAL ADVISOR
# =============================================================================
with tab_ai:
    st.subheader("AI ORBITAL ADVISOR, BRIEFING EXPORT & HISTORICAL REPLAY")

    col_ai_l, col_ai_r = st.columns([1, 1])

    with col_ai_l:
        st.markdown("#### 💬 Session-Grounded Orbital AI Assistant")

        # Chat display
        for msg in st.session_state.assistant_history:
            st.markdown(f"**{msg['role'].upper()}:** {msg['content']}")

        user_query = st.text_input("Ask about current session events (e.g. 'which pair is highest risk?'):")
        if st.button("Send Query"):
            if user_query:
                ans = answer_question(user_query, res, force_template=False)
                st.session_state.assistant_history.append({"role": "user", "content": user_query})
                st.session_state.assistant_history.append({"role": "assistant", "content": ans})
                st.rerun()

        st.markdown("---")

        st.markdown("#### 📋 Exportable Orbital Safety Briefing")
        reports_dict = {}
        for i in range(min(5, len(events_df))):
            reports_dict[i] = generate_report(events_df.iloc[i].to_dict(), force_template=True)

        if _HAS_FPDF:
            pdf_bytes = generate_pdf_report(events_df,reports_dict,res["training_report"])

            # Streamlit requires bytes, not bytearray
            if isinstance(pdf_bytes, bytearray):
                pdf_bytes = bytes(pdf_bytes)

            elif isinstance(pdf_bytes, memoryview):
                pdf_bytes = pdf_bytes.tobytes()

            elif not isinstance(pdf_bytes, bytes):
                pdf_bytes = bytes(pdf_bytes)

            st.download_button(
                label="📄 Download PDF Briefing",
                data=pdf_bytes,
                file_name=(
                    f"Orbital_Sentinel_Briefing_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                ),
                mime="application/pdf",
                use_container_width=True
            )

        html_str = generate_html_report(events_df, reports_dict, res["training_report"])
        st.download_button(
            label="🌐 Download HTML Briefing",
            data=html_str,
            file_name=f"Orbital_Sentinel_Briefing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            use_container_width=True
        )

    with col_ai_r:
        st.markdown("#### 📜 AI Risk Advisories")

        if len(events_df) > 0:
            for idx in range(min(3, len(events_df))):
                event_dict = events_df.iloc[idx].to_dict()
                advisory = generate_report(event_dict, force_template=False)

                with st.expander(f"Advisory #{idx+1}: {event_dict['object_a']} ↔ {event_dict['object_b']} [{event_dict['risk_category']}]", expanded=(idx == 0)):
                    st.markdown(f'<div class="advisory-box">{advisory}</div>', unsafe_allow_html=True)
        else:
            st.write("No events to generate advisories for.")

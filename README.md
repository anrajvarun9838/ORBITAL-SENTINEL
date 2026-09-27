# 🛰️ Orbital Sentinel

> **AI-assisted Space Situational Awareness (SSA) dashboard for orbital object monitoring, conjunction detection, and collision-risk analysis.**

Orbital Sentinel is a Streamlit-based research and demonstration platform that brings together **orbital mechanics, machine learning, analytical collision-probability estimation, anomaly detection, interactive visualization, and AI-assisted reporting** in a single interface.

The system is designed to make orbital-risk analysis easier to inspect and understand while clearly treating its outputs as **advisory estimates rather than certified operational predictions**.

---

## ✨ What It Does

Orbital Sentinel processes orbital-object data through a multi-stage analysis pipeline:

1. **Acquire orbital data** from live CelesTrak TLE feeds, offline sample data, historical replay data, or user-uploaded datasets.
2. **Propagate satellite trajectories** with SGP4 over a configurable prediction window.
3. **Detect close approaches** by scanning pairwise object separation over time.
4. **Score conjunction risk** with a Gradient Boosting Regressor.
5. **Estimate analytical collision probability (Pc)** and approximate maneuver Δv.
6. **Flag orbital anomalies** for additional inspection.
7. **Explain and visualize results** through an interactive mission-style dashboard.
8. **Generate AI-assisted mission summaries** grounded in the current scenario.
9. **Export mission briefings** as HTML or PDF.

---

## 🚀 Core Capabilities

### 🌐 Orbital Overview

The main dashboard provides a high-level view of the current scenario, including:

- Tracked-object counts
- Conjunction-event counts
- Risk-category summaries
- Orbital-regime information
- Exposure and threat visualizations
- Scenario configuration and analysis timestamp

### 🌍 3D Orbital Visualization

Explore propagated trajectories in an interactive 3D orbital environment using Plotly.

The dashboard can display:

- Orbital tracks
- Object positions
- Earth-centered views
- Orbital-regime information
- Configurable visualization windows

### ⚠️ Threat Analysis

For detected conjunctions, Orbital Sentinel calculates and presents:

- Miss distance
- Relative velocity
- Time to closest approach (TCA)
- ML-derived risk score
- Risk category
- Analytical collision probability
- Approximate Δv estimate
- Event-level explainability information

Risk thresholds are configurable, allowing the user to control how aggressively close approaches are surfaced.

### 📡 Telemetry & Historical Analysis

The application supports several data paths:

| Data source | Purpose |
|---|---|
| Offline sample | Run the application without external data access |
| Live CelesTrak | Fetch current TLEs for configured public object groups |
| Historical replay | Reproduce a historical Iridium/Cosmos conjunction scenario |
| Custom TLE | Propagate user-supplied TLE data |
| Custom CSV/CDM | Score user-supplied conjunction records |

When live TLE retrieval fails, the application can fall back to offline sample data.

### 🤖 AI Orbital Advisor

The built-in mission assistant can answer questions about the **current scenario**, such as:

- Which event currently has the highest risk?
- Which conjunction is closest?
- When is the next TCA?
- How many critical/high-risk events are present?
- What is the miss distance of the highest-risk event?
- What can be inferred about a specific tracked object?

The assistant is designed to remain grounded in the current session data. When a Gemini API key is unavailable or the service fails, the application falls back to a local rule-based assistant.

### 📄 Mission Briefings

Orbital Sentinel can export analysis results as:

- **HTML mission briefings**
- **PDF mission briefings**

Exports can include:

- Top conjunction events
- Risk scores
- Miss distances
- Relative velocities
- TCA estimates
- Analytical Pc
- Estimated Δv
- AI advisories
- Model-validation metadata

---

## 🧠 Analysis Pipeline

```text
                ┌─────────────────────────┐
                │   Orbital Data Sources  │
                │ CelesTrak / Sample /    │
                │ Historical / Uploads    │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │   TLE / Dataset Parsing │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │   SGP4 Orbit Propagation│
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │ Conjunction Detection   │
                │ Pairwise Close Approach │
                └────────────┬────────────┘
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
      ┌────────────────────┐    ┌────────────────────┐
      │ ML Risk Scoring    │    │ Analytical Physics │
      │ Gradient Boosting  │    │ Pc + Approx. Δv    │
      └──────────┬─────────┘    └──────────┬─────────┘
                 └────────────┬────────────┘
                              ▼
                 ┌─────────────────────────┐
                 │ Anomaly Detection &     │
                 │ Explainability          │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Interactive SSA         │
                 │ Mission Dashboard       │
                 └────────────┬────────────┘
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
        ┌──────────────────┐      ┌──────────────────┐
        │ AI Mission       │      │ PDF / HTML       │
        │ Assistant        │      │ Briefing Export  │
        └──────────────────┘      └──────────────────┘
```

---

## 🧪 Risk Model

The project uses a **Gradient Boosting Regressor** to produce a normalized risk score on a **0–100 scale**.

The model uses four primary features:

| Feature | Description |
|---|---|
| `miss_distance_km` | Predicted minimum separation distance |
| `relative_velocity_kms` | Relative velocity at/around TCA |
| `tca_hours_from_now` | Time remaining until closest approach |
| `size_class` | Object-size proxy derived from object type |

The repository supports two training paths:

- **ESA Kelvins CDM data** when the expected training/test datasets are present.
- **Physics-based synthetic training data** as an offline fallback.

The trained model is stored with Joblib and accompanied by a training report containing validation metadata and feature importances.

> **Important:** The ML score is an analytical/demo indicator, not a substitute for operational conjunction assessment systems or certified collision-probability products.

---

## 📐 Analytical Collision Probability

In addition to ML scoring, Orbital Sentinel computes an analytical **collision probability (Pc)** estimate using the project's implemented Chan-style 2D approach.

This provides a second, physics-oriented signal alongside the machine-learning score.

The application also estimates a small **Δv maneuver requirement** from miss distance and time-to-TCA as a scenario-level approximation.

These quantities should be interpreted as **research/demo estimates**, not operational maneuver recommendations.

---

## 🛰️ Data Sources

### Live TLEs

Orbital Sentinel can fetch public TLE data from **CelesTrak** for configured groups such as:

- Stations
- Starlink
- Iridium-33 debris
- Cosmos-2251 debris

The live fetcher deduplicates objects by NORAD ID and falls back to sample data when all requested network fetches fail.

### Historical Replay

A built-in historical scenario supports replay-style analysis around the **2009 Iridium 33 / Cosmos 2251** collision context.

### User Uploads

The application supports:

- TLE text files
- TXT/TLE-style files
- CSV conjunction/CDM-style datasets

For custom CSV scoring, the expected fields are based on the repository's CDM processing logic.

---

## 🗂️ Project Structure

```text
ORBITAL-SENTINEL/
├── app.py
├── requirements.txt
├── test_dataset.py
├── .gitignore
├── README.md
│
└── space_project/
    ├── __init__.py
    ├── ai_report.py
    ├── anomaly_detection.py
    ├── conjunction.py
    ├── explainability.py
    ├── export_report.py
    ├── fetch_data.py
    ├── kessler_sim.py
    ├── mission_assistant.py
    ├── pc_analytical.py
    ├── propagate.py
    ├── risk_model.py
    ├── sample_data.py
    ├── upload_handler.py
    │
    └── models/
        ├── risk_model.joblib
        └── training_report.json
```

---

## 🛠️ Tech Stack

| Area | Technologies |
|---|---|
| Dashboard | Streamlit |
| Data processing | Python, Pandas, NumPy |
| Visualization | Plotly |
| Orbit propagation | SGP4, Skyfield |
| Machine learning | Scikit-learn |
| Model persistence | Joblib |
| Live data | CelesTrak + Requests |
| AI assistant | Google Gemini API (optional) |
| PDF export | fpdf2 |
| Web report | HTML/CSS |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/anrajvarun9838/ORBITAL-SENTINEL.git
cd ORBITAL-SENTINEL
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the application

```bash
streamlit run app.py
```

The application will open in your browser with the Orbital Sentinel dashboard.

---

## 🔐 Optional Gemini Integration

The AI mission assistant works without an API key by using its local, scenario-grounded response logic.

To enable the Gemini-backed assistant, configure:

```bash
GEMINI_API_KEY=your_api_key_here
```

For local development, set this as an environment variable rather than hard-coding a secret into source files.

---

## 🧭 Using the Dashboard

1. Open the Streamlit application.
2. Choose a **data source** from the sidebar.
3. Configure:
   - Prediction window
   - Propagation step
   - Conjunction threshold
4. Click **Run Analysis**.
5. Inspect:
   - Orbital overview
   - 3D trajectories
   - Threat analysis
   - Telemetry
   - AI Orbital Advisor
6. Export a mission briefing when needed.

For a first run, use the **Offline Sample** source so the application can be explored without depending on external TLE retrieval.

---

## 📊 Custom CSV Inputs

The custom CSV path is designed around conjunction-data fields used by the project's CDM processing logic.

At minimum, provide values corresponding to:

```text
miss_distance
relative_speed
time_to_tca
c_object_type
```

The project maps these into its internal feature representation:

```text
miss_distance   → miss_distance_km
relative_speed  → relative_velocity_kms
time_to_tca     → tca_hours_from_now
c_object_type   → size_class
```

For exact dataset requirements, inspect `space_project/upload_handler.py` and `space_project/risk_model.py`.

---

## 🧩 Design Principles

### Physics + ML

Machine learning is used alongside orbital-mechanics calculations rather than as a standalone black box.

### Explainability

Risk events expose their underlying inputs and supporting analysis so that results can be inspected rather than treated as unexplained labels.

### Offline Resilience

The project includes sample and synthetic-data fallbacks so core demonstrations can continue without live external services.

### Advisory-Only Operation

The system is intentionally positioned as an analytical and educational platform. It does not autonomously command spacecraft or issue certified maneuver directives.

---

## ⚠️ Limitations & Responsible Use

Orbital Sentinel is a **research / demonstration system**, not an operational space-traffic-management service.

Results may be affected by:

- TLE accuracy and age
- Propagation assumptions
- Simplified conjunction detection
- Limited feature sets used by the ML model
- Synthetic training fallback when real CDM data is unavailable
- Approximate collision-probability assumptions
- Simplified Δv estimation
- Missing covariance and higher-fidelity state information

**Do not use the dashboard's scores, Pc estimates, or Δv estimates as sole inputs for real spacecraft maneuver decisions.**

---

## 🔭 Future Work

Potential directions for extending Orbital Sentinel include:

- Higher-fidelity state and covariance handling
- More complete CDM ingestion
- Improved conjunction screening and TCA refinement
- Model calibration and uncertainty quantification
- Additional orbital data providers
- Multi-model risk comparison
- Automated experiment tracking
- More extensive unit/integration testing
- CI/CD and reproducible evaluation pipelines
- Expanded debris-environment and Kessler-syndrome simulations
- Operational-style event timelines and alerting

---

## 🤝 Contributing

Contributions are welcome.

A typical workflow is:

```bash
git checkout -b feature/your-change
git add .
git commit -m "docs: improve Orbital Sentinel documentation"
git push origin feature/your-change
```

Then open a pull request against `main`.

For larger changes, please describe:

- What changed
- Why it changed
- How it was tested
- Any new dependencies or assumptions

---

## 👥 Contributors

Orbital Sentinel is a collaborative project. See the repository's GitHub contributors page for the current contributor list and contribution history.

---

## 📜 License

Check the repository for the current license configuration before redistributing or using the project in another context.

---

## ⭐ Project Status

Orbital Sentinel is an evolving prototype focused on combining **space situational awareness, orbital mechanics, machine learning, and interactive decision-support tooling** in one accessible research platform.

If you find the project useful, consider starring the repository and contributing improvements.

---

<p align="center">
  <b>🛰️ ORBITAL SENTINEL</b><br>
  AI-assisted orbital monitoring & collision-risk analysis
</p>
```


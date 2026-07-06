# AI-Assisted Network Optimization Dashboard

TeleOptAI is a portfolio project that combines a telecom network simulator, a Flask backend, and a browser dashboard to demonstrate AI-assisted network monitoring and resource optimization.

It is designed to show how a live telecom workload can be observed, diagnosed, and adjusted in real time through automated control logic. The current implementation focuses on:

- real-time KPI collection for throughput, latency, packet loss, and utilization,
- anomaly detection for congestion-style conditions,
- automatic resource adjustment through an orchestrator,
- manual vs automated control mode switching,
- scenario presets for normal, congestion, optimized, and link-failure cases.

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Architecture](#architecture)
- [Core Features](#core-features)
- [Network Topology](#network-topology)
- [Setup Guide](#setup-guide)
- [API Reference](#api-reference)
- [Scenarios](#scenarios)
- [KPI and Anomaly Model](#kpi-and-anomaly-model)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Notes](#notes)
- [License](#license)

## What This Project Does

The project simulates a small telecom topology with two UEs, an eNodeB, a core router, and a server. The simulator computes KPIs from link bandwidth, queue sizes, scheduling policy, and traffic load rather than generating random values.

The dashboard lets you:

- start and stop the simulation,
- switch between manual configuration and automated allocation,
- seed the system with scenario presets,
- watch live KPI charts and topology updates,
- inspect anomaly alerts and parameter change logs.

On the backend, a master orchestrator reads the live KPI stream, applies anomaly detection, generates optimization decisions, and updates the simulator when automation is enabled.

## Architecture

```text
┌───────────────────────────────────────────────────────────────┐
│                         Browser UI                            │
│  frontend/index.html · frontend/script.js · frontend/style.css│
│  Live KPI cards · charts · topology view · scenario presets   │
└───────────────────────────────┬───────────────────────────────┘
                                │ REST / JSON
                                ▼
┌────────────────────────────────────────────────────────────────┐
│                     Flask Backend API                          │
│                      backend/api.py                            │
│  /api/start · /api/stop · /api/kpis · /api/parameters          │
│  /api/control-mode · /api/topology · /api/aggregated · /api/log│
└───────────────────────┬────────────────────────────────────────┘
                        │
        ┌───────────────┴────────────────┬─────────────────┐
        ▼                                ▼                 ▼
┌───────────────────────┐   ┌───────────────────────┐   ┌───────────────────────┐
│ Network Simulator     │   │ Master Orchestrator   │   │ Anomaly Agent         │
│ src/simulation/       │   │ src/orchestration/    │   │ src/agents/           │
│ Mock or Mininet path  │   │ KPI -> decision -> act│   │ Rule-based alerts     │
└───────────────────────┘   └───────────────────────┘   └───────────────────────┘
```

```text
Frontend (frontend/)
  index.html · script.js · style.css
  - control mode toggle
  - scenario presets
  - live KPI cards and charts
  - topology view and change log

Backend (backend/api.py)
  Flask REST API
  - /api/start, /api/stop
  - /api/kpis, /api/topology, /api/parameters
  - /api/control-mode
  - /api/aggregated, /api/log

Orchestration (src/orchestration/)
  MasterOrchestrator
  - collects KPI snapshots
  - runs anomaly detection
  - generates optimization decisions
  - applies resource updates in automated mode

Simulation (src/simulation/)
  MockNetworkSimulator on Windows/macOS
  MininetNetworkSimulator on Linux when Mininet is available
```

## Core Features

- **Live KPI monitoring**: throughput, latency, packet loss, and utilization update every second.
- **Adaptive automation**: the orchestrator can apply bandwidth and queue adjustments from live KPI pressure.
- **Manual / automated toggle**: manual sliders are disabled in automated mode, but scenario presets remain usable as seeds.
- **Scenario presets**: normal, congestion, optimized, and link-failure presets can be selected during a run.
- **Anomaly detection**: rule-based thresholds trigger alerts for loss, latency, utilization, and throughput drops.
- **Topology visualization**: animated network view with link saturation feedback.
- **Audit trail**: parameter changes are logged in the UI and exposed through the API.

## Network Topology

```text
UE1 --(10 Mbps)-- eNodeB --(100 Mbps)-- CoreRouter --(50 Mbps)-- Server
                                                          |
UE2 ---------------------------(20 Mbps)-------------------+
```

| Node | Type | Role |
| --- | --- | --- |
| `ue1` | UE | Access-side user equipment |
| `ue2` | UE | Direct-path user equipment |
| `enodeb` | eNodeB | Radio access node |
| `core_router` | CoreRouter | Backbone routing node |
| `server` | Server | Application/content endpoint |

Default link capacities:

| Link | Default BW |
| --- | --- |
| `ue1_enodeb` | 10 Mbps |
| `enodeb_core` | 100 Mbps |
| `core_server` | 50 Mbps |
| `ue2_server` | 20 Mbps |

## Setup Guide

### Requirements

- Python 3.8+
- Windows, macOS, or Linux
- Optional: Mininet on Linux for the Mininet-backed simulator

### 1. Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The optional ML packages are commented out in `requirements.txt`. Keep them disabled unless you want TensorFlow-based or forecasting extras.

### 3. Start the backend

```bash
python backend/api.py
```

The API runs on `http://localhost:5050`.

### 4. Open the dashboard

Open `frontend/index.html` directly in a browser. No separate frontend server is required.

### 5. Run tests

```bash
pytest
```

## API Reference

Base URL: `http://localhost:5050/api`

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service health check |
| `GET` | `/status` | Simulation and automation state |
| `GET` | `/control-mode` | Read current manual/automated mode |
| `POST` | `/control-mode` | Switch between manual config and automated allocation |
| `POST` | `/start` | Start simulation and optionally seed initial parameters |
| `POST` | `/stop` | Stop simulation |
| `GET` | `/kpis` | Latest KPI snapshot and active anomalies |
| `GET` | `/topology` | Current topology and parameters |
| `GET` | `/parameters` | Current network parameters |
| `POST` | `/parameters` | Update parameters manually or as a scenario seed |
| `GET` | `/aggregated` | Rolling aggregated KPIs |
| `GET` | `/log` | Parameter change log |

Example parameter update:

```bash
curl -X POST http://localhost:5050/api/parameters \
  -H "Content-Type: application/json" \
  -d '{
    "bandwidth": {"ue1_enodeb": 3.0},
    "queue_size": {"enodeb": 60},
    "scheduling_algorithm": {"enodeb": "FIFO", "core_router": "FIFO"},
    "traffic_load": {"ue1": 0.95, "ue2": 0.90}
  }'
```

## Scenarios

The dashboard includes four presets:

- **Normal**: balanced traffic and default capacities.
- **Congestion**: low bandwidth, small queues, high traffic load.
- **Optimized**: larger bandwidth, larger queues, PQ scheduling, lighter load.
- **Link Failure**: very low access-link bandwidth to model a degraded path.

When automation is enabled, presets act as scenario seeds. The orchestrator then tries to adjust the network dynamically based on the live KPI stream.

## KPI and Anomaly Model

The simulator uses a bottleneck-aware KPI model:

1. Demand is computed from bandwidth and per-UE load.
2. Each path is evaluated independently.
3. Utilization reflects the worst congested path.
4. Throughput is scaled by the active scheduling profile.
5. Latency includes congestion and queue-bloat effects.
6. Packet loss increases with overflow and very low-bandwidth failure conditions.

The `AnomalyAgent` currently focuses on rule-based detection for:

- packet loss above 5%,
- latency above 100 ms,
- utilization above 90%,
- throughput drops above 30% relative to recent history.

Optional ML-based detectors are available in the codebase, but the current backend is configured to rely on the rule-based path for stable demo behavior.

## Project Structure

```text
telecom-optimization/
├── backend/
│   └── api.py
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
├── src/
│   ├── agents/
│   │   ├── anomaly_agent.py
│   │   ├── optimization_agent.py
│   │   └── predictive_agent.py
│   ├── orchestration/
│   │   └── master_orchestrator.py
│   ├── simulation/
│   │   └── network_simulator.py
│   ├── interfaces.py
│   ├── main.py
│   └── models.py
├── tests/
├── requirements.txt
└── setup.py
```

## Technology Stack

| Layer | Technology |
| --- | --- |
| Backend | Python, Flask, Flask-CORS |
| Simulation | Custom network simulator, optional Mininet |
| Optimization | Rule-based orchestration with optional predictive hooks |
| Detection | scikit-learn, statsmodels, optional TensorFlow |
| Frontend | Vanilla HTML, CSS, and JavaScript |
| Testing | pytest |

## Notes

- The project is currently tuned for demo and portfolio use rather than production deployment.
- On Windows and macOS, the mock simulator is the intended runtime path.
- On Linux with Mininet installed, the simulator can use the Mininet-backed path.

## License

MIT

# AI-Assisted Network Optimization Dashboard

> A physics-accurate telecom network simulation with real-time anomaly detection and an interactive web dashboard for monitoring and controlling network KPIs.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Network Topology](#network-topology)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Anomaly Detection](#anomaly-detection)
- [KPI Model](#kpi-model)

---

## Overview

TeleOptAI simulates a 4G/5G-style telecom network and applies AI-driven anomaly detection to continuously monitor network health. Network operators can tune parameters (bandwidth, queue sizes, scheduling algorithms, per-UE traffic load) in real time and immediately observe the impact on KPIs — throughput, latency, packet loss, and utilization — through a live dashboard.

The simulation engine uses a **bottleneck-aware, physics-accurate KPI model** rather than random number generation. Every metric is derived from the actual bandwidth constraints, scheduling policy, and traffic demand configured for each link and node.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Browser (Frontend)                 │
│   index.html · script.js · style.css                 │
│   Live charts · Topology view · Parameter controls   │
└─────────────────────┬────────────────────────────────┘
                      │ REST (JSON)
                      │ http://localhost:5050/api/*
┌─────────────────────▼────────────────────────────────┐
│               Flask Backend  (backend/api.py)        │
│   /api/start  /api/stop  /api/kpis  /api/parameters  │
│   /api/topology  /api/aggregated  /api/log           │
└──────┬──────────────────────────┬────────────────────┘
       │                          │
┌──────▼──────────┐    ┌──────────▼──────────────────┐
│ MockNetwork     │    │  AnomalyAgent               │
│ Simulator       │    │  Rule-based + ML detection  │
│ (src/simulation)│    │  (src/agents)               │
└─────────────────┘    └─────────────────────────────┘
```

The backend runs a **background thread** that calls `collect_kpis()` every second, feeds the result to the `AnomalyAgent`, and stores the latest snapshot. The frontend polls `/api/kpis` every second and updates charts in real time.

---

## Network Topology

```
UE1 ──(10 Mbps)── eNodeB ──(100 Mbps)── CoreRouter ──(50 Mbps)── Server
                                                                      │
UE2 ────────────────────────────(20 Mbps)─────────────────────────────┘
```

| Node          | Type       | Role                                    |
| ------------- | ---------- | --------------------------------------- |
| `ue1`         | UE         | Mobile device on the access network     |
| `ue2`         | UE         | Mobile device with a direct server path |
| `enodeb`      | eNodeB     | LTE/5G base station                     |
| `core_router` | CoreRouter | Backbone routing node                   |
| `server`      | Server     | Application / content server            |

**Links** and their default bandwidths:

| Link          | Default BW |
| ------------- | ---------- |
| `ue1_enodeb`  | 10 Mbps    |
| `enodeb_core` | 100 Mbps   |
| `core_server` | 50 Mbps    |
| `ue2_server`  | 20 Mbps    |

---

## Features

- **Physics-accurate KPI engine** — throughput, latency, packet loss, and utilization derived from actual link capacities and per-UE traffic loads, not random numbers.
- **Bottleneck identification** — the narrowest link on the most-congested path is identified every tick and reported as the active bottleneck node.
- **Scheduling algorithm profiles** — FIFO, WFQ, PQ, and RR each carry distinct latency, loss, and throughput multipliers.
- **Real-time anomaly detection** — rule-based thresholds (packet loss > 5%, latency > 100 ms, utilization > 90%, throughput drop > 30%) with optional ML models (Isolation Forest, One-Class SVM, Autoencoder).
- **Live dashboard** — animated topology, KPI time-series charts, severity-coded alert feed, and a parameter control panel — all updating every second.
- **Dynamic parameter updates** — change any bandwidth, queue size, scheduling algorithm, or traffic load fraction while the simulation is running; the effect on KPIs is immediate.
- **Parameter change log** — every parameter mutation is timestamped and surfaced in the dashboard's audit panel.
- **Aggregated metrics** — `/api/aggregated` returns min/max/avg KPIs over a rolling 10-second window.
- **Mininet-ready** — the `MininetNetworkSimulator` class supports real Mininet topologies on Linux; `MockNetworkSimulator` is used as a portable fallback on Windows / macOS.

---

## Tech Stack

| Layer             | Technology                                                                     |
| ----------------- | ------------------------------------------------------------------------------ |
| Backend API       | Python 3.9+, Flask 2.3, Flask-CORS                                             |
| Simulation        | Custom Python simulator (Mininet-compatible)                                   |
| Anomaly Detection | scikit-learn (IsolationForest, OneClassSVM), TensorFlow (optional Autoencoder) |
| Numerics          | NumPy, SciPy, pandas, statsmodels                                              |
| Frontend          | Vanilla HTML5 / CSS3 / JavaScript (no build step)                              |
| Testing           | pytest                                                                         |

---

## Project Structure

```
telecom-optimization/
├── backend/
│   └── api.py                  # Flask REST API; simulation lifecycle endpoints
├── frontend/
│   ├── index.html              # Dashboard UI
│   ├── script.js               # Polling, chart rendering, event handlers
│   └── style.css               # Dark-mode dashboard styling
├── src/
│   ├── models.py               # Dataclasses: KPIMetrics, NetworkParameters, Anomaly, …
│   ├── interfaces.py           # Abstract base interfaces for simulator & agents
│   ├── main.py                 # CLI entry-point
│   ├── agents/
│   │   ├── anomaly_agent.py    # Rule-based + ML anomaly detection
│   │   ├── optimization_agent.py
│   │   └── predictive_agent.py
│   ├── simulation/
│   │   └── network_simulator.py  # MockNetworkSimulator & MininetNetworkSimulator
│   ├── orchestration/
│   └── telemetry/
├── config/                     # YAML configuration files
├── examples/                   # Usage examples
├── tests/                      # pytest test suite
├── requirements.txt
└── setup.py
```

---

## Getting Started

### Prerequisites

- Python 3.9 or newer
- (Optional) Mininet — Linux only; the mock simulator works on all platforms

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd telecom-optimization

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

> **Note:** TensorFlow and Prophet are commented out in `requirements.txt` by default. Uncomment them if you want the Autoencoder-based detection or load forecasting.

### 3. Start the backend

```bash
python backend/api.py
```

The API server starts at **http://localhost:5050**.

### 4. Open the dashboard

Open `frontend/index.html` directly in your browser (no web server required — CORS is enabled for all origins).

Click **▶ Start Simulation** to initialise the network topology and begin the live KPI feed.

---

## API Reference

All endpoints return JSON. Base URL: `http://localhost:5050`

| Method | Endpoint          | Description                                                |
| ------ | ----------------- | ---------------------------------------------------------- |
| `GET`  | `/api/health`     | Service health check                                       |
| `GET`  | `/api/status`     | Simulation running state, traffic active, KPI availability |
| `POST` | `/api/start`      | Initialise topology, start traffic & background loop       |
| `POST` | `/api/stop`       | Stop simulation and background loop                        |
| `GET`  | `/api/kpis`       | Latest KPI snapshot + any active anomalies                 |
| `GET`  | `/api/topology`   | Node list, link map, node types, current parameters        |
| `GET`  | `/api/parameters` | Current `NetworkParameters` object                         |
| `POST` | `/api/parameters` | Update bandwidth / queue size / scheduling / traffic load  |
| `GET`  | `/api/aggregated` | Rolling 10-second min/max/avg KPIs                         |
| `GET`  | `/api/log`        | Recent parameter change log (last 50 entries)              |

### Example: update parameters

```bash
curl -X POST http://localhost:5050/api/parameters \
  -H "Content-Type: application/json" \
  -d '{
    "bandwidth": { "ue1_enodeb": 2.0 },
    "scheduling_algorithm": { "enodeb": "PQ", "core_router": "WFQ" },
    "traffic_load": { "ue1": 0.90, "ue2": 0.40 }
  }'
```

---

## Configuration

### Default network parameters

| Parameter               | Default         |
| ----------------------- | --------------- |
| `ue1_enodeb` bandwidth  | 10 Mbps         |
| `enodeb_core` bandwidth | 100 Mbps        |
| `core_server` bandwidth | 50 Mbps         |
| `ue2_server` bandwidth  | 20 Mbps         |
| Queue size (all nodes)  | 100–200 packets |
| eNodeB scheduling       | WFQ             |
| CoreRouter scheduling   | WFQ             |

### Bandwidth constraints (validated server-side)

- Minimum: **0.1 Mbps** per link
- Maximum: **1000 Mbps** per link
- Queue size range: **10 – 10000 packets**

### Supported scheduling algorithms

| Algorithm | Effect vs WFQ baseline                         |
| --------- | ---------------------------------------------- |
| `FIFO`    | +20% latency, +30% packet loss, −8% throughput |
| `WFQ`     | Baseline (1×)                                  |
| `PQ`      | −15% latency, −20% packet loss, +5% throughput |
| `RR`      | +5% latency, −10% packet loss, −2% throughput  |

---

## Anomaly Detection

The `AnomalyAgent` uses two detection layers:

### Rule-based (always active)

| Anomaly Type        | Threshold                                    |
| ------------------- | -------------------------------------------- |
| `PACKET_LOSS`       | > 5%                                         |
| `LATENCY_SPIKE`     | > 100 ms                                     |
| `THROUGHPUT_DROP`   | > 30% relative drop vs 5-sample rolling mean |
| `UTILIZATION_SPIKE` | > 90%                                        |

### ML-based (activates after 50 samples)

- **Isolation Forest** — unsupervised outlier detection on the 4-feature KPI vector
- **One-Class SVM** — boundary-based novelty detection
- **Autoencoder** (optional, requires TensorFlow) — reconstruction-error-based detection

All anomalies are classified with a severity level:

| Severity   | Colour |
| ---------- | ------ |
| `LOW`      | Blue   |
| `MEDIUM`   | Yellow |
| `HIGH`     | Orange |
| `CRITICAL` | Red    |

The affected node reported with each anomaly reflects the actual bottleneck node identified by the KPI engine.

---

## KPI Model

The `MockNetworkSimulator` derives all metrics from first principles each tick:

1. **Demand** = link bandwidth × per-UE load fraction
2. **Bottleneck** = `min(demand, min_link_on_path)` for each UE path independently
3. **Utilization** = worst-case path utilization (0–100%)
4. **Throughput** = sum of actual goodput across both paths × scheduler throughput multiplier
5. **Latency** = base propagation + congestion term (quadratic above 75% utilization) + buffer-bloat term (queue depth × utilization) × scheduler latency multiplier
6. **Packet loss** = overflow term (linear above 85% utilization) × scheduler loss multiplier × queue-depth softener
7. **Bottleneck node** = the node on the most-congested path that has the smallest bandwidth

Small Gaussian noise (±3–5%) is added to each metric to simulate real-world measurement jitter.

---

## License

Academic / research use. RVCE — Department of ECE.

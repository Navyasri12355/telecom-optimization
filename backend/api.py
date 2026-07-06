"""
TeleOptAI — Flask REST API Backend
Bridges the Python simulation engine with the web frontend dashboard.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import threading
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS

from src.simulation.network_simulator import create_network_simulator
from src.agents.anomaly_agent import AnomalyAgent
from src.orchestration import MasterOrchestrator
from src.models import NetworkParameters, KPIMetrics

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow all origins so the frontend file:// or localhost can connect

# ── Simulation state ──────────────────────────────────────────────────────────
simulator    = None
# Use rule-based detection only -- ML models (IsolationForest) always flag
# contamination_rate% of points as anomalies by design, causing false positives
# in normal operation. Rule-based thresholds are reliable and meaningful.
anomaly_agent = AnomalyAgent(detection_algorithms=[])
orchestrator = None
automation_enabled = False
sim_lock     = threading.Lock()
sim_thread   = None
sim_running  = False
latest_kpi   = None        # most recent KPIMetrics object
latest_anomalies = []      # most recent anomaly list
param_change_log = []      # list of change-log strings


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _apply_parameter_payload(payload: dict, scenario_seed: bool = False):
    """Apply a parameter payload to the active simulator."""
    global param_change_log

    if simulator is None or simulator.current_parameters is None:
        return {"error": "Simulator not initialized"}, 400

    if sim_running and automation_enabled and not scenario_seed:
        return {
            "error": "Manual parameter updates are disabled while automated resource allocation is enabled"
        }, 409

    current = simulator.current_parameters
    changes = []

    new_bw = {**current.bandwidth}
    new_qs = {**current.queue_size}
    new_sched = {**current.scheduling_algorithm}

    for link, val in payload.get("bandwidth", {}).items():
        val = float(val)
        if new_bw.get(link) != val:
            changes.append(f"BW {link}: {new_bw.get(link)} → {val} Mbps")
            new_bw[link] = val

    for node, val in payload.get("queue_size", {}).items():
        val = int(val)
        if new_qs.get(node) != val:
            changes.append(f"Queue {node}: {new_qs.get(node)} → {val} pkts")
            new_qs[node] = val

    for node, algo in payload.get("scheduling_algorithm", {}).items():
        if new_sched.get(node) != algo:
            changes.append(f"Scheduling {node}: {new_sched.get(node)} → {algo}")
            new_sched[node] = algo

    try:
        new_params = NetworkParameters(
            bandwidth=new_bw,
            queue_size=new_qs,
            scheduling_algorithm=new_sched,
            update_timestamp=datetime.now()
        )
        simulator.update_parameters(new_params)
    except (ValueError, RuntimeError) as e:
        logger.error(f"Parameter update failed: {e}")
        return {"error": str(e)}, 400

    traffic_load = payload.get("traffic_load", {})
    if traffic_load and hasattr(simulator, 'set_traffic_load'):
        simulator.set_traffic_load(traffic_load)
        tl_changes = [f"Traffic load {k}: {round(v*100)}%" for k, v in traffic_load.items()]
        changes.extend(tl_changes)

    ts = datetime.now().strftime("%H:%M:%S")
    for c in changes:
        param_change_log.insert(0, {"time": ts, "change": c})

    if len(param_change_log) > 50:
        param_change_log = param_change_log[:50]

    logger.info(f"Parameters updated: {changes}")
    return {"status": "updated", "changes": changes}, 200


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build KPI dict for JSON response
# ─────────────────────────────────────────────────────────────────────────────
def kpi_to_dict(kpi: KPIMetrics) -> dict:
    return {
        "throughput":  round(kpi.throughput,  2),
        "latency":     round(kpi.latency,     2),
        "packet_loss": round(kpi.packet_loss, 3),
        "utilization": round(kpi.utilization, 2),
        "node_id":     kpi.node_id,
        "timestamp":   kpi.timestamp.isoformat(),
    }


def anomaly_to_dict(a) -> dict:
    return {
        "type":       a.anomaly_type.value,
        "severity":   a.severity.value,
        "nodes":      a.affected_nodes,
        "confidence": round(a.confidence_score, 3),
        "time":       a.detection_time.isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Background simulation loop
# ─────────────────────────────────────────────────────────────────────────────
def _simulation_loop():
    global latest_kpi, latest_anomalies, sim_running, orchestrator
    logger.info("Simulation loop started")

    while sim_running:
        try:
            with sim_lock:
                if simulator and simulator.is_simulation_running():
                    kpi = simulator.collect_kpis()
                    latest_kpi = kpi

                    if orchestrator is not None:
                        cycle_result = orchestrator.process_kpi(kpi)
                        latest_anomalies = cycle_result.anomalies
                    else:
                        # Fallback path if orchestration is unavailable
                        anomalies = anomaly_agent.detect_anomalies(kpi)
                        if anomalies:
                            anomaly_agent.trigger_alerts(anomalies)
                            latest_anomalies = anomalies
                        else:
                            latest_anomalies = []

        except Exception as e:
            logger.error(f"Error in simulation loop: {e}")

        import time
        time.sleep(1)   # collect every second

    logger.info("Simulation loop stopped")


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/status", methods=["GET"])
def get_status():
    """Return current simulation status."""
    global simulator, sim_running, automation_enabled
    with sim_lock:
        is_init    = simulator is not None and simulator.is_simulation_running()
        is_traffic = simulator.is_traffic_running() if is_init else False

    return jsonify({
        "running":         sim_running,
        "initialized":     is_init,
        "traffic_active":  is_traffic,
        "automation_enabled": automation_enabled,
        "has_kpi":         latest_kpi is not None,
    })


@app.route("/api/control-mode", methods=["GET"])
def get_control_mode():
    """Return the current control mode."""
    return jsonify({"automation_enabled": automation_enabled})


@app.route("/api/control-mode", methods=["POST"])
def set_control_mode():
    """Switch between manual configuration and automated resource allocation."""
    global automation_enabled, orchestrator

    data = request.get_json(silent=True) or {}
    automation_enabled = bool(data.get("automation_enabled", False))

    with sim_lock:
        if orchestrator is not None:
            orchestrator.automation_enabled = automation_enabled

    logger.info("Control mode updated: %s", "automated" if automation_enabled else "manual")
    return jsonify({"status": "updated", "automation_enabled": automation_enabled})


@app.route("/api/start", methods=["POST"])
def start_simulation():
    """Initialize topology, start traffic, start background loop."""
    global simulator, orchestrator, automation_enabled, sim_running, sim_thread, latest_kpi, latest_anomalies, param_change_log

    data = request.get_json(silent=True) or {}
    automation_enabled = bool(data.get("automation_enabled", automation_enabled))
    initial_parameters = data.get("initial_parameters") or {}

    with sim_lock:
        if sim_running:
            return jsonify({"error": "Simulation already running"}), 400

        try:
            simulator = create_network_simulator(use_mininet=False)  # use MockSimulator
            simulator.initialize_topology()
            if initial_parameters:
                apply_result, status_code = _apply_parameter_payload(initial_parameters, scenario_seed=True)
                if status_code >= 400:
                    return jsonify(apply_result), status_code
            simulator.start_traffic_generation()
            orchestrator = MasterOrchestrator(
                simulator=simulator,
                anomaly_agent=anomaly_agent,
                automation_enabled=automation_enabled,
                prediction_horizon=10,
                control_interval=1.0,
            )
            orchestrator.initialize_agents()
            logger.info("Topology initialized and traffic started")
        except Exception as e:
            logger.error(f"Failed to start simulation: {e}")
            return jsonify({"error": str(e)}), 500

    # Reset state
    latest_kpi        = None
    latest_anomalies  = []
    param_change_log  = []
    anomaly_agent.reset_detection_history()

    sim_running = True
    sim_thread  = threading.Thread(target=_simulation_loop, daemon=True)
    sim_thread.start()

    return jsonify({"status": "started", "message": "Simulation started successfully"})


@app.route("/api/stop", methods=["POST"])
def stop_simulation():
    """Stop the simulation and background loop."""
    global simulator, orchestrator, sim_running

    sim_running = False

    if orchestrator is not None:
        orchestrator.shutdown_system()

    with sim_lock:
        if simulator:
            try:
                simulator.stop_simulation()
            except Exception as e:
                logger.warning(f"Error stopping simulator: {e}")
            simulator = None
            orchestrator = None

    return jsonify({"status": "stopped"})


@app.route("/api/kpis", methods=["GET"])
def get_kpis():
    """Return latest KPI metrics."""
    if not sim_running:
        return jsonify({"error": "Simulation not running"}), 404
    if latest_kpi is None:
        return jsonify({"error": "No KPI data yet"}), 204

    return jsonify({
        "kpi": kpi_to_dict(latest_kpi),
        "anomalies": [anomaly_to_dict(a) for a in latest_anomalies],
    })


@app.route("/api/topology", methods=["GET"])
def get_topology():
    """Return network topology."""
    with sim_lock:
        if simulator is None or simulator.topology is None:
            return jsonify({"error": "Simulator not initialized"}), 404

        topo = simulator.topology
        params = simulator.current_parameters

    return jsonify({
        "nodes": topo.nodes,
        "links": {k: list(v) for k, v in topo.links.items()},
        "node_types": topo.node_types,
        "parameters": params.to_dict() if params else {},
    })


@app.route("/api/parameters", methods=["GET"])
def get_parameters():
    """Return current network parameters."""
    with sim_lock:
        if simulator is None or simulator.current_parameters is None:
            return jsonify({"error": "No parameters available"}), 404
        params = simulator.current_parameters.to_dict()

    return jsonify({"parameters": params})


@app.route("/api/parameters", methods=["POST"])
def update_parameters():
    """
    Update network parameters.
    Body: { "bandwidth": {...}, "queue_size": {...},
            "scheduling_algorithm": {...}, "traffic_load": {"ue1": 0.85, "ue2": 0.40} }
    All keys are optional — only provided keys are updated.
    """
    data = request.get_json(force=True) or {}
    scenario_seed = bool(data.pop("scenario_seed", False))

    with sim_lock:
        result, status_code = _apply_parameter_payload(data, scenario_seed=scenario_seed)

    return jsonify(result), status_code


@app.route("/api/log", methods=["GET"])
def get_log():
    """Return recent parameter change log."""
    return jsonify({"log": param_change_log})


@app.route("/api/aggregated", methods=["GET"])
def get_aggregated():
    """Return aggregated KPI metrics over the last 10 seconds."""
    with sim_lock:
        if simulator is None:
            return jsonify({"error": "Not initialized"}), 404
        agg = simulator.get_aggregated_metrics(window_seconds=10)

    return jsonify({"aggregated": agg})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "TeleOptAI Backend"})


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting TeleOptAI backend on http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)

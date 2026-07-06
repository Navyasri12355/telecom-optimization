"""Tests for the master orchestrator automation loop."""

from datetime import datetime

from src.models import KPIMetrics, ActionType, NetworkParameters
from src.orchestration import MasterOrchestrator


class DummySimulator:
    def __init__(self):
        self._running = True
        self.current_parameters = NetworkParameters(
            bandwidth={"ue1_enodeb": 10.0, "enodeb_core": 100.0, "core_server": 50.0, "ue2_server": 20.0},
            queue_size={"ue1": 100, "ue2": 100, "enodeb": 200, "core_router": 150, "server": 100},
            scheduling_algorithm={"ue1": "FIFO", "ue2": "FIFO", "enodeb": "WFQ", "core_router": "WFQ", "server": "FIFO"},
            update_timestamp=datetime.now(),
        )
        self.updated_parameters = None

    def is_simulation_running(self):
        return self._running

    def collect_kpis(self):
        return KPIMetrics(
            timestamp=datetime.now(),
            throughput=42.0,
            latency=180.0,
            packet_loss=6.5,
            utilization=92.0,
            node_id="core_router",
        )

    def update_parameters(self, params):
        self.updated_parameters = params
        self.current_parameters = params


class DummyAnomalyAgent:
    def __init__(self):
        self.triggered = []

    def detect_anomalies(self, kpi):
        return []

    def trigger_alerts(self, anomalies):
        self.triggered.extend(anomalies)

    def reset_detection_history(self):
        self.triggered = []


def test_orchestrator_applies_automatic_capacity_change():
    simulator = DummySimulator()
    anomaly_agent = DummyAnomalyAgent()
    original_bandwidth = simulator.current_parameters.bandwidth["ue1_enodeb"]
    orchestrator = MasterOrchestrator(
        simulator=simulator,
        anomaly_agent=anomaly_agent,
        predictive_agent=None,
        automation_enabled=True,
    )

    result = orchestrator.process_kpi(simulator.collect_kpis())

    assert result.decision.action_type == ActionType.INCREASE_CAPACITY
    assert result.automation_applied is True
    assert simulator.updated_parameters is not None
    assert simulator.updated_parameters.bandwidth["ue1_enodeb"] > original_bandwidth
    assert orchestrator.get_cycle_log()

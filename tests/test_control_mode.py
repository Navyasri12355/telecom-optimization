"""Tests for control-mode switching and automated resource allocation guardrails."""

from datetime import datetime

from backend import api
from src.models import NetworkParameters


class DummySimulator:
    def __init__(self):
        self.current_parameters = NetworkParameters(
            bandwidth={"ue1_enodeb": 10.0, "enodeb_core": 100.0, "core_server": 50.0, "ue2_server": 20.0},
            queue_size={"ue1": 100, "ue2": 100, "enodeb": 200, "core_router": 150, "server": 100},
            scheduling_algorithm={"ue1": "FIFO", "ue2": "FIFO", "enodeb": "WFQ", "core_router": "WFQ", "server": "FIFO"},
            update_timestamp=datetime.now(),
        )
        self.topology = object()
        self._running = True
        self.traffic_load = {}

    def is_simulation_running(self):
        return self._running

    def is_traffic_running(self):
        return self._running

    def start_traffic_generation(self):
        self._running = True

    def initialize_topology(self):
        return self.topology

    def stop_simulation(self):
        self._running = False

    def update_parameters(self, params):
        self.current_parameters = params

    def set_traffic_load(self, traffic_load):
        self.traffic_load = dict(traffic_load)


def test_control_mode_toggle_updates_backend_state():
    client = api.app.test_client()

    response = client.post("/api/control-mode", json={"automation_enabled": True})

    assert response.status_code == 200
    assert response.get_json()["automation_enabled"] is True
    assert api.automation_enabled is True


def test_manual_parameter_updates_are_blocked_in_automated_mode(monkeypatch):
    client = api.app.test_client()
    api.simulator = DummySimulator()
    api.sim_running = True
    api.automation_enabled = True

    response = client.post(
        "/api/parameters",
        json={"bandwidth": {"ue1_enodeb": 25.0}},
    )

    assert response.status_code == 409
    assert "disabled" in response.get_json()["error"].lower()

    api.simulator = None
    api.sim_running = False
    api.automation_enabled = False


def test_scenario_seed_updates_are_allowed_in_automated_mode(monkeypatch):
    client = api.app.test_client()
    api.simulator = DummySimulator()
    api.sim_running = True
    api.automation_enabled = True

    response = client.post(
        "/api/parameters",
        json={
            "scenario_seed": True,
            "bandwidth": {"ue1_enodeb": 3.0},
            "queue_size": {"enodeb": 60},
            "scheduling_algorithm": {"enodeb": "FIFO"},
            "traffic_load": {"ue1": 0.95},
        },
    )

    assert response.status_code == 200
    assert response.get_json()["status"] == "updated"
    assert api.simulator.current_parameters.bandwidth["ue1_enodeb"] == 3.0
    assert api.simulator.traffic_load == {"ue1": 0.95}

    api.simulator = None
    api.sim_running = False
    api.automation_enabled = False
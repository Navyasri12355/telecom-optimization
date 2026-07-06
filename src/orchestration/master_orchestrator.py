"""
Master orchestrator for AI-driven telecom network optimization.

This module coordinates KPI ingestion, optional load forecasting, optimization
decisions, anomaly detection, and automatic network parameter updates.
"""

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from src.interfaces import MasterOrchestratorInterface
from src.models import KPIMetrics, LoadForecast, NetworkParameters, OptimizationDecision, ActionType
from src.agents.optimization_agent import OptimizationAgent

try:
    from src.agents.predictive_agent import PredictiveAgent
except Exception:
    PredictiveAgent = None


@dataclass
class AutomationCycleResult:
    """Summary of one automation cycle."""

    timestamp: datetime
    kpi: KPIMetrics
    forecast: LoadForecast
    decision: OptimizationDecision
    anomalies: List[Any]
    applied_parameters: Optional[NetworkParameters]
    automation_applied: bool


class MasterOrchestrator(MasterOrchestratorInterface):
    """Coordinate prediction, optimization, anomaly handling, and application."""

    def __init__(
        self,
        simulator: Any,
        anomaly_agent: Optional[Any] = None,
        predictive_agent: Optional[Any] = None,
        optimization_agent: Optional[OptimizationAgent] = None,
        automation_enabled: bool = True,
        prediction_horizon: int = 10,
        control_interval: float = 1.0,
    ):
        self.logger = logging.getLogger(__name__)
        self.simulator = simulator
        self.anomaly_agent = anomaly_agent
        self.optimization_agent = optimization_agent or OptimizationAgent()
        self.automation_enabled = automation_enabled
        self.prediction_horizon = prediction_horizon
        self.control_interval = control_interval
        self._stop_event = threading.Event()
        self._history: List[KPIMetrics] = []
        self._cycle_log: List[AutomationCycleResult] = []

        self.predictive_agent = predictive_agent
        if self.predictive_agent is None and PredictiveAgent is not None:
            try:
                self.predictive_agent = PredictiveAgent(window_size_seconds=15, prediction_horizon=prediction_horizon)
            except Exception as exc:
                self.logger.warning(f"PredictiveAgent unavailable, using heuristic forecasting: {exc}")
                self.predictive_agent = None

    def initialize_agents(self) -> None:
        """Initialize agent state for a fresh simulation run."""
        self._stop_event.clear()
        self._history.clear()
        self._cycle_log.clear()
        if self.anomaly_agent and hasattr(self.anomaly_agent, "reset_detection_history"):
            self.anomaly_agent.reset_detection_history()

    def coordinate_agents(self) -> None:
        """No-op coordination hook retained for interface compatibility."""
        return

    def execute_control_loop(self) -> None:
        """Continuously process KPIs until stopped."""
        while not self._stop_event.is_set():
            if not self.simulator or not getattr(self.simulator, "is_simulation_running", lambda: False)():
                break

            kpi = self._collect_latest_kpi()
            if kpi is not None:
                self.process_kpi(kpi)

            self._stop_event.wait(self.control_interval)

    def stop(self) -> None:
        """Stop the control loop."""
        self._stop_event.set()

    def process_kpi(self, kpi: KPIMetrics) -> AutomationCycleResult:
        """Process a KPI snapshot and apply automation if needed."""
        self._history.append(kpi)

        anomalies = self._detect_anomalies(kpi)
        forecast = self._generate_forecast()
        decision = self.optimization_agent.evaluate_predictions(forecast)

        if self.automation_enabled:
            live_pressure = (
                kpi.utilization >= 90.0 or
                kpi.packet_loss >= 5.0 or
                kpi.latency >= 100.0
            )

            if live_pressure:
                pressure_forecast = self._build_pressure_forecast(kpi)
                pressure_decision = self.optimization_agent.evaluate_predictions(pressure_forecast)
                if pressure_decision.action_type != ActionType.NO_ACTION:
                    decision = pressure_decision

        execution_context: Dict[str, Any] = {
            "current_kpi": kpi.to_dict(),
            "anomalies_detected": [a.to_dict() for a in anomalies] if anomalies else [],
            "automation_enabled": self.automation_enabled,
            "history_size": len(self._history),
        }

        applied_parameters = None
        automation_applied = False

        if self.automation_enabled and decision.action_type != ActionType.NO_ACTION:
            applied_parameters = self.optimization_agent.adjust_capacity(
                decision,
                current_parameters=getattr(self.simulator, "current_parameters", None),
            )
            self.simulator.update_parameters(applied_parameters)
            self.optimization_agent.log_actions(
                decision,
                "Automatic control loop applied capacity adjustment based on live KPI forecast.",
                network_params=applied_parameters,
                execution_context=execution_context,
            )
            automation_applied = True

        cycle_result = AutomationCycleResult(
            timestamp=datetime.now(),
            kpi=kpi,
            forecast=forecast,
            decision=decision,
            anomalies=anomalies,
            applied_parameters=applied_parameters,
            automation_applied=automation_applied,
        )
        self._cycle_log.append(cycle_result)

        return cycle_result

    def _build_pressure_forecast(self, kpi: KPIMetrics) -> LoadForecast:
        """Build a pressure forecast from the live KPI when the network is already stressed."""
        pressure_load = max(kpi.utilization, 0.0)
        if kpi.packet_loss >= 5.0 or kpi.latency >= 100.0:
            pressure_load = max(pressure_load, 96.0)
        elif kpi.utilization >= 90.0:
            pressure_load = max(pressure_load, 92.0)

        return LoadForecast(
            predicted_values=[pressure_load for _ in range(max(self.prediction_horizon, 1))],
            confidence_interval=(pressure_load, pressure_load),
            prediction_horizon=self.prediction_horizon,
            model_accuracy=95.0,
            timestamp=datetime.now(),
        )

    def resolve_conflicts(self, decisions: List[OptimizationDecision]) -> OptimizationDecision:
        """Resolve conflicts through the optimization agent."""
        return self.optimization_agent.resolve_conflicts(decisions)

    def maintain_system_logs(self, event: str, context: Dict[str, Any]) -> None:
        """Write structured orchestration logs."""
        self.logger.info(f"ORCHESTRATION_EVENT [{event}]: {context}")

    def shutdown_system(self) -> None:
        """Stop all orchestration activity."""
        self.stop()

    def get_cycle_log(self) -> List[AutomationCycleResult]:
        """Return a copy of the automation cycle history."""
        return list(self._cycle_log)

    def _collect_latest_kpi(self) -> Optional[KPIMetrics]:
        try:
            return self.simulator.collect_kpis()
        except Exception as exc:
            self.logger.error(f"Failed to collect KPI for automation cycle: {exc}")
            return None

    def _detect_anomalies(self, kpi: KPIMetrics) -> List[Any]:
        if not self.anomaly_agent:
            return []

        try:
            anomalies = self.anomaly_agent.detect_anomalies(kpi)
            if anomalies and hasattr(self.anomaly_agent, "trigger_alerts"):
                self.anomaly_agent.trigger_alerts(anomalies)
            return anomalies
        except Exception as exc:
            self.logger.error(f"Anomaly detection failed during orchestration: {exc}")
            return []

    def _generate_forecast(self) -> LoadForecast:
        if self.predictive_agent is not None:
            try:
                self.predictive_agent.analyze_historical_data(self._history)
                return self.predictive_agent.predict_load(horizon=self.prediction_horizon)
            except Exception as exc:
                self.logger.warning(f"PredictiveAgent forecast failed, using heuristic fallback: {exc}")

        return self._heuristic_forecast()

    def _heuristic_forecast(self) -> LoadForecast:
        recent_history = self._history[-10:]
        utilization_values = [sample.utilization for sample in recent_history]

        if not utilization_values:
            utilization_values = [50.0]

        if len(utilization_values) >= 2:
            slope = (utilization_values[-1] - utilization_values[0]) / max(len(utilization_values) - 1, 1)
        else:
            slope = 0.0

        base_value = utilization_values[-1]
        predicted_values = []
        for step in range(1, self.prediction_horizon + 1):
            prediction = base_value + (slope * step)
            predicted_values.append(float(np.clip(prediction, 0.0, 100.0)))

        average_value = float(np.mean(utilization_values))
        spread = float(max(5.0, np.std(utilization_values) if len(utilization_values) > 1 else 10.0))
        confidence_interval = (
            float(np.clip(average_value - spread, 0.0, 100.0)),
            float(np.clip(average_value + spread, 0.0, 100.0)),
        )

        model_accuracy = float(np.clip(100.0 - (abs(slope) * 5.0), 5.0, 95.0))

        return LoadForecast(
            predicted_values=predicted_values,
            confidence_interval=confidence_interval,
            prediction_horizon=self.prediction_horizon,
            model_accuracy=model_accuracy,
            timestamp=datetime.now(),
        )
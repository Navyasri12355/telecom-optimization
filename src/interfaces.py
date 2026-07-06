"""
Base interfaces for the AI-driven telecom network optimization system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from src.models import (
    KPIMetrics, LoadForecast, OptimizationDecision, Anomaly, 
    NetworkParameters, NetworkTopology, ExportFormat, SeverityLevel
)


class NetworkSimulatorInterface(ABC):
    """Interface for network simulation components."""
    
    @abstractmethod
    def initialize_topology(self) -> NetworkTopology:
        """Initialize the network topology."""
        pass
    
    @abstractmethod
    def start_traffic_generation(self) -> None:
        """Start traffic generation between network nodes."""
        pass
    
    @abstractmethod
    def collect_kpis(self) -> KPIMetrics:
        """Collect current KPI metrics from the simulation."""
        pass
    
    @abstractmethod
    def update_parameters(self, params: NetworkParameters) -> None:
        """Update network parameters dynamically."""
        pass
    
    @abstractmethod
    def stop_simulation(self) -> None:
        """Stop the network simulation."""
        pass
    
    def get_aggregated_metrics(self, window_seconds: int = 10) -> Optional[Dict[str, float]]:
        """
        Get aggregated KPI metrics over a specified time window.
        
        Args:
            window_seconds: Time window in seconds for aggregation
            
        Returns:
            Dict containing aggregated metrics or None if insufficient data
        """
        pass
    
    def validate_current_parameters(self) -> bool:
        """
        Validate current network parameters for consistency.
        
        Returns:
            bool: True if parameters are valid, False otherwise
        """
        pass
    
    def get_parameter_history(self) -> List[Dict[str, Any]]:
        """
        Get history of parameter changes.
        
        Returns:
            List of parameter change records
        """
        pass


class KPIPipelineInterface(ABC):
    """Interface for KPI collection and export pipeline."""
    
    @abstractmethod
    def collect_metrics(self, simulator: NetworkSimulatorInterface) -> KPIMetrics:
        """Collect metrics from the network simulator."""
        pass
    
    @abstractmethod
    def export_to_storage(self, metrics: KPIMetrics, format: ExportFormat) -> None:
        """Export metrics to specified storage format."""
        pass
    
    @abstractmethod
    def stream_to_dashboard(self, metrics: KPIMetrics) -> None:
        """Stream metrics to dashboard systems."""
        pass
    
    @abstractmethod
    def apply_retention_policy(self) -> None:
        """Apply data retention policies to prevent storage overflow."""
        pass


class PredictiveAgentInterface(ABC):
    """Interface for predictive analytics agent."""
    
    @abstractmethod
    def analyze_historical_data(self, kpis: List[KPIMetrics]) -> None:
        """Analyze historical KPI data for prediction model training."""
        pass
    
    @abstractmethod
    def predict_load(self, horizon: int = 10) -> LoadForecast:
        """Generate load forecast for specified time horizon."""
        pass
    
    @abstractmethod
    def calculate_accuracy(self, predictions: LoadForecast, actual: List[KPIMetrics]) -> float:
        """Calculate prediction accuracy using MAPE."""
        pass
    
    @abstractmethod
    def update_model(self, feedback_data: List[KPIMetrics]) -> None:
        """Update prediction model based on feedback."""
        pass


class OptimizationAgentInterface(ABC):
    """Interface for network optimization agent."""
    
    @abstractmethod
    def evaluate_predictions(self, forecast: LoadForecast) -> OptimizationDecision:
        """Evaluate predictions and make optimization decisions."""
        pass
    
    @abstractmethod
    def adjust_capacity(self, decision: OptimizationDecision, current_parameters: Optional[NetworkParameters] = None) -> NetworkParameters:
        """Generate network parameter adjustments based on decision."""
        pass
    
    @abstractmethod
    def log_actions(self, decision: OptimizationDecision, rationale: str, 
                   network_params: Optional['NetworkParameters'] = None,
                   execution_context: Optional[Dict[str, Any]] = None) -> None:
        """Log optimization actions with comprehensive details and rationale."""
        pass
    
    @abstractmethod
    def resolve_conflicts(self, decisions: List[OptimizationDecision]) -> OptimizationDecision:
        """Resolve conflicts between multiple optimization decisions."""
        pass


class AnomalyAgentInterface(ABC):
    """Interface for anomaly detection agent."""
    
    @abstractmethod
    def detect_anomalies(self, kpis: KPIMetrics) -> List[Anomaly]:
        """Detect anomalies in current KPI metrics."""
        pass
    
    @abstractmethod
    def classify_severity(self, anomaly: Anomaly) -> 'SeverityLevel':
        """Classify the severity of detected anomaly."""
        pass
    
    @abstractmethod
    def trigger_alerts(self, anomalies: List[Anomaly]) -> None:
        """Trigger alerts for detected anomalies."""
        pass


class MasterOrchestratorInterface(ABC):
    """Interface for master orchestrator component."""
    
    @abstractmethod
    def initialize_agents(self) -> None:
        """Initialize all system agents."""
        pass
    
    @abstractmethod
    def coordinate_agents(self) -> None:
        """Coordinate communication between agents."""
        pass
    
    @abstractmethod
    def execute_control_loop(self) -> None:
        """Execute the sense-predict-decide-act-log control loop."""
        pass
    
    @abstractmethod
    def resolve_conflicts(self, decisions: List[OptimizationDecision]) -> OptimizationDecision:
        """Resolve conflicts between agent decisions."""
        pass
    
    @abstractmethod
    def maintain_system_logs(self, event: str, context: Dict[str, Any]) -> None:
        """Maintain comprehensive system logs."""
        pass
    
    @abstractmethod
    def shutdown_system(self) -> None:
        """Gracefully shutdown the entire system."""
        pass


class MessageBusInterface(ABC):
    """Interface for inter-agent communication."""
    
    @abstractmethod
    def send_message(self, recipient: str, message: Dict[str, Any]) -> None:
        """Send message to specified recipient."""
        pass
    
    @abstractmethod
    def receive_message(self, sender: str) -> Optional[Dict[str, Any]]:
        """Receive message from specified sender."""
        pass
    
    @abstractmethod
    def broadcast_message(self, message: Dict[str, Any]) -> None:
        """Broadcast message to all connected agents."""
        pass
    
    @abstractmethod
    def subscribe_to_topic(self, topic: str, callback: callable) -> None:
        """Subscribe to message topic with callback."""
        pass


class ClosedLoopControllerInterface(ABC):
    """Interface for closed-loop feedback control."""
    
    @abstractmethod
    def integrate_feedback(self, action_results: List[KPIMetrics]) -> None:
        """Integrate action results into system feedback."""
        pass
    
    @abstractmethod
    def adapt_system_behavior(self, performance_metrics: Dict[str, float]) -> None:
        """Adapt system behavior based on performance feedback."""
        pass
    
    @abstractmethod
    def learn_from_decisions(self, decisions: List[OptimizationDecision], outcomes: List[KPIMetrics]) -> None:
        """Learn from decision outcomes to improve future decisions."""
        pass
    
    @abstractmethod
    def maintain_performance_targets(self) -> bool:
        """Check if system is maintaining performance targets."""
        pass
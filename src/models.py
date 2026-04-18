"""
Core data models for the AI-driven telecom network optimization system.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Tuple, Any
import json


class ActionType(Enum):
    """Types of optimization actions."""
    INCREASE_CAPACITY = "increase_capacity"
    DECREASE_CAPACITY = "decrease_capacity"
    NO_ACTION = "no_action"


class SeverityLevel(Enum):
    """Severity levels for anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyType(Enum):
    """Types of network anomalies."""
    PACKET_LOSS = "packet_loss"
    LATENCY_SPIKE = "latency_spike"
    THROUGHPUT_DROP = "throughput_drop"
    NODE_FAILURE = "node_failure"
    UTILIZATION_SPIKE = "utilization_spike"
    GENERAL_ANOMALY = "general_anomaly"


class ExportFormat(Enum):
    """Export formats for telemetry data."""
    CSV = "csv"
    SQLITE = "sqlite"
    PROMETHEUS = "prometheus"


@dataclass
class KPIMetrics:
    """Network performance metrics collected from simulation."""
    timestamp: datetime
    throughput: float  # Mbps
    latency: float     # milliseconds
    packet_loss: float # percentage
    utilization: float # percentage
    node_id: str

    def __post_init__(self):
        """Validate KPI metrics after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate KPI metrics values."""
        if self.throughput < 0:
            raise ValueError("Throughput must be non-negative")
        if self.latency < 0:
            raise ValueError("Latency must be non-negative")
        if not (0 <= self.packet_loss <= 100):
            raise ValueError("Packet loss must be between 0 and 100 percent")
        if not (0 <= self.utilization <= 100):
            raise ValueError("Utilization must be between 0 and 100 percent")
        if not self.node_id or not self.node_id.strip():
            raise ValueError("Node ID cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'throughput': self.throughput,
            'latency': self.latency,
            'packet_loss': self.packet_loss,
            'utilization': self.utilization,
            'node_id': self.node_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KPIMetrics':
        """Create from dictionary for deserialization."""
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            throughput=data['throughput'],
            latency=data['latency'],
            packet_loss=data['packet_loss'],
            utilization=data['utilization'],
            node_id=data['node_id']
        )

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'KPIMetrics':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class LoadForecast:
    """Load prediction results from predictive agent."""
    predicted_values: List[float]
    confidence_interval: Tuple[float, float]
    prediction_horizon: int  # seconds
    model_accuracy: float    # MAPE
    timestamp: datetime

    def __post_init__(self):
        """Validate load forecast after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate load forecast values."""
        if not self.predicted_values:
            raise ValueError("Predicted values cannot be empty")
        if any(val < 0 for val in self.predicted_values):
            raise ValueError("Predicted values must be non-negative")
        if self.confidence_interval[0] > self.confidence_interval[1]:
            raise ValueError("Confidence interval lower bound must be <= upper bound")
        if self.prediction_horizon <= 0:
            raise ValueError("Prediction horizon must be positive")
        if not (0 <= self.model_accuracy <= 100):
            raise ValueError("Model accuracy (MAPE) must be between 0 and 100 percent")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'predicted_values': self.predicted_values,
            'confidence_interval': list(self.confidence_interval),
            'prediction_horizon': self.prediction_horizon,
            'model_accuracy': self.model_accuracy,
            'timestamp': self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LoadForecast':
        """Create from dictionary for deserialization."""
        return cls(
            predicted_values=data['predicted_values'],
            confidence_interval=tuple(data['confidence_interval']),
            prediction_horizon=data['prediction_horizon'],
            model_accuracy=data['model_accuracy'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'LoadForecast':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class OptimizationDecision:
    """Decision made by optimization agent."""
    action_type: ActionType
    target_parameters: Dict[str, Any]
    rationale: str
    priority: int
    timestamp: datetime

    def __post_init__(self):
        """Validate optimization decision after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate optimization decision values."""
        if not self.rationale or not self.rationale.strip():
            raise ValueError("Rationale cannot be empty")
        if self.priority < 1:
            raise ValueError("Priority must be at least 1")
        for param_name, param_value in self.target_parameters.items():
            if not param_name or not param_name.strip():
                raise ValueError("Parameter name cannot be empty")
            # Skip validation for metadata fields (dictionaries) and allow negative values for some parameters
            if isinstance(param_value, (int, float)) and param_name not in ["confidence_lower", "confidence_upper", "load_deficit", "load_excess"]:
                if param_value < 0:
                    raise ValueError(f"Parameter value for {param_name} must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'action_type': self.action_type.value,
            'target_parameters': self.target_parameters,
            'rationale': self.rationale,
            'priority': self.priority,
            'timestamp': self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationDecision':
        """Create from dictionary for deserialization."""
        return cls(
            action_type=ActionType(data['action_type']),
            target_parameters=data['target_parameters'],
            rationale=data['rationale'],
            priority=data['priority'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'OptimizationDecision':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class Anomaly:
    """Network anomaly detected by anomaly agent."""
    anomaly_type: AnomalyType
    severity: SeverityLevel
    affected_nodes: List[str]
    detection_time: datetime
    confidence_score: float

    def __post_init__(self):
        """Validate anomaly after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate anomaly values."""
        if not self.affected_nodes:
            raise ValueError("Affected nodes list cannot be empty")
        for node_id in self.affected_nodes:
            if not node_id or not node_id.strip():
                raise ValueError("Node ID in affected nodes cannot be empty")
        if not (0 <= self.confidence_score <= 1):
            raise ValueError("Confidence score must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'anomaly_type': self.anomaly_type.value,
            'severity': self.severity.value,
            'affected_nodes': self.affected_nodes,
            'detection_time': self.detection_time.isoformat(),
            'confidence_score': self.confidence_score
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Anomaly':
        """Create from dictionary for deserialization."""
        return cls(
            anomaly_type=AnomalyType(data['anomaly_type']),
            severity=SeverityLevel(data['severity']),
            affected_nodes=data['affected_nodes'],
            detection_time=datetime.fromisoformat(data['detection_time']),
            confidence_score=data['confidence_score']
        )

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'Anomaly':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class NetworkParameters:
    """Network configuration parameters for simulation."""
    bandwidth: Dict[str, float]  # link_id -> bandwidth_mbps
    queue_size: Dict[str, int]   # node_id -> queue_size
    scheduling_algorithm: Dict[str, str]  # node_id -> algorithm
    update_timestamp: datetime

    def __post_init__(self):
        """Validate network parameters after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate network parameters."""
        # Validate bandwidth values
        for link_id, bw in self.bandwidth.items():
            if not link_id or not link_id.strip():
                raise ValueError("Link ID cannot be empty")
            if bw <= 0:
                raise ValueError(f"Bandwidth for {link_id} must be positive")
        
        # Validate queue sizes
        for node_id, size in self.queue_size.items():
            if not node_id or not node_id.strip():
                raise ValueError("Node ID cannot be empty")
            if size <= 0:
                raise ValueError(f"Queue size for {node_id} must be positive")
        
        # Validate scheduling algorithms
        valid_algorithms = {"FIFO", "WFQ", "PQ", "RR"}
        for node_id, algorithm in self.scheduling_algorithm.items():
            if not node_id or not node_id.strip():
                raise ValueError("Node ID cannot be empty")
            if algorithm not in valid_algorithms:
                raise ValueError(f"Invalid scheduling algorithm '{algorithm}' for {node_id}. "
                               f"Must be one of: {valid_algorithms}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'bandwidth': self.bandwidth,
            'queue_size': self.queue_size,
            'scheduling_algorithm': self.scheduling_algorithm,
            'update_timestamp': self.update_timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NetworkParameters':
        """Create from dictionary for deserialization."""
        return cls(
            bandwidth=data['bandwidth'],
            queue_size=data['queue_size'],
            scheduling_algorithm=data['scheduling_algorithm'],
            update_timestamp=datetime.fromisoformat(data['update_timestamp'])
        )

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'NetworkParameters':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class NetworkTopology:
    """Network topology configuration."""
    nodes: List[str]
    links: Dict[str, Tuple[str, str]]  # link_id -> (source_node, dest_node)
    node_types: Dict[str, str]  # node_id -> node_type (UE, eNodeB, CoreRouter, Server)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'nodes': self.nodes,
            'links': {k: list(v) for k, v in self.links.items()},
            'node_types': self.node_types
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NetworkTopology':
        """Create from dictionary for deserialization."""
        return cls(
            nodes=data['nodes'],
            links={k: tuple(v) for k, v in data['links'].items()},
            node_types=data['node_types']
        )
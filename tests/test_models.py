"""
Tests for core data models.
"""

import pytest
import json
from datetime import datetime
from hypothesis import given, strategies as st
from tests.conftest import (
    kpi_metrics_strategy, load_forecast_strategy, optimization_decision_strategy,
    anomaly_strategy, network_parameters_strategy
)
from src.models import (
    KPIMetrics, LoadForecast, OptimizationDecision, Anomaly, NetworkParameters,
    ActionType, SeverityLevel, AnomalyType
)


class TestKPIMetrics:
    """Test KPIMetrics data model."""
    
    def test_kpi_metrics_creation(self, sample_kpi_metrics):
        """Test basic KPI metrics creation."""
        assert sample_kpi_metrics.throughput == 100.5
        assert sample_kpi_metrics.latency == 25.3
        assert sample_kpi_metrics.packet_loss == 0.1
        assert sample_kpi_metrics.utilization == 75.0
        assert sample_kpi_metrics.node_id == "UE1"
    
    def test_kpi_metrics_validation(self):
        """Test KPI metrics validation."""
        # Test negative throughput
        with pytest.raises(ValueError, match="Throughput must be non-negative"):
            KPIMetrics(
                timestamp=datetime.now(),
                throughput=-1.0,
                latency=25.3,
                packet_loss=0.1,
                utilization=75.0,
                node_id="UE1"
            )
        
        # Test negative latency
        with pytest.raises(ValueError, match="Latency must be non-negative"):
            KPIMetrics(
                timestamp=datetime.now(),
                throughput=100.5,
                latency=-1.0,
                packet_loss=0.1,
                utilization=75.0,
                node_id="UE1"
            )
        
        # Test invalid packet loss
        with pytest.raises(ValueError, match="Packet loss must be between 0 and 100 percent"):
            KPIMetrics(
                timestamp=datetime.now(),
                throughput=100.5,
                latency=25.3,
                packet_loss=150.0,
                utilization=75.0,
                node_id="UE1"
            )
        
        # Test invalid utilization
        with pytest.raises(ValueError, match="Utilization must be between 0 and 100 percent"):
            KPIMetrics(
                timestamp=datetime.now(),
                throughput=100.5,
                latency=25.3,
                packet_loss=0.1,
                utilization=150.0,
                node_id="UE1"
            )
        
        # Test empty node ID
        with pytest.raises(ValueError, match="Node ID cannot be empty"):
            KPIMetrics(
                timestamp=datetime.now(),
                throughput=100.5,
                latency=25.3,
                packet_loss=0.1,
                utilization=75.0,
                node_id=""
            )
    
    def test_kpi_metrics_serialization(self, sample_kpi_metrics):
        """Test KPI metrics serialization to dict and JSON."""
        # Test to_dict
        data_dict = sample_kpi_metrics.to_dict()
        assert isinstance(data_dict, dict)
        assert data_dict['throughput'] == 100.5
        assert data_dict['node_id'] == "UE1"
        
        # Test to_json
        json_str = sample_kpi_metrics.to_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed['throughput'] == 100.5
    
    def test_kpi_metrics_deserialization(self, sample_kpi_metrics):
        """Test KPI metrics deserialization from dict and JSON."""
        # Test round-trip through dict
        data_dict = sample_kpi_metrics.to_dict()
        restored = KPIMetrics.from_dict(data_dict)
        assert restored.throughput == sample_kpi_metrics.throughput
        assert restored.node_id == sample_kpi_metrics.node_id
        
        # Test round-trip through JSON
        json_str = sample_kpi_metrics.to_json()
        restored_json = KPIMetrics.from_json(json_str)
        assert restored_json.throughput == sample_kpi_metrics.throughput
        assert restored_json.node_id == sample_kpi_metrics.node_id
    
    @given(kpi_metrics_strategy())
    def test_kpi_metrics_serialization_roundtrip(self, kpi_metrics):
        """Property test: KPI metrics serialization should be reversible."""
        # Test dict round-trip
        dict_restored = KPIMetrics.from_dict(kpi_metrics.to_dict())
        assert dict_restored.throughput == kpi_metrics.throughput
        assert dict_restored.latency == kpi_metrics.latency
        assert dict_restored.packet_loss == kpi_metrics.packet_loss
        assert dict_restored.utilization == kpi_metrics.utilization
        assert dict_restored.node_id == kpi_metrics.node_id
        assert dict_restored.timestamp == kpi_metrics.timestamp
        
        # Test JSON round-trip
        json_restored = KPIMetrics.from_json(kpi_metrics.to_json())
        assert json_restored.throughput == kpi_metrics.throughput
        assert json_restored.latency == kpi_metrics.latency
        assert json_restored.packet_loss == kpi_metrics.packet_loss
        assert json_restored.utilization == kpi_metrics.utilization
        assert json_restored.node_id == kpi_metrics.node_id
        assert json_restored.timestamp == kpi_metrics.timestamp


class TestLoadForecast:
    """Test LoadForecast data model."""
    
    def test_load_forecast_creation(self, sample_load_forecast):
        """Test basic load forecast creation."""
        assert len(sample_load_forecast.predicted_values) == 5
        assert sample_load_forecast.prediction_horizon == 10
        assert sample_load_forecast.model_accuracy == 12.5
    
    def test_load_forecast_validation(self):
        """Test load forecast validation."""
        # Test empty predicted values
        with pytest.raises(ValueError, match="Predicted values cannot be empty"):
            LoadForecast(
                predicted_values=[],
                confidence_interval=(75.0, 105.0),
                prediction_horizon=10,
                model_accuracy=12.5,
                timestamp=datetime.now()
            )
        
        # Test negative predicted values
        with pytest.raises(ValueError, match="Predicted values must be non-negative"):
            LoadForecast(
                predicted_values=[80.0, -5.0, 90.0],
                confidence_interval=(75.0, 105.0),
                prediction_horizon=10,
                model_accuracy=12.5,
                timestamp=datetime.now()
            )
        
        # Test invalid confidence interval
        with pytest.raises(ValueError, match="Confidence interval lower bound must be <= upper bound"):
            LoadForecast(
                predicted_values=[80.0, 85.0, 90.0],
                confidence_interval=(105.0, 75.0),
                prediction_horizon=10,
                model_accuracy=12.5,
                timestamp=datetime.now()
            )
        
        # Test invalid prediction horizon
        with pytest.raises(ValueError, match="Prediction horizon must be positive"):
            LoadForecast(
                predicted_values=[80.0, 85.0, 90.0],
                confidence_interval=(75.0, 105.0),
                prediction_horizon=0,
                model_accuracy=12.5,
                timestamp=datetime.now()
            )
        
        # Test invalid model accuracy
        with pytest.raises(ValueError, match="Model accuracy \\(MAPE\\) must be between 0 and 100 percent"):
            LoadForecast(
                predicted_values=[80.0, 85.0, 90.0],
                confidence_interval=(75.0, 105.0),
                prediction_horizon=10,
                model_accuracy=150.0,
                timestamp=datetime.now()
            )
    
    @given(load_forecast_strategy())
    def test_load_forecast_serialization_roundtrip(self, load_forecast):
        """Property test: Load forecast serialization should be reversible."""
        # Test dict round-trip
        dict_restored = LoadForecast.from_dict(load_forecast.to_dict())
        assert dict_restored.predicted_values == load_forecast.predicted_values
        assert dict_restored.confidence_interval == load_forecast.confidence_interval
        assert dict_restored.prediction_horizon == load_forecast.prediction_horizon
        assert dict_restored.model_accuracy == load_forecast.model_accuracy
        assert dict_restored.timestamp == load_forecast.timestamp
        
        # Test JSON round-trip
        json_restored = LoadForecast.from_json(load_forecast.to_json())
        assert json_restored.predicted_values == load_forecast.predicted_values
        assert json_restored.confidence_interval == load_forecast.confidence_interval
        assert json_restored.prediction_horizon == load_forecast.prediction_horizon
        assert json_restored.model_accuracy == load_forecast.model_accuracy
        assert json_restored.timestamp == load_forecast.timestamp


class TestOptimizationDecision:
    """Test OptimizationDecision data model."""
    
    def test_optimization_decision_creation(self, sample_optimization_decision):
        """Test basic optimization decision creation."""
        assert sample_optimization_decision.action_type == ActionType.INCREASE_CAPACITY
        assert sample_optimization_decision.priority == 1
        assert "capacity expansion" in sample_optimization_decision.rationale
    
    def test_optimization_decision_validation(self):
        """Test optimization decision validation."""
        # Test empty rationale
        with pytest.raises(ValueError, match="Rationale cannot be empty"):
            OptimizationDecision(
                action_type=ActionType.INCREASE_CAPACITY,
                target_parameters={"bandwidth": 150.0},
                rationale="",
                priority=1,
                timestamp=datetime.now()
            )
        
        # Test whitespace-only rationale
        with pytest.raises(ValueError, match="Rationale cannot be empty"):
            OptimizationDecision(
                action_type=ActionType.INCREASE_CAPACITY,
                target_parameters={"bandwidth": 150.0},
                rationale="   ",
                priority=1,
                timestamp=datetime.now()
            )
        
        # Test invalid priority (less than 1)
        with pytest.raises(ValueError, match="Priority must be at least 1"):
            OptimizationDecision(
                action_type=ActionType.INCREASE_CAPACITY,
                target_parameters={"bandwidth": 150.0},
                rationale="Valid rationale",
                priority=0,
                timestamp=datetime.now()
            )
        
        # Test empty parameter name
        with pytest.raises(ValueError, match="Parameter name cannot be empty"):
            OptimizationDecision(
                action_type=ActionType.INCREASE_CAPACITY,
                target_parameters={"": 150.0},
                rationale="Valid rationale",
                priority=1,
                timestamp=datetime.now()
            )
        
        # Test negative parameter value
        with pytest.raises(ValueError, match="Parameter value for bandwidth must be non-negative"):
            OptimizationDecision(
                action_type=ActionType.INCREASE_CAPACITY,
                target_parameters={"bandwidth": -150.0},
                rationale="Valid rationale",
                priority=1,
                timestamp=datetime.now()
            )
    
    @given(optimization_decision_strategy())
    def test_optimization_decision_serialization_roundtrip(self, decision):
        """Property test: Optimization decision serialization should be reversible."""
        # Test dict round-trip
        dict_restored = OptimizationDecision.from_dict(decision.to_dict())
        assert dict_restored.action_type == decision.action_type
        assert dict_restored.target_parameters == decision.target_parameters
        assert dict_restored.rationale == decision.rationale
        assert dict_restored.priority == decision.priority
        assert dict_restored.timestamp == decision.timestamp
        
        # Test JSON round-trip
        json_restored = OptimizationDecision.from_json(decision.to_json())
        assert json_restored.action_type == decision.action_type
        assert json_restored.target_parameters == decision.target_parameters
        assert json_restored.rationale == decision.rationale
        assert json_restored.priority == decision.priority
        assert json_restored.timestamp == decision.timestamp


class TestAnomaly:
    """Test Anomaly data model."""
    
    def test_anomaly_creation(self, sample_anomaly):
        """Test basic anomaly creation."""
        assert sample_anomaly.anomaly_type == AnomalyType.PACKET_LOSS
        assert sample_anomaly.severity == SeverityLevel.HIGH
        assert sample_anomaly.confidence_score == 0.95
        assert len(sample_anomaly.affected_nodes) == 2
    
    def test_anomaly_validation(self):
        """Test anomaly validation."""
        # Test empty affected nodes list
        with pytest.raises(ValueError, match="Affected nodes list cannot be empty"):
            Anomaly(
                anomaly_type=AnomalyType.PACKET_LOSS,
                severity=SeverityLevel.HIGH,
                affected_nodes=[],
                detection_time=datetime.now(),
                confidence_score=0.95
            )
        
        # Test empty node ID in affected nodes
        with pytest.raises(ValueError, match="Node ID in affected nodes cannot be empty"):
            Anomaly(
                anomaly_type=AnomalyType.PACKET_LOSS,
                severity=SeverityLevel.HIGH,
                affected_nodes=["UE1", ""],
                detection_time=datetime.now(),
                confidence_score=0.95
            )
        
        # Test whitespace-only node ID in affected nodes
        with pytest.raises(ValueError, match="Node ID in affected nodes cannot be empty"):
            Anomaly(
                anomaly_type=AnomalyType.PACKET_LOSS,
                severity=SeverityLevel.HIGH,
                affected_nodes=["UE1", "   "],
                detection_time=datetime.now(),
                confidence_score=0.95
            )
        
        # Test confidence score below 0
        with pytest.raises(ValueError, match="Confidence score must be between 0 and 1"):
            Anomaly(
                anomaly_type=AnomalyType.PACKET_LOSS,
                severity=SeverityLevel.HIGH,
                affected_nodes=["UE1"],
                detection_time=datetime.now(),
                confidence_score=-0.1
            )
        
        # Test confidence score above 1
        with pytest.raises(ValueError, match="Confidence score must be between 0 and 1"):
            Anomaly(
                anomaly_type=AnomalyType.PACKET_LOSS,
                severity=SeverityLevel.HIGH,
                affected_nodes=["UE1"],
                detection_time=datetime.now(),
                confidence_score=1.5
            )
    
    @given(anomaly_strategy())
    def test_anomaly_serialization_roundtrip(self, anomaly):
        """Property test: Anomaly serialization should be reversible."""
        # Test dict round-trip
        dict_restored = Anomaly.from_dict(anomaly.to_dict())
        assert dict_restored.anomaly_type == anomaly.anomaly_type
        assert dict_restored.severity == anomaly.severity
        assert dict_restored.affected_nodes == anomaly.affected_nodes
        assert dict_restored.detection_time == anomaly.detection_time
        assert dict_restored.confidence_score == anomaly.confidence_score
        
        # Test JSON round-trip
        json_restored = Anomaly.from_json(anomaly.to_json())
        assert json_restored.anomaly_type == anomaly.anomaly_type
        assert json_restored.severity == anomaly.severity
        assert json_restored.affected_nodes == anomaly.affected_nodes
        assert json_restored.detection_time == anomaly.detection_time
        assert json_restored.confidence_score == anomaly.confidence_score


class TestNetworkParameters:
    """Test NetworkParameters data model."""
    
    def test_network_parameters_creation(self, sample_network_parameters):
        """Test basic network parameters creation."""
        assert sample_network_parameters.bandwidth["link1"] == 100.0
        assert sample_network_parameters.queue_size["UE1"] == 50
        assert sample_network_parameters.scheduling_algorithm["UE1"] == "FIFO"
    
    def test_network_parameters_validation(self):
        """Test network parameters validation."""
        # Test invalid bandwidth
        with pytest.raises(ValueError, match="Bandwidth for link1 must be positive"):
            NetworkParameters(
                bandwidth={"link1": 0.0},
                queue_size={"UE1": 50},
                scheduling_algorithm={"UE1": "FIFO"},
                update_timestamp=datetime.now()
            )
        
        # Test invalid queue size
        with pytest.raises(ValueError, match="Queue size for UE1 must be positive"):
            NetworkParameters(
                bandwidth={"link1": 100.0},
                queue_size={"UE1": 0},
                scheduling_algorithm={"UE1": "FIFO"},
                update_timestamp=datetime.now()
            )
        
        # Test invalid scheduling algorithm
        with pytest.raises(ValueError, match="Invalid scheduling algorithm 'INVALID' for UE1"):
            NetworkParameters(
                bandwidth={"link1": 100.0},
                queue_size={"UE1": 50},
                scheduling_algorithm={"UE1": "INVALID"},
                update_timestamp=datetime.now()
            )
        
        # Test empty link ID
        with pytest.raises(ValueError, match="Link ID cannot be empty"):
            NetworkParameters(
                bandwidth={"": 100.0},
                queue_size={"UE1": 50},
                scheduling_algorithm={"UE1": "FIFO"},
                update_timestamp=datetime.now()
            )
    
    @given(network_parameters_strategy())
    def test_network_parameters_serialization_roundtrip(self, params):
        """Property test: Network parameters serialization should be reversible."""
        # Test dict round-trip
        dict_restored = NetworkParameters.from_dict(params.to_dict())
        assert dict_restored.bandwidth == params.bandwidth
        assert dict_restored.queue_size == params.queue_size
        assert dict_restored.scheduling_algorithm == params.scheduling_algorithm
        assert dict_restored.update_timestamp == params.update_timestamp
        
        # Test JSON round-trip
        json_restored = NetworkParameters.from_json(params.to_json())
        assert json_restored.bandwidth == params.bandwidth
        assert json_restored.queue_size == params.queue_size
        assert json_restored.scheduling_algorithm == params.scheduling_algorithm
        assert json_restored.update_timestamp == params.update_timestamp
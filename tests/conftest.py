"""
Pytest configuration and fixtures for AI telecom optimization system tests.
"""

import pytest
from datetime import datetime, timedelta
from typing import List
from hypothesis import strategies as st
from src.models import (
    KPIMetrics, LoadForecast, OptimizationDecision, Anomaly, 
    NetworkParameters, ActionType, SeverityLevel, AnomalyType
)


@pytest.fixture
def sample_kpi_metrics():
    """Sample KPI metrics for testing."""
    return KPIMetrics(
        timestamp=datetime.now(),
        throughput=100.5,
        latency=25.3,
        packet_loss=0.1,
        utilization=75.0,
        node_id="UE1"
    )


@pytest.fixture
def sample_load_forecast():
    """Sample load forecast for testing."""
    return LoadForecast(
        predicted_values=[80.0, 85.0, 90.0, 95.0, 100.0],
        confidence_interval=(75.0, 105.0),
        prediction_horizon=10,
        model_accuracy=12.5,
        timestamp=datetime.now()
    )


@pytest.fixture
def sample_optimization_decision():
    """Sample optimization decision for testing."""
    return OptimizationDecision(
        action_type=ActionType.INCREASE_CAPACITY,
        target_parameters={"bandwidth": 150.0, "queue_size": 100},
        rationale="Predicted load increase requires capacity expansion",
        priority=1,
        timestamp=datetime.now()
    )


@pytest.fixture
def sample_anomaly():
    """Sample anomaly for testing."""
    return Anomaly(
        anomaly_type=AnomalyType.PACKET_LOSS,
        severity=SeverityLevel.HIGH,
        affected_nodes=["UE1", "eNodeB1"],
        detection_time=datetime.now(),
        confidence_score=0.95
    )


@pytest.fixture
def sample_network_parameters():
    """Sample network parameters for testing."""
    return NetworkParameters(
        bandwidth={"link1": 100.0, "link2": 150.0},
        queue_size={"UE1": 50, "eNodeB1": 100},
        scheduling_algorithm={"UE1": "FIFO", "eNodeB1": "WFQ"},
        update_timestamp=datetime.now()
    )


# Hypothesis strategies for property-based testing
@st.composite
def kpi_metrics_strategy(draw):
    """Strategy for generating KPI metrics."""
    return KPIMetrics(
        timestamp=draw(st.datetimes(
            min_value=datetime(2023, 1, 1),
            max_value=datetime(2025, 12, 31)
        )),
        throughput=draw(st.floats(min_value=0.0, max_value=1000.0)),
        latency=draw(st.floats(min_value=0.0, max_value=1000.0)),
        packet_loss=draw(st.floats(min_value=0.0, max_value=100.0)),
        utilization=draw(st.floats(min_value=0.0, max_value=100.0)),
        node_id=draw(st.text(
            min_size=1, max_size=20, 
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))
        ).filter(lambda x: x.strip()))
    )


@st.composite
def load_forecast_strategy(draw):
    """Strategy for generating load forecasts."""
    predicted_values = draw(st.lists(
        st.floats(min_value=0.0, max_value=100.0),
        min_size=1, max_size=20
    ))
    min_val = min(predicted_values) if predicted_values else 0.0
    max_val = max(predicted_values) if predicted_values else 100.0
    
    # Ensure confidence interval is valid (lower <= upper)
    lower_bound = draw(st.floats(min_value=0.0, max_value=min_val))
    upper_bound = draw(st.floats(min_value=max(lower_bound, max_val), max_value=200.0))
    
    return LoadForecast(
        predicted_values=predicted_values,
        confidence_interval=(lower_bound, upper_bound),
        prediction_horizon=draw(st.integers(min_value=1, max_value=60)),
        model_accuracy=draw(st.floats(min_value=0.0, max_value=100.0)),
        timestamp=draw(st.datetimes(
            min_value=datetime(2023, 1, 1),
            max_value=datetime(2025, 12, 31)
        ))
    )


@st.composite
def optimization_decision_strategy(draw):
    """Strategy for generating optimization decisions."""
    valid_id_strategy = st.text(
        min_size=1, max_size=20, 
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))
    ).filter(lambda x: x.strip())
    
    return OptimizationDecision(
        action_type=draw(st.sampled_from(ActionType)),
        target_parameters=draw(st.dictionaries(
            valid_id_strategy,
            st.floats(min_value=0.0, max_value=1000.0),
            min_size=1, max_size=10
        )),
        rationale=draw(st.text(min_size=1, max_size=200).filter(lambda x: x.strip())),
        priority=draw(st.integers(min_value=1, max_value=10)),
        timestamp=draw(st.datetimes(
            min_value=datetime(2023, 1, 1),
            max_value=datetime(2025, 12, 31)
        ))
    )


@st.composite
def anomaly_strategy(draw):
    """Strategy for generating anomalies."""
    valid_node_id_strategy = st.text(
        min_size=1, max_size=20, 
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))
    ).filter(lambda x: x.strip())
    
    return Anomaly(
        anomaly_type=draw(st.sampled_from(AnomalyType)),
        severity=draw(st.sampled_from(SeverityLevel)),
        affected_nodes=draw(st.lists(
            valid_node_id_strategy,
            min_size=1, max_size=10
        )),
        detection_time=draw(st.datetimes(
            min_value=datetime(2023, 1, 1),
            max_value=datetime(2025, 12, 31)
        )),
        confidence_score=draw(st.floats(min_value=0.0, max_value=1.0))
    )


@st.composite
def network_parameters_strategy(draw):
    """Strategy for generating network parameters."""
    # Generate valid non-empty identifiers
    valid_id_strategy = st.text(
        min_size=1, max_size=20, 
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))
    ).filter(lambda x: x.strip())  # Ensure non-empty after stripping
    
    return NetworkParameters(
        bandwidth=draw(st.dictionaries(
            valid_id_strategy,
            st.floats(min_value=1.0, max_value=1000.0),
            min_size=1, max_size=10
        )),
        queue_size=draw(st.dictionaries(
            valid_id_strategy,
            st.integers(min_value=1, max_value=1000),
            min_size=1, max_size=10
        )),
        scheduling_algorithm=draw(st.dictionaries(
            valid_id_strategy,
            st.sampled_from(["FIFO", "WFQ", "PQ", "RR"]),
            min_size=1, max_size=10
        )),
        update_timestamp=draw(st.datetimes(
            min_value=datetime(2023, 1, 1),
            max_value=datetime(2025, 12, 31)
        ))
    )


# Configure Hypothesis settings
from hypothesis import settings, Verbosity

# Global Hypothesis settings
settings.register_profile("default", max_examples=100, verbosity=Verbosity.normal)
settings.register_profile("ci", max_examples=1000, verbosity=Verbosity.verbose)
settings.load_profile("default")
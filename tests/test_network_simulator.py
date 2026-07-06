"""
Tests for NetworkSimulator implementation.
"""

import pytest
from datetime import datetime
from src.simulation import create_network_simulator, MockNetworkSimulator, MininetNetworkSimulator
from src.models import NetworkParameters, KPIMetrics


class TestNetworkSimulator:
    """Test cases for NetworkSimulator implementations."""
    
    def test_mock_simulator_initialization(self):
        """Test mock simulator initialization and topology setup."""
        simulator = create_network_simulator(use_mininet=False)
        
        # Test topology initialization
        topology = simulator.initialize_topology()
        
        assert topology is not None
        assert len(topology.nodes) == 5
        assert 'ue1' in topology.nodes
        assert 'ue2' in topology.nodes
        assert 'enodeb' in topology.nodes
        assert 'core_router' in topology.nodes
        assert 'server' in topology.nodes
        
        # Verify node types
        assert topology.node_types['ue1'] == 'UE'
        assert topology.node_types['ue2'] == 'UE'
        assert topology.node_types['enodeb'] == 'eNodeB'
        assert topology.node_types['core_router'] == 'CoreRouter'
        assert topology.node_types['server'] == 'Server'
        
        # Verify links
        assert len(topology.links) == 4
        assert topology.links['ue1_enodeb'] == ('ue1', 'enodeb')
        assert topology.links['enodeb_core'] == ('enodeb', 'core_router')
        assert topology.links['core_server'] == ('core_router', 'server')
        assert topology.links['ue2_server'] == ('ue2', 'server')
        
        # Verify simulator state
        assert simulator.is_simulation_running()
        
        # Clean up
        simulator.stop_simulation()
        assert not simulator.is_simulation_running()
    
    def test_traffic_generation(self):
        """Test traffic generation functionality."""
        simulator = create_network_simulator(use_mininet=False)
        
        # Initialize topology first
        simulator.initialize_topology()
        
        # Test traffic generation
        assert not simulator.is_traffic_running()
        simulator.start_traffic_generation()
        assert simulator.is_traffic_running()
        
        # Clean up
        simulator.stop_simulation()
    
    def test_kpi_collection(self):
        """Test KPI metrics collection."""
        simulator = create_network_simulator(use_mininet=False)
        
        # Initialize and start traffic
        simulator.initialize_topology()
        simulator.start_traffic_generation()
        
        # Collect KPIs
        metrics = simulator.collect_kpis()
        
        assert isinstance(metrics, KPIMetrics)
        assert metrics.node_id in {'ue1', 'ue2', 'enodeb', 'core_router', 'server'}
        assert metrics.throughput >= 0
        assert metrics.latency >= 0
        assert 0 <= metrics.packet_loss <= 100
        assert 0 <= metrics.utilization <= 100
        assert isinstance(metrics.timestamp, datetime)
        
        # Clean up
        simulator.stop_simulation()
    
    def test_parameter_updates(self):
        """Test dynamic parameter updates."""
        simulator = create_network_simulator(use_mininet=False)
        
        # Initialize topology
        simulator.initialize_topology()
        
        # Create new parameters
        new_params = NetworkParameters(
            bandwidth={
                'ue1_enodeb': 20.0,  # Increased from 10.0
                'enodeb_core': 150.0,  # Increased from 100.0
                'core_server': 75.0,   # Increased from 50.0
                'ue2_server': 30.0     # Increased from 20.0
            },
            queue_size={
                'ue1': 150,
                'ue2': 150,
                'enodeb': 300,
                'core_router': 250,
                'server': 150
            },
            scheduling_algorithm={
                'ue1': 'WFQ',  # Changed from FIFO
                'ue2': 'WFQ',  # Changed from FIFO
                'enodeb': 'PQ',  # Changed from WFQ
                'core_router': 'RR',  # Changed from WFQ
                'server': 'FIFO'
            },
            update_timestamp=datetime.now()
        )
        
        # Update parameters
        simulator.update_parameters(new_params)
        
        # Verify parameters were updated
        current_params = simulator.current_parameters
        assert current_params.bandwidth['ue1_enodeb'] == 20.0
        assert current_params.queue_size['ue1'] == 150
        assert current_params.scheduling_algorithm['ue1'] == 'WFQ'
        
        # Clean up
        simulator.stop_simulation()
    
    def test_enhanced_parameter_updates_with_validation(self):
        """Test enhanced parameter updates with validation and rollback mechanisms."""
        simulator = create_network_simulator(use_mininet=False)
        simulator.initialize_topology()
        
        # Test valid parameter updates
        valid_params = NetworkParameters(
            bandwidth={
                'ue1_enodeb': 25.0,
                'enodeb_core': 200.0,
                'core_server': 100.0,
                'ue2_server': 40.0
            },
            queue_size={
                'ue1': 200,
                'ue2': 200,
                'enodeb': 400,
                'core_router': 300,
                'server': 200
            },
            scheduling_algorithm={
                'ue1': 'WFQ',
                'ue2': 'PQ',
                'enodeb': 'RR',
                'core_router': 'FIFO',
                'server': 'WFQ'
            },
            update_timestamp=datetime.now()
        )
        
        # Should succeed
        simulator.update_parameters(valid_params)
        assert simulator.current_parameters.bandwidth['ue1_enodeb'] == 25.0
        assert simulator.current_parameters.queue_size['ue1'] == 200
        assert simulator.current_parameters.scheduling_algorithm['ue1'] == 'WFQ'
        
        # Test parameter validation
        assert simulator.validate_current_parameters() == True
        
        # Test parameter history
        history = simulator.get_parameter_history()
        assert len(history) >= 1
        assert 'timestamp' in history[0]
        assert 'parameters' in history[0]
        
        simulator.stop_simulation()
    
    def test_parameter_validation_constraints(self):
        """Test network-specific parameter validation constraints."""
        simulator = create_network_simulator(use_mininet=False)
        simulator.initialize_topology()
        
        # Test bandwidth constraints - too high
        with pytest.raises(ValueError, match="exceeds maximum limit"):
            invalid_params = NetworkParameters(
                bandwidth={'ue1_enodeb': 2000.0},  # Exceeds 1000 Mbps limit
                queue_size={'ue1': 100},
                scheduling_algorithm={'ue1': 'FIFO'},
                update_timestamp=datetime.now()
            )
            simulator.update_parameters(invalid_params)
        
        # Test bandwidth constraints - too low
        with pytest.raises(ValueError, match="below minimum limit"):
            invalid_params = NetworkParameters(
                bandwidth={'ue1_enodeb': 0.05},  # Below 0.1 Mbps limit
                queue_size={'ue1': 100},
                scheduling_algorithm={'ue1': 'FIFO'},
                update_timestamp=datetime.now()
            )
            simulator.update_parameters(invalid_params)
        
        # Test queue size constraints - too high
        with pytest.raises(ValueError, match="exceeds maximum limit"):
            invalid_params = NetworkParameters(
                bandwidth={'ue1_enodeb': 10.0},
                queue_size={'ue1': 15000},  # Exceeds 10000 limit
                scheduling_algorithm={'ue1': 'FIFO'},
                update_timestamp=datetime.now()
            )
            simulator.update_parameters(invalid_params)
        
        # Test queue size constraints - too low
        with pytest.raises(ValueError, match="below minimum limit"):
            invalid_params = NetworkParameters(
                bandwidth={'ue1_enodeb': 10.0},
                queue_size={'ue1': 5},  # Below 10 limit
                scheduling_algorithm={'ue1': 'FIFO'},
                update_timestamp=datetime.now()
            )
            simulator.update_parameters(invalid_params)
        
        simulator.stop_simulation()
    
    def test_partial_parameter_updates(self):
        """Test partial parameter updates with missing topology elements."""
        simulator = create_network_simulator(use_mininet=False)
        simulator.initialize_topology()
        
        # Create parameters with some non-existent elements
        partial_params = NetworkParameters(
            bandwidth={
                'ue1_enodeb': 15.0,  # Valid link
                'nonexistent_link': 50.0  # Invalid link - should be skipped
            },
            queue_size={
                'ue1': 120,  # Valid node
                'nonexistent_node': 200  # Invalid node - should be skipped
            },
            scheduling_algorithm={
                'ue1': 'WFQ',  # Valid node
                'nonexistent_node': 'PQ'  # Invalid node - should be skipped
            },
            update_timestamp=datetime.now()
        )
        
        # Should succeed and skip non-existent elements
        simulator.update_parameters(partial_params)
        
        # Verify valid parameters were updated
        assert simulator.current_parameters.bandwidth['ue1_enodeb'] == 15.0
        assert simulator.current_parameters.queue_size['ue1'] == 120
        assert simulator.current_parameters.scheduling_algorithm['ue1'] == 'WFQ'
        
        # Verify non-existent elements are still in parameters but were skipped during application
        assert 'nonexistent_link' in simulator.current_parameters.bandwidth
        assert 'nonexistent_node' in simulator.current_parameters.queue_size
        
        simulator.stop_simulation()

    def test_scenario_packet_loss_is_nonzero_and_scaled(self):
        """Test that baseline scenarios have small loss and stressed scenarios have higher loss."""

        def collect_for_scenario(bandwidth, queue_size, scheduling_algorithm, traffic_load):
            simulator = create_network_simulator(use_mininet=False)
            simulator.initialize_topology()
            simulator.start_traffic_generation()
            simulator.update_parameters(
                NetworkParameters(
                    bandwidth=bandwidth,
                    queue_size=queue_size,
                    scheduling_algorithm=scheduling_algorithm,
                    update_timestamp=datetime.now(),
                )
            )
            simulator.set_traffic_load(traffic_load)
            metrics = simulator.collect_kpis()
            simulator.stop_simulation()
            return metrics.packet_loss

        normal_loss = collect_for_scenario(
            bandwidth={
                'ue1_enodeb': 10.0,
                'enodeb_core': 100.0,
                'core_server': 50.0,
                'ue2_server': 20.0,
            },
            queue_size={
                'ue1': 100,
                'ue2': 100,
                'enodeb': 200,
                'core_router': 150,
                'server': 100,
            },
            scheduling_algorithm={
                'ue1': 'FIFO',
                'ue2': 'FIFO',
                'enodeb': 'WFQ',
                'core_router': 'WFQ',
                'server': 'FIFO',
            },
            traffic_load={'ue1': 0.60, 'ue2': 0.40},
        )

        optimized_loss = collect_for_scenario(
            bandwidth={
                'ue1_enodeb': 80.0,
                'enodeb_core': 500.0,
                'core_server': 300.0,
                'ue2_server': 80.0,
            },
            queue_size={
                'ue1': 500,
                'ue2': 500,
                'enodeb': 1000,
                'core_router': 800,
                'server': 500,
            },
            scheduling_algorithm={
                'ue1': 'FIFO',
                'ue2': 'FIFO',
                'enodeb': 'PQ',
                'core_router': 'PQ',
                'server': 'FIFO',
            },
            traffic_load={'ue1': 0.50, 'ue2': 0.35},
        )

        congestion_loss = collect_for_scenario(
            bandwidth={
                'ue1_enodeb': 3.0,
                'enodeb_core': 20.0,
                'core_server': 10.0,
                'ue2_server': 5.0,
            },
            queue_size={
                'ue1': 50,
                'ue2': 50,
                'enodeb': 60,
                'core_router': 60,
                'server': 50,
            },
            scheduling_algorithm={
                'ue1': 'FIFO',
                'ue2': 'FIFO',
                'enodeb': 'FIFO',
                'core_router': 'FIFO',
                'server': 'FIFO',
            },
            traffic_load={'ue1': 0.95, 'ue2': 0.90},
        )

        failure_loss = collect_for_scenario(
            bandwidth={
                'ue1_enodeb': 1.0,
                'enodeb_core': 100.0,
                'core_server': 50.0,
                'ue2_server': 20.0,
            },
            queue_size={
                'ue1': 30,
                'ue2': 100,
                'enodeb': 200,
                'core_router': 150,
                'server': 100,
            },
            scheduling_algorithm={
                'ue1': 'FIFO',
                'ue2': 'FIFO',
                'enodeb': 'WFQ',
                'core_router': 'WFQ',
                'server': 'FIFO',
            },
            traffic_load={'ue1': 0.95, 'ue2': 0.40},
        )

        assert normal_loss > 0.0
        assert optimized_loss > 0.0
        assert normal_loss < 1.0
        assert optimized_loss < 1.0
        assert congestion_loss > normal_loss
        assert failure_loss > congestion_loss
        assert congestion_loss > 5.0 or failure_loss > 5.0
    
    def test_parameter_update_without_simulation(self):
        """Test parameter update error handling when simulation is not running."""
        simulator = create_network_simulator(use_mininet=False)
        
        # Don't initialize topology
        params = NetworkParameters(
            bandwidth={'ue1_enodeb': 10.0},
            queue_size={'ue1': 100},
            scheduling_algorithm={'ue1': 'FIFO'},
            update_timestamp=datetime.now()
        )
        
        # Should fail because simulation is not running
        with pytest.raises(RuntimeError, match="Network must be running"):
            simulator.update_parameters(params)
    
    def test_invalid_parameter_updates(self):
        """Test parameter validation during updates."""
        simulator = create_network_simulator(use_mininet=False)
        simulator.initialize_topology()
        
        # Test invalid bandwidth (negative value) - should fail during construction
        with pytest.raises(ValueError):
            NetworkParameters(
                bandwidth={'ue1_enodeb': -10.0},  # Invalid negative bandwidth
                queue_size={'ue1': 100},
                scheduling_algorithm={'ue1': 'FIFO'},
                update_timestamp=datetime.now()
            )
        
        # Test invalid queue size (zero) - should fail during construction
        with pytest.raises(ValueError):
            NetworkParameters(
                bandwidth={'ue1_enodeb': 10.0},
                queue_size={'ue1': 0},  # Invalid zero queue size
                scheduling_algorithm={'ue1': 'FIFO'},
                update_timestamp=datetime.now()
            )
        
        # Test invalid scheduling algorithm - should fail during construction
        with pytest.raises(ValueError):
            NetworkParameters(
                bandwidth={'ue1_enodeb': 10.0},
                queue_size={'ue1': 100},
                scheduling_algorithm={'ue1': 'INVALID_ALGO'},  # Invalid algorithm
                update_timestamp=datetime.now()
            )
        
        # Clean up
        simulator.stop_simulation()
    
    def test_simulator_state_management(self):
        """Test simulator state management and error handling."""
        simulator = create_network_simulator(use_mininet=False)
        
        # Test operations before initialization
        assert not simulator.is_simulation_running()
        
        with pytest.raises(RuntimeError):
            simulator.start_traffic_generation()
        
        with pytest.raises(RuntimeError):
            simulator.collect_kpis()
        
        with pytest.raises(RuntimeError):
            simulator.update_parameters(NetworkParameters(
                bandwidth={'test': 10.0},
                queue_size={'test': 100},
                scheduling_algorithm={'test': 'FIFO'},
                update_timestamp=datetime.now()
            ))
        
        # Initialize and test proper state
        simulator.initialize_topology()
        assert simulator.is_simulation_running()
        
        # Clean up
        simulator.stop_simulation()
        assert not simulator.is_simulation_running()
    
    def test_factory_function(self):
        """Test the factory function for creating simulators."""
        # Test mock simulator creation
        mock_sim = create_network_simulator(use_mininet=False)
        assert isinstance(mock_sim, MockNetworkSimulator)
        
        # Test mininet simulator creation (will fall back to mock if Mininet not available)
        mininet_sim = create_network_simulator(use_mininet=True)
        # Should be either MininetNetworkSimulator or MockNetworkSimulator depending on availability
        assert hasattr(mininet_sim, 'initialize_topology')
        assert hasattr(mininet_sim, 'start_traffic_generation')
        assert hasattr(mininet_sim, 'collect_kpis')
        assert hasattr(mininet_sim, 'update_parameters')
        assert hasattr(mininet_sim, 'stop_simulation')
    
    def test_enhanced_traffic_generation_and_kpi_collection(self):
        """Test enhanced OnOff traffic generation and KPI collection with timestamping and aggregation."""
        simulator = create_network_simulator(use_mininet=False)
        
        # Initialize topology and start traffic
        simulator.initialize_topology()
        simulator.start_traffic_generation()
        
        # Collect multiple KPI samples
        metrics_samples = []
        for i in range(5):
            metrics = simulator.collect_kpis()
            metrics_samples.append(metrics)
            assert isinstance(metrics, KPIMetrics)
            assert metrics.node_id in {'ue1', 'ue2', 'enodeb', 'core_router', 'server'}
            assert isinstance(metrics.timestamp, datetime)
            
            # Verify metrics are within expected ranges
            assert metrics.throughput >= 0
            assert metrics.latency >= 0
            assert 0 <= metrics.packet_loss <= 100
            assert 0 <= metrics.utilization <= 100
            
            # Small delay to ensure different timestamps
            import time
            time.sleep(0.1)
        
        # Test metric aggregation
        aggregated = simulator.get_aggregated_metrics(window_seconds=10)
        if aggregated:  # May be None if insufficient data
            assert 'throughput_mean' in aggregated
            assert 'latency_mean' in aggregated
            assert 'packet_loss_mean' in aggregated
            assert 'utilization_mean' in aggregated
            assert 'sample_count' in aggregated
            assert aggregated['sample_count'] > 0
            
            # Verify aggregated values are reasonable
            assert aggregated['throughput_mean'] >= 0
            assert aggregated['latency_mean'] >= 0
            assert 0 <= aggregated['packet_loss_mean'] <= 100
            assert 0 <= aggregated['utilization_mean'] <= 100
        
        # Verify timestamps are properly ordered
        for i in range(1, len(metrics_samples)):
            assert metrics_samples[i].timestamp >= metrics_samples[i-1].timestamp
        
        # Clean up
        simulator.stop_simulation()
    
    def test_kpi_collection_timing(self):
        """Test that KPI collection respects timing intervals."""
        simulator = create_network_simulator(use_mininet=False)
        simulator.initialize_topology()
        simulator.start_traffic_generation()
        
        # Collect first metrics
        metrics1 = simulator.collect_kpis()
        timestamp1 = metrics1.timestamp
        
        # Immediate second collection should return cached result or respect timing
        metrics2 = simulator.collect_kpis()
        
        # Either same timestamp (cached) or very close timestamps
        time_diff = abs((metrics2.timestamp - timestamp1).total_seconds())
        assert time_diff <= 1.0  # Within 1 second tolerance
        
        # Clean up
        simulator.stop_simulation()


if __name__ == '__main__':
    pytest.main([__file__])
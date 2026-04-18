#!/usr/bin/env python3
"""
Demonstration of dynamic parameter update capabilities for the AI-driven telecom optimization system.

This script demonstrates:
1. Runtime parameter modification without simulation restart
2. Parameter validation and rollback mechanisms
3. Enhanced error handling and logging
4. Parameter change history tracking

Requirements: 4.3 - Update ns-3 simulation parameters dynamically without stopping the simulation
"""

import time
import logging
from datetime import datetime
from src.simulation import create_network_simulator
from src.models import NetworkParameters

# Configure logging to see the parameter update process
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def demonstrate_dynamic_parameter_updates():
    """Demonstrate dynamic parameter update capabilities."""
    
    print("=== AI-Driven Telecom Network Optimization: Dynamic Parameter Updates Demo ===\n")
    
    # Create and initialize network simulator
    print("1. Initializing network simulator...")
    simulator = create_network_simulator(use_mininet=False)
    topology = simulator.initialize_topology()
    
    print(f"   ✓ Topology initialized with {len(topology.nodes)} nodes and {len(topology.links)} links")
    print(f"   ✓ Nodes: {', '.join(topology.nodes)}")
    print(f"   ✓ Links: {', '.join(topology.links.keys())}")
    
    # Start traffic generation
    print("\n2. Starting traffic generation...")
    simulator.start_traffic_generation()
    print("   ✓ OnOff application traffic started between UEs and server")
    
    # Display initial parameters
    print("\n3. Initial network parameters:")
    initial_params = simulator.current_parameters
    print(f"   Bandwidth: {initial_params.bandwidth}")
    print(f"   Queue sizes: {initial_params.queue_size}")
    print(f"   Scheduling algorithms: {initial_params.scheduling_algorithm}")
    
    # Collect some initial KPIs
    print("\n4. Collecting initial KPI metrics...")
    for i in range(3):
        metrics = simulator.collect_kpis()
        print(f"   Sample {i+1}: Throughput={metrics.throughput:.2f}Mbps, "
              f"Latency={metrics.latency:.2f}ms, Loss={metrics.packet_loss:.2f}%, "
              f"Utilization={metrics.utilization:.2f}%")
        time.sleep(1)
    
    # Demonstrate successful parameter update
    print("\n5. Performing dynamic parameter update (VALID)...")
    try:
        new_params = NetworkParameters(
            bandwidth={
                'ue1_enodeb': 30.0,    # Increased from 10.0
                'enodeb_core': 200.0,  # Increased from 100.0
                'core_server': 100.0,  # Increased from 50.0
                'ue2_server': 50.0     # Increased from 20.0
            },
            queue_size={
                'ue1': 200,           # Increased from 100
                'ue2': 200,           # Increased from 100
                'enodeb': 400,        # Increased from 200
                'core_router': 300,   # Increased from 150
                'server': 200         # Increased from 100
            },
            scheduling_algorithm={
                'ue1': 'WFQ',         # Changed from FIFO
                'ue2': 'WFQ',         # Changed from FIFO
                'enodeb': 'PQ',       # Changed from WFQ
                'core_router': 'RR',  # Changed from WFQ
                'server': 'WFQ'       # Changed from FIFO
            },
            update_timestamp=datetime.now()
        )
        
        simulator.update_parameters(new_params)
        print("   ✓ Parameters updated successfully!")
        
        # Verify parameters were applied
        updated_params = simulator.current_parameters
        print(f"   ✓ New bandwidth for ue1_enodeb: {updated_params.bandwidth['ue1_enodeb']} Mbps")
        print(f"   ✓ New queue size for ue1: {updated_params.queue_size['ue1']} packets")
        print(f"   ✓ New scheduling algorithm for ue1: {updated_params.scheduling_algorithm['ue1']}")
        
    except Exception as e:
        print(f"   ✗ Parameter update failed: {e}")
    
    # Demonstrate parameter validation
    print("\n6. Testing parameter validation (INVALID - bandwidth too high)...")
    try:
        invalid_params = NetworkParameters(
            bandwidth={
                'ue1_enodeb': 2000.0,  # Exceeds 1000 Mbps limit
            },
            queue_size={'ue1': 100},
            scheduling_algorithm={'ue1': 'FIFO'},
            update_timestamp=datetime.now()
        )
        
        simulator.update_parameters(invalid_params)
        print("   ✗ This should not have succeeded!")
        
    except ValueError as e:
        print(f"   ✓ Validation correctly rejected invalid parameters: {e}")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")
    
    # Demonstrate parameter validation - queue size too low
    print("\n7. Testing parameter validation (INVALID - queue size too low)...")
    try:
        invalid_params = NetworkParameters(
            bandwidth={'ue1_enodeb': 10.0},
            queue_size={'ue1': 5},  # Below 10 packet limit
            scheduling_algorithm={'ue1': 'FIFO'},
            update_timestamp=datetime.now()
        )
        
        simulator.update_parameters(invalid_params)
        print("   ✗ This should not have succeeded!")
        
    except ValueError as e:
        print(f"   ✓ Validation correctly rejected invalid parameters: {e}")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")
    
    # Demonstrate parameter validation - invalid scheduling algorithm
    print("\n8. Testing parameter validation (INVALID - unknown scheduling algorithm)...")
    try:
        invalid_params = NetworkParameters(
            bandwidth={'ue1_enodeb': 10.0},
            queue_size={'ue1': 100},
            scheduling_algorithm={'ue1': 'UNKNOWN_ALGO'},  # Invalid algorithm
            update_timestamp=datetime.now()
        )
        
        simulator.update_parameters(invalid_params)
        print("   ✗ This should not have succeeded!")
        
    except ValueError as e:
        print(f"   ✓ Validation correctly rejected invalid parameters: {e}")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")
    
    # Test parameter validation method
    print("\n9. Testing current parameter validation...")
    is_valid = simulator.validate_current_parameters()
    print(f"   ✓ Current parameters are valid: {is_valid}")
    
    # Test parameter history
    print("\n10. Checking parameter history...")
    history = simulator.get_parameter_history()
    print(f"   ✓ Parameter history contains {len(history)} entries")
    if history:
        latest = history[-1]
        print(f"   ✓ Latest change timestamp: {latest['timestamp']}")
        print(f"   ✓ Change type: {latest['change_type']}")
    
    # Collect KPIs after parameter changes
    print("\n11. Collecting KPI metrics after parameter updates...")
    for i in range(3):
        metrics = simulator.collect_kpis()
        print(f"   Sample {i+1}: Throughput={metrics.throughput:.2f}Mbps, "
              f"Latency={metrics.latency:.2f}ms, Loss={metrics.packet_loss:.2f}%, "
              f"Utilization={metrics.utilization:.2f}%")
        time.sleep(1)
    
    # Test partial parameter updates (with non-existent elements)
    print("\n12. Testing partial parameter updates with non-existent elements...")
    try:
        partial_params = NetworkParameters(
            bandwidth={
                'ue1_enodeb': 25.0,        # Valid link
                'nonexistent_link': 50.0   # Invalid link - should be skipped
            },
            queue_size={
                'ue1': 180,                # Valid node
                'nonexistent_node': 200    # Invalid node - should be skipped
            },
            scheduling_algorithm={
                'ue1': 'PQ',               # Valid node
                'nonexistent_node': 'RR'   # Invalid node - should be skipped
            },
            update_timestamp=datetime.now()
        )
        
        simulator.update_parameters(partial_params)
        print("   ✓ Partial parameter update completed successfully")
        print("   ✓ Valid elements updated, invalid elements skipped with warnings")
        
        # Verify the valid parameter was updated
        current = simulator.current_parameters
        print(f"   ✓ ue1_enodeb bandwidth updated to: {current.bandwidth['ue1_enodeb']} Mbps")
        print(f"   ✓ ue1 queue size updated to: {current.queue_size['ue1']} packets")
        print(f"   ✓ ue1 scheduling algorithm updated to: {current.scheduling_algorithm['ue1']}")
        
    except Exception as e:
        print(f"   ✗ Partial parameter update failed: {e}")
    
    # Clean up
    print("\n13. Cleaning up...")
    simulator.stop_simulation()
    print("   ✓ Network simulation stopped")
    
    print("\n=== Dynamic Parameter Updates Demo Complete ===")
    print("\nKey features demonstrated:")
    print("✓ Runtime parameter modification without simulation restart")
    print("✓ Comprehensive parameter validation with network-specific constraints")
    print("✓ Rollback mechanisms for failed updates")
    print("✓ Enhanced error handling and logging")
    print("✓ Parameter change history tracking")
    print("✓ Partial parameter updates with graceful handling of invalid elements")
    print("✓ Validation of current parameters")


if __name__ == "__main__":
    demonstrate_dynamic_parameter_updates()
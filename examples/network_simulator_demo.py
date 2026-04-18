#!/usr/bin/env python3
"""
Demonstration of NetworkSimulator functionality.

This script shows how to use the NetworkSimulator to:
1. Initialize a network topology
2. Start traffic generation
3. Collect KPI metrics
4. Update network parameters dynamically
"""

import time
import logging
from datetime import datetime
from src.simulation import create_network_simulator
from src.models import NetworkParameters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Demonstrate NetworkSimulator functionality."""
    logger.info("Starting NetworkSimulator demonstration")
    
    try:
        # Create network simulator (will use mock if Mininet not available)
        simulator = create_network_simulator(use_mininet=True)
        logger.info(f"Created simulator: {type(simulator).__name__}")
        
        # 1. Initialize network topology
        logger.info("=== Step 1: Initialize Network Topology ===")
        topology = simulator.initialize_topology()
        
        print(f"Network Topology:")
        print(f"  Nodes: {topology.nodes}")
        print(f"  Links: {topology.links}")
        print(f"  Node Types: {topology.node_types}")
        print()
        
        # 2. Start traffic generation
        logger.info("=== Step 2: Start Traffic Generation ===")
        simulator.start_traffic_generation()
        print(f"Traffic generation started: {simulator.is_traffic_running()}")
        print()
        
        # 3. Collect KPI metrics multiple times
        logger.info("=== Step 3: Collect KPI Metrics ===")
        for i in range(3):
            metrics = simulator.collect_kpis()
            print(f"KPI Metrics #{i+1}:")
            print(f"  Timestamp: {metrics.timestamp}")
            print(f"  Node ID: {metrics.node_id}")
            print(f"  Throughput: {metrics.throughput:.2f} Mbps")
            print(f"  Latency: {metrics.latency:.2f} ms")
            print(f"  Packet Loss: {metrics.packet_loss:.2f}%")
            print(f"  Utilization: {metrics.utilization:.2f}%")
            print()
            
            # Wait a bit between collections
            time.sleep(1)
        
        # 4. Update network parameters dynamically
        logger.info("=== Step 4: Update Network Parameters ===")
        
        # Show current parameters
        current_params = simulator.current_parameters
        print("Current Parameters:")
        print(f"  Bandwidth: {current_params.bandwidth}")
        print(f"  Queue Sizes: {current_params.queue_size}")
        print(f"  Scheduling: {current_params.scheduling_algorithm}")
        print()
        
        # Create updated parameters (increase bandwidth and queue sizes)
        updated_params = NetworkParameters(
            bandwidth={
                'ue1_enodeb': 25.0,    # Increased from 10.0
                'enodeb_core': 200.0,  # Increased from 100.0
                'core_server': 100.0,  # Increased from 50.0
                'ue2_server': 40.0     # Increased from 20.0
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
        
        # Apply parameter updates
        simulator.update_parameters(updated_params)
        print("Parameters updated successfully!")
        
        # Show updated parameters
        new_params = simulator.current_parameters
        print("Updated Parameters:")
        print(f"  Bandwidth: {new_params.bandwidth}")
        print(f"  Queue Sizes: {new_params.queue_size}")
        print(f"  Scheduling: {new_params.scheduling_algorithm}")
        print()
        
        # 5. Collect metrics after parameter update
        logger.info("=== Step 5: Collect Metrics After Update ===")
        updated_metrics = simulator.collect_kpis()
        print(f"Post-Update KPI Metrics:")
        print(f"  Timestamp: {updated_metrics.timestamp}")
        print(f"  Node ID: {updated_metrics.node_id}")
        print(f"  Throughput: {updated_metrics.throughput:.2f} Mbps")
        print(f"  Latency: {updated_metrics.latency:.2f} ms")
        print(f"  Packet Loss: {updated_metrics.packet_loss:.2f}%")
        print(f"  Utilization: {updated_metrics.utilization:.2f}%")
        print()
        
        logger.info("=== Demonstration Complete ===")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        raise
    
    finally:
        # Clean up
        if 'simulator' in locals():
            simulator.stop_simulation()
            logger.info("Simulator stopped and cleaned up")


if __name__ == '__main__':
    main()
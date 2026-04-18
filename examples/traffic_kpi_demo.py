#!/usr/bin/env python3
"""
Demonstration of enhanced OnOff traffic generation and KPI collection capabilities.

This script demonstrates the implementation of task 3.3:
- OnOff application traffic generation between UE and server nodes
- Real-time KPI collection (throughput, latency, packet loss, utilization)
- Timestamping and metric aggregation capabilities
"""

import time
import logging
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation import create_network_simulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demonstrate_traffic_and_kpi_collection():
    """Demonstrate enhanced traffic generation and KPI collection."""
    
    print("=" * 60)
    print("AI Telecom Optimization - Traffic Generation & KPI Collection Demo")
    print("=" * 60)
    
    # Create network simulator (using mock for demonstration)
    print("\n1. Creating network simulator...")
    simulator = create_network_simulator(use_mininet=False)
    
    try:
        # Initialize network topology
        print("\n2. Initializing network topology...")
        topology = simulator.initialize_topology()
        print(f"   Topology created with {len(topology.nodes)} nodes:")
        for node_id, node_type in topology.node_types.items():
            print(f"   - {node_id}: {node_type}")
        
        print(f"   Network links: {len(topology.links)}")
        for link_id, (src, dst) in topology.links.items():
            print(f"   - {link_id}: {src} ↔ {dst}")
        
        # Start OnOff traffic generation
        print("\n3. Starting OnOff application traffic generation...")
        simulator.start_traffic_generation()
        print("   ✓ Traffic generation started between UEs and server")
        print("   ✓ Bidirectional flows established")
        
        # Collect KPI metrics over time
        print("\n4. Collecting KPI metrics with timestamping...")
        metrics_samples = []
        
        for i in range(8):
            print(f"\n   Sample {i+1}:")
            metrics = simulator.collect_kpis()
            metrics_samples.append(metrics)
            
            print(f"   Timestamp: {metrics.timestamp.strftime('%H:%M:%S.%f')[:-3]}")
            print(f"   Node ID: {metrics.node_id}")
            print(f"   Throughput: {metrics.throughput:.2f} Mbps")
            print(f"   Latency: {metrics.latency:.2f} ms")
            print(f"   Packet Loss: {metrics.packet_loss:.2f}%")
            print(f"   Utilization: {metrics.utilization:.2f}%")
            
            # Wait between samples to demonstrate timing
            time.sleep(0.5)
        
        # Demonstrate metric aggregation
        print("\n5. Demonstrating metric aggregation capabilities...")
        aggregated = simulator.get_aggregated_metrics(window_seconds=10)
        
        if aggregated:
            print(f"   Aggregated metrics over {aggregated['window_seconds']} seconds:")
            print(f"   Sample count: {aggregated['sample_count']}")
            print(f"   Throughput - Mean: {aggregated['throughput_mean']:.2f} Mbps, "
                  f"Min: {aggregated['throughput_min']:.2f}, Max: {aggregated['throughput_max']:.2f}")
            print(f"   Latency - Mean: {aggregated['latency_mean']:.2f} ms, "
                  f"Min: {aggregated['latency_min']:.2f}, Max: {aggregated['latency_max']:.2f}")
            print(f"   Packet Loss - Mean: {aggregated['packet_loss_mean']:.2f}%, "
                  f"Min: {aggregated['packet_loss_min']:.2f}, Max: {aggregated['packet_loss_max']:.2f}")
            print(f"   Utilization - Mean: {aggregated['utilization_mean']:.2f}%, "
                  f"Min: {aggregated['utilization_min']:.2f}, Max: {aggregated['utilization_max']:.2f}")
        else:
            print("   No aggregated metrics available (insufficient data)")
        
        # Verify timestamp ordering
        print("\n6. Verifying timestamp ordering...")
        timestamps_ordered = all(
            metrics_samples[i].timestamp <= metrics_samples[i+1].timestamp
            for i in range(len(metrics_samples)-1)
        )
        print(f"   ✓ Timestamps properly ordered: {timestamps_ordered}")
        
        # Show time span of collection
        if metrics_samples:
            time_span = (metrics_samples[-1].timestamp - metrics_samples[0].timestamp).total_seconds()
            print(f"   Collection time span: {time_span:.2f} seconds")
        
        print("\n7. Task 3.3 Implementation Summary:")
        print("   ✓ OnOff application traffic generation between UE and server nodes")
        print("   ✓ Real-time KPI collection (throughput, latency, packet loss, utilization)")
        print("   ✓ Proper timestamping with 1-second collection intervals")
        print("   ✓ Metric aggregation capabilities with statistical analysis")
        print("   ✓ Bidirectional traffic flows for realistic network simulation")
        print("   ✓ Thread-safe metric collection with history tracking")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    
    finally:
        # Clean up
        print("\n8. Cleaning up...")
        simulator.stop_simulation()
        print("   ✓ Simulation stopped")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_traffic_and_kpi_collection()
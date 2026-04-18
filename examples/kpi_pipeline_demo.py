#!/usr/bin/env python3
"""
KPI Pipeline demonstration script.

This script demonstrates the core functionality of the KPIPipeline class:
- Creating sample KPI metrics
- Exporting to different formats (CSV, SQLite, Prometheus)
- Dashboard streaming capabilities
- Data retention policies
"""

import time
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.telemetry.kpi_pipeline import KPIPipeline
from src.models import KPIMetrics, ExportFormat


def create_sample_metrics(node_id: str, base_time: datetime = None) -> KPIMetrics:
    """Create sample KPI metrics for demonstration."""
    if base_time is None:
        base_time = datetime.now()
        
    return KPIMetrics(
        timestamp=base_time,
        throughput=100.0 + (hash(node_id) % 50),  # 100-150 Mbps
        latency=20.0 + (hash(node_id) % 30),      # 20-50 ms
        packet_loss=0.1 + (hash(node_id) % 10) / 100,  # 0.1-0.2%
        utilization=60.0 + (hash(node_id) % 30),  # 60-90%
        node_id=node_id
    )


def dashboard_callback(metrics: KPIMetrics):
    """Sample dashboard callback function."""
    print(f"Dashboard Update - {metrics.node_id}: "
          f"Throughput={metrics.throughput:.1f}Mbps, "
          f"Latency={metrics.latency:.1f}ms")


def main():
    """Main demonstration function."""
    print("=== KPI Pipeline Demonstration ===\n")
    
    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}\n")
        
        # Initialize KPI Pipeline
        pipeline = KPIPipeline(
            storage_dir=temp_dir,
            max_storage_mb=10,
            retention_days=7,
            dashboard_update_interval=1.0
        )
        
        # Register dashboard callback
        pipeline.register_dashboard_callback(dashboard_callback)
        
        print("1. Creating sample KPI metrics...")
        
        # Create sample metrics for different nodes
        nodes = ["UE1", "UE2", "eNodeB", "CoreRouter", "Server"]
        sample_metrics = []
        
        for i, node in enumerate(nodes):
            metrics = create_sample_metrics(
                node_id=node,
                base_time=datetime.now() + timedelta(seconds=i)
            )
            sample_metrics.append(metrics)
            print(f"   {node}: {metrics.throughput:.1f}Mbps, {metrics.latency:.1f}ms")
        
        print("\n2. Exporting metrics to different formats...")
        
        # Export to CSV
        print("   Exporting to CSV...")
        for metrics in sample_metrics:
            pipeline.export_to_storage(metrics, ExportFormat.CSV)
        
        # Export to SQLite
        print("   Exporting to SQLite...")
        for metrics in sample_metrics:
            pipeline.export_to_storage(metrics, ExportFormat.SQLITE)
        
        # Export to Prometheus
        print("   Exporting to Prometheus...")
        for metrics in sample_metrics:
            pipeline.export_to_storage(metrics, ExportFormat.PROMETHEUS)
        
        print("\n3. Demonstrating batch export...")
        
        # Create additional metrics for batch export
        batch_metrics = []
        for i in range(3):
            metrics = create_sample_metrics(
                node_id=f"BatchNode_{i}",
                base_time=datetime.now() + timedelta(seconds=10 + i)
            )
            batch_metrics.append(metrics)
        
        pipeline.export_metrics_batch(batch_metrics, ExportFormat.CSV)
        pipeline.export_metrics_batch(batch_metrics, ExportFormat.SQLITE)
        print(f"   Exported {len(batch_metrics)} metrics in batch")
        
        print("\n4. Demonstrating dashboard streaming...")
        
        # Start streaming
        pipeline.start_streaming()
        
        # Stream some metrics
        for metrics in sample_metrics[:3]:
            pipeline.stream_to_dashboard(metrics)
            time.sleep(0.5)  # Small delay to see streaming in action
        
        # Stop streaming
        pipeline.stop_streaming()
        
        print("\n5. Getting metrics summary...")
        
        summary = pipeline.get_metrics_summary()
        print(f"   Total records: {summary.get('total_records', 0)}")
        print(f"   Unique nodes: {summary.get('unique_nodes', 0)}")
        print(f"   Average throughput: {summary.get('avg_throughput', 0):.1f} Mbps")
        print(f"   Average latency: {summary.get('avg_latency', 0):.1f} ms")
        print(f"   Storage size: {summary.get('storage_size_mb', 0):.2f} MB")
        
        print("\n6. Checking exported files...")
        
        storage_path = Path(temp_dir)
        
        # Check CSV file
        csv_file = storage_path / "kpi_metrics.csv"
        if csv_file.exists():
            with open(csv_file, 'r') as f:
                lines = f.readlines()
                print(f"   CSV file: {len(lines)} lines (including header)")
        
        # Check SQLite file
        sqlite_file = storage_path / "kpi_metrics.db"
        if sqlite_file.exists():
            print(f"   SQLite file: {sqlite_file.stat().st_size} bytes")
        
        # Check Prometheus file
        prometheus_file = storage_path / "prometheus_metrics.txt"
        if prometheus_file.exists():
            print(f"   Prometheus file: {prometheus_file.stat().st_size} bytes")
        
        print("\n7. Demonstrating dashboard integration...")
        
        # Configure Grafana integration
        pipeline.configure_grafana("http://grafana.example.com:3000", "demo-api-key")
        print("   Configured Grafana integration")
        
        # Configure LibreNMS integration  
        pipeline.configure_librenms("http://librenms.example.com", "demo-token")
        print("   Configured LibreNMS integration")
        
        # Show dashboard configuration
        config = pipeline.get_dashboard_config()
        print(f"   Dashboard config: {len(config)} settings configured")
        print(f"   Grafana endpoint: {config.get('grafana_endpoint', 'Not set')}")
        print(f"   LibreNMS endpoint: {config.get('librenms_endpoint', 'Not set')}")
        
        print("\n8. Demonstrating retention policy...")
        
        # Create old metrics to test retention
        old_time = datetime.now() - timedelta(days=10)
        old_metrics = create_sample_metrics("OldNode", old_time)
        pipeline.export_to_storage(old_metrics, ExportFormat.SQLITE)
        
        print("   Added old metrics (10 days ago)")
        
        # Apply retention policy
        pipeline.apply_retention_policy()
        print("   Applied retention policy")
        
        # Check summary again
        final_summary = pipeline.get_metrics_summary()
        print(f"   Records after cleanup: {final_summary.get('total_records', 0)}")
        
        print("\n=== Demonstration Complete ===")
        print(f"\nAll files were created in: {temp_dir}")
        print("Note: Temporary directory will be cleaned up automatically.")


if __name__ == "__main__":
    main()
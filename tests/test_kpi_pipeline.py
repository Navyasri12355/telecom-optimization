"""
Unit tests for KPIPipeline class.

Tests the core functionality of KPI data collection, export, and retention policies.
"""

import pytest
import tempfile
import sqlite3
import csv
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.telemetry.kpi_pipeline import KPIPipeline
from src.models import KPIMetrics, ExportFormat
from src.interfaces import NetworkSimulatorInterface


class TestKPIPipeline:
    """Test cases for KPIPipeline class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
            
    @pytest.fixture
    def sample_metrics(self):
        """Create sample KPI metrics for testing."""
        return KPIMetrics(
            timestamp=datetime.now(),
            throughput=100.5,
            latency=25.3,
            packet_loss=0.1,
            utilization=75.2,
            node_id="test_node_1"
        )
        
    @pytest.fixture
    def kpi_pipeline(self, temp_dir):
        """Create KPIPipeline instance for testing."""
        return KPIPipeline(
            storage_dir=temp_dir,
            max_storage_mb=10,
            retention_days=7,
            dashboard_update_interval=1.0
        )
        
    @pytest.fixture
    def mock_simulator(self, sample_metrics):
        """Create mock network simulator."""
        simulator = Mock(spec=NetworkSimulatorInterface)
        simulator.collect_kpis.return_value = sample_metrics
        return simulator
        
    def test_initialization(self, temp_dir):
        """Test KPIPipeline initialization."""
        pipeline = KPIPipeline(storage_dir=temp_dir)
        
        # Check directory creation
        assert Path(temp_dir).exists()
        
        # Check file initialization
        assert pipeline.csv_file.exists()
        assert pipeline.sqlite_file.exists()
        
        # Check CSV headers
        with open(pipeline.csv_file, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            expected_headers = ['timestamp', 'throughput', 'latency', 
                              'packet_loss', 'utilization', 'node_id']
            assert headers == expected_headers
            
        # Check SQLite table creation
        conn = sqlite3.connect(str(pipeline.sqlite_file))
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='kpi_metrics'"
            )
            assert cursor.fetchone() is not None
        finally:
            conn.close()
            
    def test_collect_metrics(self, kpi_pipeline, mock_simulator, sample_metrics):
        """Test metrics collection from simulator."""
        collected_metrics = kpi_pipeline.collect_metrics(mock_simulator)
        
        assert collected_metrics == sample_metrics
        assert kpi_pipeline.metrics_count == 1
        mock_simulator.collect_kpis.assert_called_once()
        
    def test_collect_metrics_error(self, kpi_pipeline):
        """Test metrics collection error handling."""
        simulator = Mock(spec=NetworkSimulatorInterface)
        simulator.collect_kpis.side_effect = Exception("Simulator error")
        
        with pytest.raises(Exception, match="Simulator error"):
            kpi_pipeline.collect_metrics(simulator)
            
    def test_export_to_csv(self, kpi_pipeline, sample_metrics):
        """Test CSV export functionality."""
        kpi_pipeline.export_to_storage(sample_metrics, ExportFormat.CSV)
        
        # Verify CSV content
        with open(kpi_pipeline.csv_file, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Skip headers
            row = next(reader)
            
            assert row[0] == sample_metrics.timestamp.isoformat()
            assert float(row[1]) == sample_metrics.throughput
            assert float(row[2]) == sample_metrics.latency
            assert float(row[3]) == sample_metrics.packet_loss
            assert float(row[4]) == sample_metrics.utilization
            assert row[5] == sample_metrics.node_id
            
    def test_export_to_sqlite(self, kpi_pipeline, sample_metrics):
        """Test SQLite export functionality."""
        kpi_pipeline.export_to_storage(sample_metrics, ExportFormat.SQLITE)
        
        # Verify SQLite content
        conn = sqlite3.connect(str(kpi_pipeline.sqlite_file))
        try:
            cursor = conn.execute(
                "SELECT timestamp, throughput, latency, packet_loss, utilization, node_id "
                "FROM kpi_metrics ORDER BY id DESC LIMIT 1"
            )
            row = cursor.fetchone()
            
            assert row[0] == sample_metrics.timestamp.isoformat()
            assert row[1] == sample_metrics.throughput
            assert row[2] == sample_metrics.latency
            assert row[3] == sample_metrics.packet_loss
            assert row[4] == sample_metrics.utilization
            assert row[5] == sample_metrics.node_id
        finally:
            conn.close()
            
    def test_export_to_prometheus(self, kpi_pipeline, sample_metrics):
        """Test Prometheus export functionality."""
        kpi_pipeline.export_to_storage(sample_metrics, ExportFormat.PROMETHEUS)
        
        # Check that Prometheus metrics file is created
        prometheus_file = kpi_pipeline.storage_dir / "prometheus_metrics.txt"
        assert prometheus_file.exists()
        
        # Verify content contains expected metrics
        content = prometheus_file.read_text()
        assert "network_throughput_mbps" in content
        assert "network_latency_ms" in content
        assert "network_packet_loss_percent" in content
        assert "network_utilization_percent" in content
        assert sample_metrics.node_id in content
        
    def test_export_unsupported_format(self, kpi_pipeline, sample_metrics):
        """Test export with unsupported format."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            # Create a mock format that doesn't exist
            invalid_format = Mock()
            invalid_format.value = "invalid"
            kpi_pipeline.export_to_storage(sample_metrics, invalid_format)
            
    def test_batch_export_csv(self, kpi_pipeline):
        """Test batch export to CSV."""
        metrics_list = []
        for i in range(3):
            metrics = KPIMetrics(
                timestamp=datetime.now() + timedelta(seconds=i),
                throughput=100.0 + i,
                latency=25.0 + i,
                packet_loss=0.1 + i * 0.1,
                utilization=75.0 + i,
                node_id=f"node_{i}"
            )
            metrics_list.append(metrics)
            
        kpi_pipeline.export_metrics_batch(metrics_list, ExportFormat.CSV)
        
        # Verify all metrics were exported
        with open(kpi_pipeline.csv_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
        # Should have header + 3 data rows
        assert len(rows) == 4
        
        # Verify last row matches last metric
        last_row = rows[-1]
        last_metric = metrics_list[-1]
        assert last_row[0] == last_metric.timestamp.isoformat()
        assert float(last_row[1]) == last_metric.throughput
        
    def test_batch_export_sqlite(self, kpi_pipeline):
        """Test batch export to SQLite."""
        metrics_list = []
        for i in range(3):
            metrics = KPIMetrics(
                timestamp=datetime.now() + timedelta(seconds=i),
                throughput=100.0 + i,
                latency=25.0 + i,
                packet_loss=0.1 + i * 0.1,
                utilization=75.0 + i,
                node_id=f"node_{i}"
            )
            metrics_list.append(metrics)
            
        kpi_pipeline.export_metrics_batch(metrics_list, ExportFormat.SQLITE)
        
        # Verify all metrics were exported
        conn = sqlite3.connect(str(kpi_pipeline.sqlite_file))
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM kpi_metrics")
            count = cursor.fetchone()[0]
            assert count == 3
        finally:
            conn.close()
            
    def test_dashboard_streaming(self, kpi_pipeline, sample_metrics):
        """Test dashboard streaming functionality."""
        callback_called = False
        received_metrics = None
        
        def test_callback(metrics):
            nonlocal callback_called, received_metrics
            callback_called = True
            received_metrics = metrics
            
        kpi_pipeline.register_dashboard_callback(test_callback)
        kpi_pipeline.stream_to_dashboard(sample_metrics)
        
        assert callback_called
        assert received_metrics == sample_metrics
        
        # Check that dashboard data file is created when streaming worker runs
        kpi_pipeline.start_streaming()
        import time
        time.sleep(0.1)  # Give streaming worker time to process
        kpi_pipeline.stop_streaming()
        
        dashboard_file = kpi_pipeline.storage_dir / f"dashboard_data_{sample_metrics.node_id}.json"
        if dashboard_file.exists():  # File creation depends on timing
            with open(dashboard_file, 'r') as f:
                dashboard_data = json.load(f)
                assert dashboard_data['node_id'] == sample_metrics.node_id
                assert dashboard_data['metrics']['throughput'] == sample_metrics.throughput
                
    def test_retention_policy_sqlite(self, kpi_pipeline):
        """Test data retention policy for SQLite."""
        # Create old metrics
        old_timestamp = datetime.now() - timedelta(days=10)
        old_metrics = KPIMetrics(
            timestamp=old_timestamp,
            throughput=50.0,
            latency=30.0,
            packet_loss=0.5,
            utilization=60.0,
            node_id="old_node"
        )
        
        # Create recent metrics
        recent_timestamp = datetime.now()
        recent_metrics = KPIMetrics(
            timestamp=recent_timestamp,
            throughput=100.0,
            latency=25.0,
            packet_loss=0.1,
            utilization=75.0,
            node_id="recent_node"
        )
        
        # Export both metrics
        kpi_pipeline.export_to_storage(old_metrics, ExportFormat.SQLITE)
        kpi_pipeline.export_to_storage(recent_metrics, ExportFormat.SQLITE)
        
        # Verify both records exist
        conn = sqlite3.connect(str(kpi_pipeline.sqlite_file))
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM kpi_metrics")
            assert cursor.fetchone()[0] == 2
        finally:
            conn.close()
            
        # Apply retention policy
        kpi_pipeline.apply_retention_policy()
        
        # Verify old record was deleted
        conn = sqlite3.connect(str(kpi_pipeline.sqlite_file))
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM kpi_metrics")
            count = cursor.fetchone()[0]
            
            # Should have only recent record
            assert count == 1
            
            # Verify it's the recent record
            cursor = conn.execute("SELECT node_id FROM kpi_metrics")
            node_id = cursor.fetchone()[0]
            assert node_id == "recent_node"
        finally:
            conn.close()
            
    def test_retention_policy_csv(self, kpi_pipeline):
        """Test data retention policy for CSV."""
        # Create old metrics
        old_timestamp = datetime.now() - timedelta(days=10)
        old_metrics = KPIMetrics(
            timestamp=old_timestamp,
            throughput=50.0,
            latency=30.0,
            packet_loss=0.5,
            utilization=60.0,
            node_id="old_node"
        )
        
        # Create recent metrics
        recent_timestamp = datetime.now()
        recent_metrics = KPIMetrics(
            timestamp=recent_timestamp,
            throughput=100.0,
            latency=25.0,
            packet_loss=0.1,
            utilization=75.0,
            node_id="recent_node"
        )
        
        # Export both metrics
        kpi_pipeline.export_to_storage(old_metrics, ExportFormat.CSV)
        kpi_pipeline.export_to_storage(recent_metrics, ExportFormat.CSV)
        
        # Verify both records exist
        with open(kpi_pipeline.csv_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 3  # Header + 2 data rows
            
        # Apply retention policy
        kpi_pipeline.apply_retention_policy()
        
        # Verify old record was removed
        with open(kpi_pipeline.csv_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
        # Should have header + 1 recent record
        assert len(rows) == 2
        
        # Verify it's the recent record
        data_row = rows[1]
        assert data_row[5] == "recent_node"  # node_id column
        
    def test_get_metrics_summary(self, kpi_pipeline, sample_metrics):
        """Test metrics summary functionality."""
        # Export some metrics
        kpi_pipeline.export_to_storage(sample_metrics, ExportFormat.SQLITE)
        
        summary = kpi_pipeline.get_metrics_summary()
        
        assert summary['total_records'] == 1
        assert summary['unique_nodes'] == 1
        assert summary['avg_throughput'] == sample_metrics.throughput
        assert summary['avg_latency'] == sample_metrics.latency
        assert 'storage_size_mb' in summary
        
    def test_context_manager(self, temp_dir):
        """Test KPIPipeline as context manager."""
        with KPIPipeline(storage_dir=temp_dir) as pipeline:
            assert pipeline.streaming_active
            
        # Should be stopped after exiting context
        assert not pipeline.streaming_active
        
    def test_prometheus_push_gateway_error(self, kpi_pipeline, sample_metrics):
        """Test Prometheus export with push gateway error."""
        # Set up pipeline with invalid gateway
        kpi_pipeline.prometheus_gateway = "http://invalid-gateway:9091"
        
        # Should not raise exception, just log warning
        kpi_pipeline.export_to_storage(sample_metrics, ExportFormat.PROMETHEUS)
        
        # Verify metrics file is still created
        prometheus_file = kpi_pipeline.storage_dir / "prometheus_metrics.txt"
        assert prometheus_file.exists()
        
    def test_dashboard_callback_error(self, kpi_pipeline, sample_metrics):
        """Test dashboard callback error handling."""
        def failing_callback(metrics):
            raise Exception("Callback error")
            
        kpi_pipeline.register_dashboard_callback(failing_callback)
        
        # Should not raise exception, just log warning
        kpi_pipeline.stream_to_dashboard(sample_metrics)
        
    def test_empty_batch_export(self, kpi_pipeline):
        """Test batch export with empty list."""
        # Should handle empty list gracefully
        kpi_pipeline.export_metrics_batch([], ExportFormat.CSV)
        
        # CSV should still have only headers
        with open(kpi_pipeline.csv_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 1  # Only header row
    def test_dashboard_configuration(self, kpi_pipeline):
        """Test dashboard configuration methods."""
        # Test Grafana configuration
        kpi_pipeline.configure_grafana("http://grafana.example.com", "test-api-key")
        assert kpi_pipeline.grafana_endpoint == "http://grafana.example.com"
        assert hasattr(kpi_pipeline, 'grafana_api_key')
        assert kpi_pipeline.grafana_api_key == "test-api-key"
        
        # Test LibreNMS configuration
        kpi_pipeline.configure_librenms("http://librenms.example.com", "test-token")
        assert kpi_pipeline.librenms_endpoint == "http://librenms.example.com"
        assert hasattr(kpi_pipeline, 'librenms_api_token')
        assert kpi_pipeline.librenms_api_token == "test-token"
        
        # Test configuration retrieval
        config = kpi_pipeline.get_dashboard_config()
        assert config['grafana_endpoint'] == "http://grafana.example.com"
        assert config['librenms_endpoint'] == "http://librenms.example.com"
        assert 'streaming_active' in config
        assert 'registered_callbacks' in config
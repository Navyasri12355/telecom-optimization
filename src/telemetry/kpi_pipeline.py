"""
KPI Pipeline implementation for telemetry data collection and export.

This module provides the KPIPipeline class that handles:
- Real-time KPI metrics collection from network simulators
- Export to multiple formats (CSV, SQLite, Prometheus)
- Dashboard integration and streaming
- Data retention policies and storage management
"""

import csv
import sqlite3
import threading
import time
import os
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from queue import Queue, Empty
import json
import logging

from prometheus_client import CollectorRegistry, Gauge, Counter, push_to_gateway
from prometheus_client.exposition import generate_latest

from src.models import KPIMetrics, ExportFormat
from src.interfaces import KPIPipelineInterface, NetworkSimulatorInterface


class KPIPipeline(KPIPipelineInterface):
    """
    KPI Pipeline for collecting, processing, and exporting network telemetry data.
    
    Supports multiple export formats and real-time streaming to dashboards.
    Implements data retention policies to manage storage efficiently.
    """
    
    def __init__(self, 
                 storage_dir: str = "data",
                 max_storage_mb: int = 1000,
                 retention_days: int = 30,
                 dashboard_update_interval: float = 2.0,
                 prometheus_gateway: Optional[str] = None,
                 grafana_endpoint: Optional[str] = None,
                 librenms_endpoint: Optional[str] = None):
        """
        Initialize KPI Pipeline.
        
        Args:
            storage_dir: Directory for storing exported data
            max_storage_mb: Maximum storage size in MB before retention policy kicks in
            retention_days: Number of days to retain data
            dashboard_update_interval: Interval in seconds for dashboard updates
            prometheus_gateway: Prometheus pushgateway URL if using push mode
            grafana_endpoint: Grafana API endpoint for dashboard integration
            librenms_endpoint: LibreNMS API endpoint for monitoring integration
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.max_storage_bytes = max_storage_mb * 1024 * 1024
        self.retention_days = retention_days
        self.dashboard_update_interval = dashboard_update_interval
        self.prometheus_gateway = prometheus_gateway
        self.grafana_endpoint = grafana_endpoint
        self.librenms_endpoint = librenms_endpoint
        
        # Initialize storage files
        self.csv_file = self.storage_dir / "kpi_metrics.csv"
        self.sqlite_file = self.storage_dir / "kpi_metrics.db"
        
        # Initialize Prometheus metrics
        self.prometheus_registry = CollectorRegistry()
        self._init_prometheus_metrics()
        
        # Initialize SQLite database
        self._init_sqlite_db()
        
        # Initialize CSV file with headers
        self._init_csv_file()
        
        # Dashboard streaming
        self.dashboard_callbacks: List[Callable[[KPIMetrics], None]] = []
        self.streaming_queue = Queue()
        self.streaming_thread = None
        self.streaming_active = False
        
        # Metrics tracking
        self.metrics_count = 0
        self.last_retention_check = datetime.now()
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def _init_prometheus_metrics(self) -> None:
        """Initialize Prometheus metric collectors."""
        self.prometheus_metrics = {
            'throughput': Gauge('network_throughput_mbps', 
                              'Network throughput in Mbps', 
                              ['node_id'], registry=self.prometheus_registry),
            'latency': Gauge('network_latency_ms', 
                           'Network latency in milliseconds', 
                           ['node_id'], registry=self.prometheus_registry),
            'packet_loss': Gauge('network_packet_loss_percent', 
                               'Network packet loss percentage', 
                               ['node_id'], registry=self.prometheus_registry),
            'utilization': Gauge('network_utilization_percent', 
                               'Network utilization percentage', 
                               ['node_id'], registry=self.prometheus_registry),
            'metrics_exported': Counter('kpi_metrics_exported_total', 
                                      'Total number of KPI metrics exported', 
                                      ['format'], registry=self.prometheus_registry)
        }
        
    def _init_sqlite_db(self) -> None:
        """Initialize SQLite database with KPI metrics table."""
        conn = sqlite3.connect(str(self.sqlite_file))
        try:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS kpi_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    throughput REAL NOT NULL,
                    latency REAL NOT NULL,
                    packet_loss REAL NOT NULL,
                    utilization REAL NOT NULL,
                    node_id TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create index for efficient querying
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON kpi_metrics(timestamp)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_node_id 
                ON kpi_metrics(node_id)
            ''')
            
            conn.commit()
        finally:
            conn.close()
            
    def _init_csv_file(self) -> None:
        """Initialize CSV file with headers if it doesn't exist."""
        if not self.csv_file.exists():
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'throughput', 'latency', 
                    'packet_loss', 'utilization', 'node_id'
                ])
                
    def collect_metrics(self, simulator: NetworkSimulatorInterface) -> KPIMetrics:
        """
        Collect metrics from the network simulator.
        
        Args:
            simulator: Network simulator instance
            
        Returns:
            KPIMetrics: Collected metrics
        """
        try:
            metrics = simulator.collect_kpis()
            self.metrics_count += 1
            
            # Log collection
            self.logger.debug(f"Collected metrics from {metrics.node_id}: "
                            f"throughput={metrics.throughput}, latency={metrics.latency}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics from simulator: {e}")
            raise
            
    def export_to_storage(self, metrics: KPIMetrics, format: ExportFormat) -> None:
        """
        Export metrics to specified storage format.
        
        Args:
            metrics: KPI metrics to export
            format: Export format (CSV, SQLite, or Prometheus)
        """
        try:
            with self.lock:
                if format == ExportFormat.CSV:
                    self._export_to_csv(metrics)
                elif format == ExportFormat.SQLITE:
                    self._export_to_sqlite(metrics)
                elif format == ExportFormat.PROMETHEUS:
                    self._export_to_prometheus(metrics)
                else:
                    raise ValueError(f"Unsupported export format: {format}")
                    
                # Update export counter
                self.prometheus_metrics['metrics_exported'].labels(format=format.value).inc()
                
                # Check retention policy periodically
                if (datetime.now() - self.last_retention_check).seconds > 3600:  # Check hourly
                    self.apply_retention_policy()
                    self.last_retention_check = datetime.now()
                    
        except Exception as e:
            self.logger.error(f"Failed to export metrics to {format.value}: {e}")
            raise
            
    def _export_to_csv(self, metrics: KPIMetrics) -> None:
        """Export metrics to CSV file."""
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.timestamp.isoformat(),
                metrics.throughput,
                metrics.latency,
                metrics.packet_loss,
                metrics.utilization,
                metrics.node_id
            ])
            
    def _export_to_sqlite(self, metrics: KPIMetrics) -> None:
        """Export metrics to SQLite database."""
        conn = sqlite3.connect(str(self.sqlite_file))
        try:
            conn.execute('''
                INSERT INTO kpi_metrics 
                (timestamp, throughput, latency, packet_loss, utilization, node_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp.isoformat(),
                metrics.throughput,
                metrics.latency,
                metrics.packet_loss,
                metrics.utilization,
                metrics.node_id
            ))
            conn.commit()
        finally:
            conn.close()
            
    def _export_to_prometheus(self, metrics: KPIMetrics) -> None:
        """Export metrics to Prometheus format."""
        # Update Prometheus metrics
        self.prometheus_metrics['throughput'].labels(node_id=metrics.node_id).set(metrics.throughput)
        self.prometheus_metrics['latency'].labels(node_id=metrics.node_id).set(metrics.latency)
        self.prometheus_metrics['packet_loss'].labels(node_id=metrics.node_id).set(metrics.packet_loss)
        self.prometheus_metrics['utilization'].labels(node_id=metrics.node_id).set(metrics.utilization)
        
        # Push to gateway if configured
        if self.prometheus_gateway:
            try:
                push_to_gateway(self.prometheus_gateway, job='kpi_pipeline', 
                              registry=self.prometheus_registry)
            except Exception as e:
                self.logger.warning(f"Failed to push to Prometheus gateway: {e}")
                
        # Also save to file for pull-based scraping
        prometheus_file = self.storage_dir / "prometheus_metrics.txt"
        with open(prometheus_file, 'w') as f:
            f.write(generate_latest(self.prometheus_registry).decode('utf-8'))
            
    def stream_to_dashboard(self, metrics: KPIMetrics) -> None:
        """
        Stream metrics to dashboard systems.
        
        Args:
            metrics: KPI metrics to stream
        """
        try:
            # Add to streaming queue for real-time processing
            self.streaming_queue.put(metrics)
            
            # Call registered dashboard callbacks
            for callback in self.dashboard_callbacks:
                try:
                    callback(metrics)
                except Exception as e:
                    self.logger.warning(f"Dashboard callback failed: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to stream metrics to dashboard: {e}")
            
    def register_dashboard_callback(self, callback: Callable[[KPIMetrics], None]) -> None:
        """
        Register a callback function for dashboard updates.
        
        Args:
            callback: Function to call with new metrics
        """
        self.dashboard_callbacks.append(callback)
        
    def start_streaming(self) -> None:
        """Start the dashboard streaming thread."""
        if not self.streaming_active:
            self.streaming_active = True
            self.streaming_thread = threading.Thread(target=self._streaming_worker)
            self.streaming_thread.daemon = True
            self.streaming_thread.start()
            self.logger.info("Dashboard streaming started")
            
    def stop_streaming(self) -> None:
        """Stop the dashboard streaming thread."""
        self.streaming_active = False
        if self.streaming_thread:
            self.streaming_thread.join(timeout=5.0)
            self.logger.info("Dashboard streaming stopped")
            
    def _streaming_worker(self) -> None:
        """Worker thread for processing dashboard streaming."""
        while self.streaming_active:
            try:
                # Process queued metrics with timeout
                metrics = self.streaming_queue.get(timeout=self.dashboard_update_interval)
                
                # Create dashboard-compatible data format
                dashboard_data = {
                    'timestamp': metrics.timestamp.isoformat(),
                    'metrics': {
                        'throughput': metrics.throughput,
                        'latency': metrics.latency,
                        'packet_loss': metrics.packet_loss,
                        'utilization': metrics.utilization
                    },
                    'node_id': metrics.node_id
                }
                
                # Save for LibreNMS/Grafana integration
                dashboard_file = self.storage_dir / f"dashboard_data_{metrics.node_id}.json"
                with open(dashboard_file, 'w') as f:
                    json.dump(dashboard_data, f, indent=2)
                
                # Send to external dashboard systems
                self._send_to_grafana(metrics)
                self._send_to_librenms(metrics)
                    
                self.streaming_queue.task_done()
                
            except Empty:
                # No metrics to process, continue
                continue
            except Exception as e:
                self.logger.error(f"Streaming worker error: {e}")
                
    def _send_to_grafana(self, metrics: KPIMetrics) -> None:
        """
        Send metrics to Grafana using compatible data format.
        
        Args:
            metrics: KPI metrics to send to Grafana
        """
        if not self.grafana_endpoint:
            return
            
        try:
            # Create Grafana-compatible data format
            grafana_data = {
                "dashboard": "telecom-network-kpis",
                "time": int(metrics.timestamp.timestamp() * 1000),  # milliseconds
                "tags": {
                    "node_id": metrics.node_id,
                    "source": "kpi_pipeline"
                },
                "fields": {
                    "throughput": metrics.throughput,
                    "latency": metrics.latency,
                    "packet_loss": metrics.packet_loss,
                    "utilization": metrics.utilization
                }
            }
            
            # Send to Grafana API
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            response = requests.post(
                f"{self.grafana_endpoint}/api/annotations",
                json=grafana_data,
                headers=headers,
                timeout=5.0
            )
            
            if response.status_code not in [200, 201]:
                self.logger.warning(f"Grafana API returned status {response.status_code}: {response.text}")
                
        except requests.RequestException as e:
            self.logger.warning(f"Failed to send metrics to Grafana: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error sending to Grafana: {e}")
            
    def _send_to_librenms(self, metrics: KPIMetrics) -> None:
        """
        Send metrics to LibreNMS using compatible data format.
        
        Args:
            metrics: KPI metrics to send to LibreNMS
        """
        if not self.librenms_endpoint:
            return
            
        try:
            # Create LibreNMS-compatible data format
            librenms_data = {
                "device": metrics.node_id,
                "timestamp": int(metrics.timestamp.timestamp()),
                "metrics": [
                    {
                        "name": "throughput",
                        "value": metrics.throughput,
                        "unit": "Mbps",
                        "type": "gauge"
                    },
                    {
                        "name": "latency", 
                        "value": metrics.latency,
                        "unit": "ms",
                        "type": "gauge"
                    },
                    {
                        "name": "packet_loss",
                        "value": metrics.packet_loss,
                        "unit": "percent",
                        "type": "gauge"
                    },
                    {
                        "name": "utilization",
                        "value": metrics.utilization,
                        "unit": "percent", 
                        "type": "gauge"
                    }
                ]
            }
            
            # Send to LibreNMS API
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            response = requests.post(
                f"{self.librenms_endpoint}/api/v0/devices/{metrics.node_id}/metrics",
                json=librenms_data,
                headers=headers,
                timeout=5.0
            )
            
            if response.status_code not in [200, 201]:
                self.logger.warning(f"LibreNMS API returned status {response.status_code}: {response.text}")
                
        except requests.RequestException as e:
            self.logger.warning(f"Failed to send metrics to LibreNMS: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error sending to LibreNMS: {e}")
    def apply_retention_policy(self) -> None:
        """
        Apply data retention policies to prevent storage overflow.
        
        Removes old data based on retention_days and max_storage_bytes settings.
        """
        try:
            with self.lock:
                self.logger.info("Applying data retention policy")
                
                # Check storage size
                total_size = self._get_storage_size()
                
                if total_size > self.max_storage_bytes:
                    self.logger.warning(f"Storage size ({total_size / 1024 / 1024:.1f} MB) "
                                      f"exceeds limit ({self.max_storage_bytes / 1024 / 1024:.1f} MB)")
                    
                # Clean old data from SQLite
                cutoff_date = datetime.now() - timedelta(days=self.retention_days)
                conn = sqlite3.connect(str(self.sqlite_file))
                try:
                    cursor = conn.execute('''
                        DELETE FROM kpi_metrics 
                        WHERE timestamp < ?
                    ''', (cutoff_date.isoformat(),))
                    
                    deleted_rows = cursor.rowcount
                    conn.commit()
                    
                    if deleted_rows > 0:
                        self.logger.info(f"Deleted {deleted_rows} old records from SQLite")
                        
                        # Vacuum to reclaim space
                        conn.execute('VACUUM')
                finally:
                    conn.close()
                        
                # Clean old CSV data by rewriting file
                self._clean_csv_file(cutoff_date)
                
                # Clean old dashboard files
                self._clean_dashboard_files(cutoff_date)
                
                self.logger.info("Data retention policy applied successfully")
                  
        except Exception as e:
            self.logger.error(f"Failed to apply retention policy: {e}")
            
    def _get_storage_size(self) -> int:
        """Get total size of storage files in bytes."""
        total_size = 0
        for file_path in self.storage_dir.glob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
        
    def _clean_csv_file(self, cutoff_date: datetime) -> None:
        """Clean old records from CSV file."""
        if not self.csv_file.exists():
            return
            
        temp_file = self.csv_file.with_suffix('.tmp')
        records_kept = 0
        
        with open(self.csv_file, 'r') as infile, open(temp_file, 'w', newline='') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            
            # Write header
            header = next(reader, None)
            if header:
                writer.writerow(header)
                
            # Filter records by date
            for row in reader:
                try:
                    if len(row) >= 6:  # Ensure row has all columns
                        timestamp = datetime.fromisoformat(row[0])
                        if timestamp >= cutoff_date:
                            writer.writerow(row)
                            records_kept += 1
                except (ValueError, IndexError):
                    # Skip malformed rows
                    continue
                    
        # Replace original file
        temp_file.replace(self.csv_file)
        self.logger.info(f"Kept {records_kept} records in CSV file after cleanup")
        
    def _clean_dashboard_files(self, cutoff_date: datetime) -> None:
        """Clean old dashboard data files."""
        for dashboard_file in self.storage_dir.glob("dashboard_data_*.json"):
            try:
                # Check file modification time
                file_time = datetime.fromtimestamp(dashboard_file.stat().st_mtime)
                if file_time < cutoff_date:
                    dashboard_file.unlink()
                    self.logger.debug(f"Deleted old dashboard file: {dashboard_file}")
            except Exception as e:
                self.logger.warning(f"Failed to clean dashboard file {dashboard_file}: {e}")
                
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of collected metrics.
        
        Returns:
            Dict containing summary statistics
        """
        try:
            conn = sqlite3.connect(str(self.sqlite_file))
            try:
                cursor = conn.execute('''
                    SELECT 
                        COUNT(*) as total_records,
                        MIN(timestamp) as earliest_record,
                        MAX(timestamp) as latest_record,
                        AVG(throughput) as avg_throughput,
                        AVG(latency) as avg_latency,
                        AVG(packet_loss) as avg_packet_loss,
                        AVG(utilization) as avg_utilization,
                        COUNT(DISTINCT node_id) as unique_nodes
                    FROM kpi_metrics
                ''')
                
                row = cursor.fetchone()
                if row:
                    return {
                        'total_records': row[0],
                        'earliest_record': row[1],
                        'latest_record': row[2],
                        'avg_throughput': round(row[3] or 0, 2),
                        'avg_latency': round(row[4] or 0, 2),
                        'avg_packet_loss': round(row[5] or 0, 2),
                        'avg_utilization': round(row[6] or 0, 2),
                        'unique_nodes': row[7],
                        'storage_size_mb': round(self._get_storage_size() / 1024 / 1024, 2)
                    }
            finally:
                conn.close()
                    
        except Exception as e:
            self.logger.error(f"Failed to get metrics summary: {e}")
            
        return {}
        
    def export_metrics_batch(self, metrics_list: List[KPIMetrics], 
                           format: ExportFormat) -> None:
        """
        Export a batch of metrics efficiently.
        
        Args:
            metrics_list: List of KPI metrics to export
            format: Export format
        """
        if not metrics_list:
            return
            
        try:
            with self.lock:
                if format == ExportFormat.CSV:
                    self._export_batch_to_csv(metrics_list)
                elif format == ExportFormat.SQLITE:
                    self._export_batch_to_sqlite(metrics_list)
                elif format == ExportFormat.PROMETHEUS:
                    # Prometheus exports latest values, so just export the last metric
                    self._export_to_prometheus(metrics_list[-1])
                    
                # Update export counter
                self.prometheus_metrics['metrics_exported'].labels(
                    format=format.value).inc(len(metrics_list))
                    
        except Exception as e:
            self.logger.error(f"Failed to export batch to {format.value}: {e}")
            raise
            
    def _export_batch_to_csv(self, metrics_list: List[KPIMetrics]) -> None:
        """Export batch of metrics to CSV file."""
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for metrics in metrics_list:
                writer.writerow([
                    metrics.timestamp.isoformat(),
                    metrics.throughput,
                    metrics.latency,
                    metrics.packet_loss,
                    metrics.utilization,
                    metrics.node_id
                ])
                
    def _export_batch_to_sqlite(self, metrics_list: List[KPIMetrics]) -> None:
        """Export batch of metrics to SQLite database."""
        conn = sqlite3.connect(str(self.sqlite_file))
        try:
            data = [
                (
                    metrics.timestamp.isoformat(),
                    metrics.throughput,
                    metrics.latency,
                    metrics.packet_loss,
                    metrics.utilization,
                    metrics.node_id
                )
                for metrics in metrics_list
            ]
            
            conn.executemany('''
                INSERT INTO kpi_metrics 
                (timestamp, throughput, latency, packet_loss, utilization, node_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', data)
            conn.commit()
        finally:
            conn.close()
            
    def __enter__(self):
        """Context manager entry."""
        self.start_streaming()
        return self
        
    def configure_grafana(self, endpoint: str, api_key: Optional[str] = None) -> None:
        """
        Configure Grafana integration.
        
        Args:
            endpoint: Grafana API endpoint URL
            api_key: Optional API key for authentication
        """
        self.grafana_endpoint = endpoint
        if api_key:
            # Store API key for future requests
            self.grafana_api_key = api_key
            
    def configure_librenms(self, endpoint: str, api_token: Optional[str] = None) -> None:
        """
        Configure LibreNMS integration.
        
        Args:
            endpoint: LibreNMS API endpoint URL
            api_token: Optional API token for authentication
        """
        self.librenms_endpoint = endpoint
        if api_token:
            # Store API token for future requests
            self.librenms_api_token = api_token
            
    def get_dashboard_config(self) -> Dict[str, Any]:
        """
        Get current dashboard configuration.
        
        Returns:
            Dict containing dashboard configuration
        """
        return {
            'grafana_endpoint': self.grafana_endpoint,
            'librenms_endpoint': self.librenms_endpoint,
            'prometheus_gateway': self.prometheus_gateway,
            'dashboard_update_interval': self.dashboard_update_interval,
            'streaming_active': self.streaming_active,
            'registered_callbacks': len(self.dashboard_callbacks)
        }
        
    def __enter__(self):
        """Context manager entry."""
        self.start_streaming()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_streaming()
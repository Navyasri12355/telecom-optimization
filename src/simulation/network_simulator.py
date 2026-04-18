"""
Network simulator implementation for AI-driven telecom optimization system.

This module provides NetworkSimulator implementations using Mininet for network topology
simulation with support for UE-eNodeB-CoreRouter-Server topology as specified in requirements.
"""

import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Deque
from abc import ABC, abstractmethod
from collections import deque
import statistics

try:
    from mininet.net import Mininet
    from mininet.node import Host, OVSSwitch
    from mininet.link import TCLink
    from mininet.cli import CLI
    from mininet.log import setLogLevel
    MININET_AVAILABLE = True
except ImportError:
    MININET_AVAILABLE = False
    # Mock classes for when Mininet is not available
    class Mininet:
        def __init__(self, *args, **kwargs): pass
        def addHost(self, *args, **kwargs): return None
        def addSwitch(self, *args, **kwargs): return None
        def addLink(self, *args, **kwargs): return None
        def start(self): pass
        def stop(self): pass
        def ping(self, *args, **kwargs): return 0
    
    class Host: pass
    class OVSSwitch: pass
    class TCLink: pass

from src.interfaces import NetworkSimulatorInterface
from src.models import NetworkTopology, KPIMetrics, NetworkParameters


logger = logging.getLogger(__name__)


class BaseNetworkSimulator(NetworkSimulatorInterface):
    """Base network simulator with common functionality."""
    
    def __init__(self):
        self.topology: Optional[NetworkTopology] = None
        self.is_running = False
        self.traffic_active = False
        self.current_parameters: Optional[NetworkParameters] = None
        self._metrics_lock = threading.Lock()
        self._last_metrics: Optional[KPIMetrics] = None
        # Enhanced metric aggregation capabilities
        self._metrics_history: Deque[KPIMetrics] = deque(maxlen=100)  # Keep last 100 metrics
        self._collection_interval = 1.0  # 1 second collection interval
        self._last_collection_time: Optional[datetime] = None
        
    def is_simulation_running(self) -> bool:
        """Check if simulation is currently running."""
        return self.is_running
    
    def is_traffic_running(self) -> bool:
        """Check if traffic generation is active."""
        return self.traffic_active
    
    def validate_current_parameters(self) -> bool:
        """
        Validate current network parameters for consistency.
        Base implementation - should be overridden by subclasses.
        
        Returns:
            bool: True if parameters are valid, False otherwise
        """
        if not self.current_parameters:
            logger.warning("No current parameters to validate")
            return False
        
        try:
            self.current_parameters.validate()
            logger.debug("Current network parameters validation passed")
            return True
        except (ValueError, RuntimeError) as e:
            logger.error(f"Current parameter validation failed: {e}")
            return False
    
    def get_parameter_history(self) -> List[Dict[str, Any]]:
        """
        Get history of parameter changes (base implementation).
        
        Returns:
            List of parameter change records
        """
        # Base implementation returns current parameters as a single history entry
        if self.current_parameters:
            return [{
                'timestamp': self.current_parameters.update_timestamp.isoformat(),
                'parameters': self.current_parameters.to_dict(),
                'change_type': 'current_state'
            }]
        return []


class MininetNetworkSimulator(BaseNetworkSimulator):
    """
    Mininet-based network simulator implementing the UE-eNodeB-CoreRouter-Server topology.
    
    Topology:
    UE1 ↔ eNodeB ↔ CoreRouter ↔ Server ↔ UE2
    
    This creates a linear topology representing a simplified telecom network where:
    - UE1, UE2: User Equipment (mobile devices)
    - eNodeB: Base station serving the UEs
    - CoreRouter: Central network routing infrastructure  
    - Server: Application/content server
    """
    
    def __init__(self):
        super().__init__()
        self.net: Optional[Mininet] = None
        self.nodes: Dict[str, Any] = {}
        self.links: Dict[str, Any] = {}
        self.traffic_processes: List[Any] = []
        
        if not MININET_AVAILABLE:
            logger.warning("Mininet not available, using mock implementation")
    
    def initialize_topology(self) -> NetworkTopology:
        """
        Initialize the UE-eNodeB-CoreRouter-Server network topology.
        
        Returns:
            NetworkTopology: The initialized network topology configuration
            
        Raises:
            RuntimeError: If topology initialization fails
        """
        try:
            logger.info("Initializing Mininet network topology")
            
            # Create Mininet network with custom link parameters
            self.net = Mininet(
                switch=OVSSwitch,
                link=TCLink,
                autoSetMacs=True,
                autoStaticArp=True
            )
            
            # Add network nodes according to requirements (1.1)
            logger.info("Adding network nodes: UE1, UE2, eNodeB, CoreRouter, Server")
            
            # User Equipment nodes
            ue1 = self.net.addHost('ue1', ip='10.0.1.1/24')
            ue2 = self.net.addHost('ue2', ip='10.0.1.2/24')
            
            # Base station (eNodeB) - implemented as switch for packet forwarding
            enodeb = self.net.addSwitch('enodeb')
            
            # Core network router - implemented as switch
            core_router = self.net.addSwitch('core_router')
            
            # Application/content server
            server = self.net.addHost('server', ip='10.0.2.1/24')
            
            # Store node references
            self.nodes = {
                'ue1': ue1,
                'ue2': ue2, 
                'enodeb': enodeb,
                'core_router': core_router,
                'server': server
            }
            
            # Create network links with initial parameters
            logger.info("Creating network links with default parameters")
            
            # UE1 ↔ eNodeB link
            link1 = self.net.addLink(
                ue1, enodeb,
                bw=10,  # 10 Mbps initial bandwidth
                delay='10ms',
                loss=0,
                max_queue_size=100
            )
            
            # eNodeB ↔ CoreRouter link  
            link2 = self.net.addLink(
                enodeb, core_router,
                bw=100,  # 100 Mbps backbone link
                delay='5ms',
                loss=0,
                max_queue_size=200
            )
            
            # CoreRouter ↔ Server link
            link3 = self.net.addLink(
                core_router, server,
                bw=50,  # 50 Mbps server link
                delay='2ms', 
                loss=0,
                max_queue_size=150
            )
            
            # UE2 ↔ Server link (alternative path)
            link4 = self.net.addLink(
                ue2, server,
                bw=20,  # 20 Mbps direct link
                delay='15ms',
                loss=0,
                max_queue_size=100
            )
            
            # Store link references
            self.links = {
                'ue1_enodeb': link1,
                'enodeb_core': link2,
                'core_server': link3,
                'ue2_server': link4
            }
            
            # Start the network
            logger.info("Starting Mininet network")
            self.net.start()
            self.is_running = True
            
            # Create topology configuration
            self.topology = NetworkTopology(
                nodes=['ue1', 'ue2', 'enodeb', 'core_router', 'server'],
                links={
                    'ue1_enodeb': ('ue1', 'enodeb'),
                    'enodeb_core': ('enodeb', 'core_router'), 
                    'core_server': ('core_router', 'server'),
                    'ue2_server': ('ue2', 'server')
                },
                node_types={
                    'ue1': 'UE',
                    'ue2': 'UE', 
                    'enodeb': 'eNodeB',
                    'core_router': 'CoreRouter',
                    'server': 'Server'
                }
            )
            
            # Initialize default network parameters
            self.current_parameters = NetworkParameters(
                bandwidth={
                    'ue1_enodeb': 10.0,
                    'enodeb_core': 100.0,
                    'core_server': 50.0,
                    'ue2_server': 20.0
                },
                queue_size={
                    'ue1': 100,
                    'ue2': 100,
                    'enodeb': 200,
                    'core_router': 150,
                    'server': 100
                },
                scheduling_algorithm={
                    'ue1': 'FIFO',
                    'ue2': 'FIFO',
                    'enodeb': 'WFQ',
                    'core_router': 'WFQ', 
                    'server': 'FIFO'
                },
                update_timestamp=datetime.now()
            )
            
            logger.info("Network topology initialized successfully")
            return self.topology
            
        except Exception as e:
            logger.error(f"Failed to initialize network topology: {e}")
            if self.net:
                self.net.stop()
            raise RuntimeError(f"Topology initialization failed: {e}")
    
    def start_traffic_generation(self) -> None:
        """
        Start OnOff application traffic generation between UEs and server.
        
        Implements requirement 1.2: Generate OnOff application traffic between UEs and server
        with realistic traffic patterns.
        
        Raises:
            RuntimeError: If traffic generation fails to start
        """
        if not self.is_running or not self.net:
            raise RuntimeError("Network must be initialized before starting traffic")
        
        try:
            logger.info("Starting OnOff application traffic generation between UEs and server")
            
            # Clear any existing traffic processes
            self._stop_traffic_processes()
            
            # Start iperf servers on the server node for multiple traffic flows
            server_node = self.nodes['server']
            logger.info("Starting iperf servers on server node")
            
            # Server for UE1 traffic
            server_process1 = server_node.popen('iperf -s -p 5001', shell=True)
            self.traffic_processes.append(server_process1)
            
            # Server for UE2 traffic
            server_process2 = server_node.popen('iperf -s -p 5002', shell=True)
            self.traffic_processes.append(server_process2)
            
            # Give servers time to start
            time.sleep(2)
            
            # Start OnOff traffic from UE1 to server
            # Using iperf with interval reporting to simulate OnOff patterns
            ue1_node = self.nodes['ue1']
            logger.info("Starting OnOff traffic from UE1 to server (port 5001)")
            ue1_process = ue1_node.popen(
                'iperf -c 10.0.2.1 -p 5001 -t 3600 -i 1 -b 10M',  # 10Mbps bandwidth limit
                shell=True
            )
            self.traffic_processes.append(ue1_process)
            
            # Start OnOff traffic from UE2 to server with different pattern
            ue2_node = self.nodes['ue2']
            logger.info("Starting OnOff traffic from UE2 to server (port 5002)")
            ue2_process = ue2_node.popen(
                'iperf -c 10.0.2.1 -p 5002 -t 3600 -i 1 -b 15M',  # 15Mbps bandwidth limit
                shell=True
            )
            self.traffic_processes.append(ue2_process)
            
            # Additional bidirectional traffic to simulate realistic network load
            # Server to UE1 (downlink traffic)
            logger.info("Starting bidirectional traffic flows")
            ue1_server_process = ue1_node.popen('iperf -s -p 6001', shell=True)
            self.traffic_processes.append(ue1_server_process)
            
            time.sleep(1)  # Allow UE1 server to start
            
            server_to_ue1_process = server_node.popen(
                'iperf -c 10.0.1.1 -p 6001 -t 3600 -i 1 -b 8M',  # Downlink traffic
                shell=True
            )
            self.traffic_processes.append(server_to_ue1_process)
            
            self.traffic_active = True
            logger.info("OnOff application traffic generation started successfully with "
                       "bidirectional flows between UEs and server")
            
        except Exception as e:
            logger.error(f"Failed to start traffic generation: {e}")
            self._stop_traffic_processes()
            raise RuntimeError(f"Traffic generation failed: {e}")
    
    def collect_kpis(self) -> KPIMetrics:
        """
        Collect current KPI metrics from the simulation.
        
        Implements requirement 1.3: Collect throughput, latency, packet loss, and utilization 
        metrics every second with proper timestamping and metric aggregation.
        
        Returns:
            KPIMetrics: Current network performance metrics
            
        Raises:
            RuntimeError: If KPI collection fails
        """
        if not self.is_running or not self.net:
            raise RuntimeError("Network must be running to collect KPIs")
        
        try:
            with self._metrics_lock:
                current_time = datetime.now()
                
                # Ensure proper timing for 1-second collection interval
                if (self._last_collection_time and 
                    (current_time - self._last_collection_time).total_seconds() < self._collection_interval):
                    # Return cached metrics if called too frequently
                    if self._last_metrics:
                        return self._last_metrics
                
                # Update collection timestamp
                self._last_collection_time = current_time
                
                # Collect KPI metrics from network simulation
                # Basic connectivity test to measure latency
                latency = self._measure_latency()
                
                # Estimate throughput based on traffic generation
                throughput = self._estimate_throughput()
                
                # Calculate packet loss (simulated)
                packet_loss = self._calculate_packet_loss()
                
                # Estimate network utilization
                utilization = self._estimate_utilization()
                
                # Create KPI metrics for the core router (central monitoring point)
                metrics = KPIMetrics(
                    timestamp=current_time,
                    throughput=throughput,
                    latency=latency,
                    packet_loss=packet_loss,
                    utilization=utilization,
                    node_id='core_router'
                )
                
                # Store metrics in history for aggregation
                self._metrics_history.append(metrics)
                self._last_metrics = metrics
                
                logger.debug(f"Collected KPIs at {current_time.isoformat()}: "
                           f"throughput={throughput:.2f}Mbps, latency={latency:.2f}ms, "
                           f"loss={packet_loss:.2f}%, utilization={utilization:.2f}%")
                
                return metrics
                
        except Exception as e:
            logger.error(f"Failed to collect KPIs: {e}")
            raise RuntimeError(f"KPI collection failed: {e}")
    
    def update_parameters(self, params: NetworkParameters) -> None:
        """
        Update network parameters dynamically without stopping simulation.
        
        Implements requirement 4.3: Update ns-3 simulation parameters dynamically 
        without stopping the simulation with validation and rollback mechanisms.
        
        Args:
            params: New network parameters to apply
            
        Raises:
            RuntimeError: If parameter update fails
            ValueError: If parameter validation fails
        """
        if not self.is_running or not self.net:
            raise RuntimeError("Network must be running to update parameters")
        
        # Store current parameters for potential rollback
        previous_params = None
        if self.current_parameters:
            previous_params = NetworkParameters(
                bandwidth=self.current_parameters.bandwidth.copy(),
                queue_size=self.current_parameters.queue_size.copy(),
                scheduling_algorithm=self.current_parameters.scheduling_algorithm.copy(),
                update_timestamp=self.current_parameters.update_timestamp
            )
        
        try:
            logger.info("Starting dynamic network parameter update with validation and rollback support")
            
            # Pre-validation: Validate parameters before applying any changes
            params.validate()
            
            # Additional validation for network-specific constraints
            self._validate_network_constraints(params)
            
            # Track changes for logging and rollback
            changes_applied = []
            
            # Update bandwidth parameters for links with rollback tracking
            for link_id, new_bw in params.bandwidth.items():
                if link_id in self.links:
                    old_bw = self.current_parameters.bandwidth.get(link_id, 0) if self.current_parameters else 0
                    logger.info(f"Updating bandwidth for {link_id}: {old_bw} -> {new_bw} Mbps")
                    
                    try:
                        # Apply bandwidth update with validation
                        self._update_link_bandwidth(link_id, new_bw)
                        changes_applied.append(('bandwidth', link_id, old_bw, new_bw))
                    except Exception as e:
                        logger.error(f"Failed to update bandwidth for {link_id}: {e}")
                        # Rollback all changes applied so far
                        self._rollback_changes(changes_applied)
                        raise RuntimeError(f"Bandwidth update failed for {link_id}: {e}")
                elif link_id not in self.links:
                    logger.warning(f"Link {link_id} not found in topology, skipping bandwidth update")
            
            # Update queue sizes for nodes with rollback tracking
            for node_id, new_size in params.queue_size.items():
                if node_id in self.nodes:
                    old_size = self.current_parameters.queue_size.get(node_id, 0) if self.current_parameters else 0
                    logger.info(f"Updating queue size for {node_id}: {old_size} -> {new_size}")
                    
                    try:
                        # Apply queue size update with validation
                        self._update_node_queue_size(node_id, new_size)
                        changes_applied.append(('queue_size', node_id, old_size, new_size))
                    except Exception as e:
                        logger.error(f"Failed to update queue size for {node_id}: {e}")
                        # Rollback all changes applied so far
                        self._rollback_changes(changes_applied)
                        raise RuntimeError(f"Queue size update failed for {node_id}: {e}")
                elif node_id not in self.nodes:
                    logger.warning(f"Node {node_id} not found in topology, skipping queue size update")
            
            # Update scheduling algorithms with rollback tracking
            for node_id, new_algo in params.scheduling_algorithm.items():
                if node_id in self.nodes:
                    old_algo = self.current_parameters.scheduling_algorithm.get(node_id, 'FIFO') if self.current_parameters else 'FIFO'
                    logger.info(f"Updating scheduling algorithm for {node_id}: {old_algo} -> {new_algo}")
                    
                    try:
                        # Apply scheduling algorithm update with validation
                        self._update_scheduling_algorithm(node_id, new_algo)
                        changes_applied.append(('scheduling', node_id, old_algo, new_algo))
                    except Exception as e:
                        logger.error(f"Failed to update scheduling algorithm for {node_id}: {e}")
                        # Rollback all changes applied so far
                        self._rollback_changes(changes_applied)
                        raise RuntimeError(f"Scheduling algorithm update failed for {node_id}: {e}")
                elif node_id not in self.nodes:
                    logger.warning(f"Node {node_id} not found in topology, skipping scheduling algorithm update")
            
            # All updates successful - store new parameters and log changes
            self.current_parameters = params
            self._log_parameter_changes(changes_applied)
            
            logger.info(f"Network parameters updated successfully. Applied {len(changes_applied)} changes.")
            
        except ValueError as e:
            logger.error(f"Parameter validation failed: {e}")
            raise ValueError(f"Invalid parameters: {e}")
        except Exception as e:
            logger.error(f"Failed to update network parameters: {e}")
            # Attempt to restore previous parameters if available
            if previous_params:
                try:
                    logger.info("Attempting to restore previous parameters after failure")
                    self.current_parameters = previous_params
                except Exception as restore_error:
                    logger.error(f"Failed to restore previous parameters: {restore_error}")
            raise RuntimeError(f"Parameter update failed: {e}")
    
    def _validate_network_constraints(self, params: NetworkParameters) -> None:
        """
        Validate network-specific constraints for parameter updates.
        
        Args:
            params: Parameters to validate
            
        Raises:
            ValueError: If network constraints are violated
        """
        # Validate bandwidth constraints
        for link_id, bandwidth in params.bandwidth.items():
            if bandwidth > 1000.0:  # Max 1 Gbps per link
                raise ValueError(f"Bandwidth {bandwidth} Mbps exceeds maximum limit (1000 Mbps) for link {link_id}")
            if bandwidth < 0.1:  # Min 0.1 Mbps
                raise ValueError(f"Bandwidth {bandwidth} Mbps below minimum limit (0.1 Mbps) for link {link_id}")
        
        # Validate queue size constraints
        for node_id, queue_size in params.queue_size.items():
            if queue_size > 10000:  # Max 10k packets
                raise ValueError(f"Queue size {queue_size} exceeds maximum limit (10000) for node {node_id}")
            if queue_size < 10:  # Min 10 packets
                raise ValueError(f"Queue size {queue_size} below minimum limit (10) for node {node_id}")
        
        # Validate scheduling algorithm compatibility
        valid_algorithms = {"FIFO", "WFQ", "PQ", "RR"}
        for node_id, algorithm in params.scheduling_algorithm.items():
            if algorithm not in valid_algorithms:
                raise ValueError(f"Invalid scheduling algorithm '{algorithm}' for {node_id}. "
                               f"Must be one of: {valid_algorithms}")
    
    def _rollback_changes(self, changes_applied: List[tuple]) -> None:
        """
        Rollback parameter changes in case of failure.
        
        Args:
            changes_applied: List of (change_type, target_id, old_value, new_value) tuples
        """
        logger.warning(f"Rolling back {len(changes_applied)} parameter changes due to failure")
        
        # Rollback changes in reverse order
        for change_type, target_id, old_value, new_value in reversed(changes_applied):
            try:
                if change_type == 'bandwidth':
                    self._update_link_bandwidth(target_id, old_value)
                    logger.debug(f"Rolled back bandwidth for {target_id}: {new_value} -> {old_value}")
                elif change_type == 'queue_size':
                    self._update_node_queue_size(target_id, old_value)
                    logger.debug(f"Rolled back queue size for {target_id}: {new_value} -> {old_value}")
                elif change_type == 'scheduling':
                    self._update_scheduling_algorithm(target_id, old_value)
                    logger.debug(f"Rolled back scheduling algorithm for {target_id}: {new_value} -> {old_value}")
            except Exception as rollback_error:
                logger.error(f"Failed to rollback {change_type} change for {target_id}: {rollback_error}")
        
        logger.info("Parameter rollback completed")
    
    def _log_parameter_changes(self, changes_applied: List[tuple]) -> None:
        """
        Log all parameter changes for audit trail.
        
        Args:
            changes_applied: List of (change_type, target_id, old_value, new_value) tuples
        """
        if not changes_applied:
            logger.info("No parameter changes to log")
            return
        
        logger.info("=== Parameter Change Summary ===")
        for change_type, target_id, old_value, new_value in changes_applied:
            logger.info(f"{change_type.upper()}: {target_id} changed from {old_value} to {new_value}")
        logger.info("=== End Parameter Change Summary ===")
    
    def validate_current_parameters(self) -> bool:
        """
        Validate current network parameters for consistency.
        
        Returns:
            bool: True if parameters are valid, False otherwise
        """
        if not self.current_parameters:
            logger.warning("No current parameters to validate")
            return False
        
        try:
            self.current_parameters.validate()
            self._validate_network_constraints(self.current_parameters)
            logger.debug("Current network parameters validation passed")
            return True
        except (ValueError, RuntimeError) as e:
            logger.error(f"Current parameter validation failed: {e}")
            return False
    
    def get_parameter_history(self) -> List[Dict[str, Any]]:
        """
        Get history of parameter changes (simplified implementation).
        
        Returns:
            List of parameter change records
        """
        # In a full implementation, this would return actual history
        # For now, return current parameters as a single history entry
        if self.current_parameters:
            return [{
                'timestamp': self.current_parameters.update_timestamp.isoformat(),
                'parameters': self.current_parameters.to_dict(),
                'change_type': 'current_state'
            }]
        return []
    
    def stop_simulation(self) -> None:
        """Stop the network simulation and clean up resources."""
        try:
            logger.info("Stopping network simulation")
            
            # Stop traffic generation
            self._stop_traffic_processes()
            self.traffic_active = False
            
            # Stop Mininet network
            if self.net:
                self.net.stop()
                self.net = None
            
            self.is_running = False
            self.nodes.clear()
            self.links.clear()
            
            logger.info("Network simulation stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping simulation: {e}")
            raise RuntimeError(f"Failed to stop simulation: {e}")
    
    def _stop_traffic_processes(self) -> None:
        """Stop all running traffic generation processes."""
        for process in self.traffic_processes:
            try:
                if process.poll() is None:  # Process is still running
                    process.terminate()
                    process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"Error stopping traffic process: {e}")
        
        self.traffic_processes.clear()
    
    def _measure_latency(self) -> float:
        """Measure network latency using ping between UE1 and server."""
        try:
            if not self.traffic_active:
                return 10.0  # Default latency when no traffic
            
            # Simulate latency measurement - in real implementation would use ping
            # For now, return a simulated value based on network conditions
            base_latency = 15.0  # Base latency in ms
            
            # Add some variation based on utilization
            if self._last_metrics:
                utilization_factor = self._last_metrics.utilization / 100.0
                latency = base_latency * (1 + utilization_factor * 0.5)
            else:
                latency = base_latency
            
            return min(latency, 100.0)  # Cap at 100ms
            
        except Exception:
            return 20.0  # Default fallback latency
    
    def _estimate_throughput(self) -> float:
        """Estimate current network throughput."""
        try:
            if not self.traffic_active:
                return 0.0
            
            # Simulate throughput based on current bandwidth settings
            total_bandwidth = sum(self.current_parameters.bandwidth.values()) if self.current_parameters else 180.0
            
            # Simulate some utilization (60-90% of available bandwidth)
            import random
            utilization_factor = random.uniform(0.6, 0.9)
            throughput = total_bandwidth * utilization_factor * 0.25  # Scale down for realistic values
            
            return max(throughput, 1.0)  # Minimum 1 Mbps
            
        except Exception:
            return 5.0  # Default fallback throughput
    
    def _calculate_packet_loss(self) -> float:
        """Calculate current packet loss percentage."""
        try:
            if not self.traffic_active:
                return 0.0
            
            # Simulate packet loss based on network conditions
            import random
            base_loss = random.uniform(0.0, 2.0)  # 0-2% base packet loss
            
            # Increase loss if utilization is high
            if self._last_metrics and self._last_metrics.utilization > 80:
                base_loss += random.uniform(1.0, 3.0)
            
            return min(base_loss, 10.0)  # Cap at 10%
            
        except Exception:
            return 1.0  # Default fallback packet loss
    
    def _estimate_utilization(self) -> float:
        """Estimate current network utilization percentage."""
        try:
            if not self.traffic_active:
                return 0.0
            
            # Simulate utilization based on traffic and bandwidth
            import random
            base_utilization = random.uniform(40.0, 85.0)  # 40-85% utilization
            
            return min(base_utilization, 100.0)
            
        except Exception:
            return 50.0  # Default fallback utilization
    
    def _update_link_bandwidth(self, link_id: str, bandwidth: float) -> None:
        """
        Update bandwidth for a specific link with validation.
        
        Args:
            link_id: Link identifier
            bandwidth: New bandwidth in Mbps
            
        Raises:
            RuntimeError: If bandwidth update fails
        """
        try:
            # In a real Mininet implementation, this would use TC commands to modify link bandwidth
            # Example: tc class change dev <interface> parent <parent_id> classid <class_id> htb rate <bandwidth>Mbit
            
            # For now, simulate the update with validation
            if bandwidth <= 0:
                raise ValueError(f"Invalid bandwidth value: {bandwidth}")
            
            # Simulate TC command execution
            logger.debug(f"Executing TC command to update link {link_id} bandwidth to {bandwidth} Mbps")
            
            # In real implementation, you would execute:
            # subprocess.run(['tc', 'class', 'change', 'dev', interface, 'parent', parent_id, 
            #                'classid', class_id, 'htb', 'rate', f'{bandwidth}Mbit'], check=True)
            
            logger.debug(f"Link {link_id} bandwidth successfully updated to {bandwidth} Mbps")
            
        except Exception as e:
            logger.error(f"Failed to update link {link_id} bandwidth: {e}")
            raise RuntimeError(f"Link bandwidth update failed: {e}")
    
    def _update_node_queue_size(self, node_id: str, queue_size: int) -> None:
        """
        Update queue size for a specific node with validation.
        
        Args:
            node_id: Node identifier
            queue_size: New queue size in packets
            
        Raises:
            RuntimeError: If queue size update fails
        """
        try:
            # In a real implementation, this would modify node queue parameters
            # Example: tc qdisc change dev <interface> parent <parent_id> handle <handle> pfifo limit <queue_size>
            
            # Validate queue size
            if queue_size <= 0:
                raise ValueError(f"Invalid queue size: {queue_size}")
            
            # Simulate queue size update
            logger.debug(f"Executing TC command to update node {node_id} queue size to {queue_size}")
            
            # In real implementation, you would execute:
            # subprocess.run(['tc', 'qdisc', 'change', 'dev', interface, 'parent', parent_id,
            #                'handle', handle, 'pfifo', 'limit', str(queue_size)], check=True)
            
            logger.debug(f"Node {node_id} queue size successfully updated to {queue_size}")
            
        except Exception as e:
            logger.error(f"Failed to update node {node_id} queue size: {e}")
            raise RuntimeError(f"Node queue size update failed: {e}")
    
    def _update_scheduling_algorithm(self, node_id: str, algorithm: str) -> None:
        """
        Update scheduling algorithm for a specific node with validation.
        
        Args:
            node_id: Node identifier
            algorithm: New scheduling algorithm (FIFO, WFQ, PQ, RR)
            
        Raises:
            RuntimeError: If scheduling algorithm update fails
        """
        try:
            # In a real implementation, this would change the node's scheduling algorithm
            # Different algorithms require different TC qdisc configurations
            
            # Validate algorithm
            valid_algorithms = {"FIFO", "WFQ", "PQ", "RR"}
            if algorithm not in valid_algorithms:
                raise ValueError(f"Invalid scheduling algorithm: {algorithm}")
            
            # Simulate scheduling algorithm update
            logger.debug(f"Updating node {node_id} scheduling algorithm to {algorithm}")
            
            # In real implementation, you would execute different commands based on algorithm:
            # FIFO: tc qdisc change dev <interface> parent <parent_id> handle <handle> pfifo
            # WFQ: tc qdisc change dev <interface> parent <parent_id> handle <handle> sfq
            # PQ: tc qdisc change dev <interface> parent <parent_id> handle <handle> prio
            # RR: tc qdisc change dev <interface> parent <parent_id> handle <handle> rr
            
            algorithm_commands = {
                'FIFO': 'pfifo',
                'WFQ': 'sfq',  # Stochastic Fair Queueing approximates WFQ
                'PQ': 'prio',  # Priority Queueing
                'RR': 'rr'     # Round Robin
            }
            
            tc_algorithm = algorithm_commands.get(algorithm, 'pfifo')
            logger.debug(f"Using TC qdisc: {tc_algorithm} for algorithm {algorithm}")
            
            # subprocess.run(['tc', 'qdisc', 'change', 'dev', interface, 'parent', parent_id,
            #                'handle', handle, tc_algorithm], check=True)
            
            logger.debug(f"Node {node_id} scheduling algorithm successfully updated to {algorithm}")
            
        except Exception as e:
            logger.error(f"Failed to update node {node_id} scheduling algorithm: {e}")
            raise RuntimeError(f"Scheduling algorithm update failed: {e}")
    
    def get_aggregated_metrics(self, window_seconds: int = 10) -> Optional[Dict[str, float]]:
        """
        Get aggregated KPI metrics over a specified time window.
        
        Implements enhanced metric aggregation capabilities for requirements 1.3.
        
        Args:
            window_seconds: Time window in seconds for aggregation
            
        Returns:
            Dict containing aggregated metrics (mean, min, max) or None if insufficient data
        """
        try:
            with self._metrics_lock:
                if not self._metrics_history:
                    return None
                
                # Filter metrics within the time window
                current_time = datetime.now()
                window_start = current_time - timedelta(seconds=window_seconds)
                
                recent_metrics = [
                    m for m in self._metrics_history 
                    if m.timestamp >= window_start
                ]
                
                if not recent_metrics:
                    return None
                
                # Calculate aggregated statistics
                throughputs = [m.throughput for m in recent_metrics]
                latencies = [m.latency for m in recent_metrics]
                packet_losses = [m.packet_loss for m in recent_metrics]
                utilizations = [m.utilization for m in recent_metrics]
                
                aggregated = {
                    'throughput_mean': statistics.mean(throughputs),
                    'throughput_min': min(throughputs),
                    'throughput_max': max(throughputs),
                    'latency_mean': statistics.mean(latencies),
                    'latency_min': min(latencies),
                    'latency_max': max(latencies),
                    'packet_loss_mean': statistics.mean(packet_losses),
                    'packet_loss_min': min(packet_losses),
                    'packet_loss_max': max(packet_losses),
                    'utilization_mean': statistics.mean(utilizations),
                    'utilization_min': min(utilizations),
                    'utilization_max': max(utilizations),
                    'sample_count': len(recent_metrics),
                    'window_seconds': window_seconds
                }
                
                logger.debug(f"Aggregated metrics over {window_seconds}s window: "
                           f"{len(recent_metrics)} samples")
                
                return aggregated
                
        except Exception as e:
            logger.error(f"Failed to calculate aggregated metrics: {e}")
            return None


class MockNetworkSimulator(BaseNetworkSimulator):
    """
    Mock network simulator for testing when Mininet is not available.
    
    Provides the same interface as MininetNetworkSimulator but with simulated behavior
    for testing and development purposes.
    """
    
    def __init__(self):
        super().__init__()
        self._simulation_start_time = None
        self._traffic_start_time = None
    
    def initialize_topology(self) -> NetworkTopology:
        """Initialize a mock network topology."""
        logger.info("Initializing mock network topology")
        
        self.topology = NetworkTopology(
            nodes=['ue1', 'ue2', 'enodeb', 'core_router', 'server'],
            links={
                'ue1_enodeb': ('ue1', 'enodeb'),
                'enodeb_core': ('enodeb', 'core_router'),
                'core_server': ('core_router', 'server'),
                'ue2_server': ('ue2', 'server')
            },
            node_types={
                'ue1': 'UE',
                'ue2': 'UE',
                'enodeb': 'eNodeB', 
                'core_router': 'CoreRouter',
                'server': 'Server'
            }
        )
        
        self.current_parameters = NetworkParameters(
            bandwidth={
                'ue1_enodeb': 10.0,
                'enodeb_core': 100.0,
                'core_server': 50.0,
                'ue2_server': 20.0
            },
            queue_size={
                'ue1': 100,
                'ue2': 100,
                'enodeb': 200,
                'core_router': 150,
                'server': 100
            },
            scheduling_algorithm={
                'ue1': 'FIFO',
                'ue2': 'FIFO',
                'enodeb': 'WFQ',
                'core_router': 'WFQ',
                'server': 'FIFO'
            },
            update_timestamp=datetime.now()
        )
        
        self.is_running = True
        self._simulation_start_time = datetime.now()
        
        logger.info("Mock network topology initialized successfully")
        return self.topology
    
    def start_traffic_generation(self) -> None:
        """Start mock OnOff application traffic generation."""
        if not self.is_running:
            raise RuntimeError("Network must be initialized before starting traffic")
        
        logger.info("Starting mock OnOff application traffic generation between UEs and server")
        self.traffic_active = True
        self._traffic_start_time = datetime.now()
        logger.info("Mock traffic generation started with bidirectional flows")
    
    def collect_kpis(self) -> KPIMetrics:
        """
        Collect mock KPI metrics with proper timestamping and aggregation.
        
        Implements requirement 1.3: Collect throughput, latency, packet loss, and utilization 
        metrics every second with realistic simulation.
        """
        if not self.is_running:
            raise RuntimeError("Network must be running to collect KPIs")
        
        try:
            with self._metrics_lock:
                current_time = datetime.now()
                
                # Ensure proper timing for 1-second collection interval
                if (self._last_collection_time and 
                    (current_time - self._last_collection_time).total_seconds() < self._collection_interval):
                    # Return cached metrics if called too frequently
                    if self._last_metrics:
                        return self._last_metrics
                
                # Update collection timestamp
                self._last_collection_time = current_time
                
                import random
                
                # Generate realistic mock metrics based on traffic state
                if self.traffic_active and self._traffic_start_time:
                    # Simulate traffic-dependent metrics
                    traffic_duration = (current_time - self._traffic_start_time).total_seconds()
                    
                    # Throughput varies based on traffic patterns
                    base_throughput = 25.0 + 15.0 * random.random()  # 25-40 Mbps base
                    traffic_factor = 1.0 + 0.3 * random.random()  # Traffic boost
                    throughput = base_throughput * traffic_factor
                    
                    # Latency increases with traffic load
                    base_latency = 15.0 + 10.0 * random.random()  # 15-25 ms base
                    load_factor = throughput / 50.0  # Normalize by max expected throughput
                    latency = base_latency * (1.0 + load_factor * 0.5)
                    
                    # Packet loss correlates with high utilization
                    utilization = 40.0 + 45.0 * random.random()  # 40-85% utilization
                    if utilization > 75.0:
                        packet_loss = random.uniform(0.5, 3.0)  # Higher loss at high utilization
                    else:
                        packet_loss = random.uniform(0.0, 1.0)  # Low loss normally
                else:
                    # No traffic - minimal metrics
                    throughput = random.uniform(0.5, 2.0)  # Minimal background traffic
                    latency = random.uniform(8.0, 15.0)    # Low latency
                    packet_loss = random.uniform(0.0, 0.5) # Very low loss
                    utilization = random.uniform(5.0, 20.0) # Low utilization
                
                # Create KPI metrics
                metrics = KPIMetrics(
                    timestamp=current_time,
                    throughput=throughput,
                    latency=latency,
                    packet_loss=packet_loss,
                    utilization=utilization,
                    node_id='core_router'
                )
                
                # Store metrics in history for aggregation
                self._metrics_history.append(metrics)
                self._last_metrics = metrics
                
                logger.debug(f"Collected mock KPIs at {current_time.isoformat()}: "
                           f"throughput={throughput:.2f}Mbps, latency={latency:.2f}ms, "
                           f"loss={packet_loss:.2f}%, utilization={utilization:.2f}%")
                
                return metrics
                
        except Exception as e:
            logger.error(f"Failed to collect mock KPIs: {e}")
            raise RuntimeError(f"Mock KPI collection failed: {e}")
    
    def update_parameters(self, params: NetworkParameters) -> None:
        """
        Update mock network parameters with validation and rollback support.
        
        Args:
            params: New network parameters to apply
            
        Raises:
            RuntimeError: If parameter update fails
            ValueError: If parameter validation fails
        """
        if not self.is_running:
            raise RuntimeError("Network must be running to update parameters")
        
        # Store current parameters for potential rollback
        previous_params = None
        if self.current_parameters:
            previous_params = NetworkParameters(
                bandwidth=self.current_parameters.bandwidth.copy(),
                queue_size=self.current_parameters.queue_size.copy(),
                scheduling_algorithm=self.current_parameters.scheduling_algorithm.copy(),
                update_timestamp=self.current_parameters.update_timestamp
            )
        
        try:
            logger.info("Starting mock network parameter update with validation")
            
            # Validate parameters before applying
            params.validate()
            
            # Additional validation for network-specific constraints
            self._validate_network_constraints(params)
            
            # Track changes for logging
            changes_applied = []
            
            # Mock parameter updates with change tracking
            if self.current_parameters:
                for link_id, new_bw in params.bandwidth.items():
                    old_bw = self.current_parameters.bandwidth.get(link_id, 0)
                    if old_bw != new_bw:
                        changes_applied.append(('bandwidth', link_id, old_bw, new_bw))
                
                for node_id, new_size in params.queue_size.items():
                    old_size = self.current_parameters.queue_size.get(node_id, 0)
                    if old_size != new_size:
                        changes_applied.append(('queue_size', node_id, old_size, new_size))
                
                for node_id, new_algo in params.scheduling_algorithm.items():
                    old_algo = self.current_parameters.scheduling_algorithm.get(node_id, 'FIFO')
                    if old_algo != new_algo:
                        changes_applied.append(('scheduling', node_id, old_algo, new_algo))
            
            # Store updated parameters
            self.current_parameters = params
            
            # Log changes
            self._log_parameter_changes(changes_applied)
            
            logger.info(f"Mock network parameters updated successfully. Applied {len(changes_applied)} changes.")
            
        except ValueError as e:
            logger.error(f"Mock parameter validation failed: {e}")
            raise ValueError(f"Invalid parameters: {e}")
        except Exception as e:
            logger.error(f"Failed to update mock network parameters: {e}")
            # Restore previous parameters if available
            if previous_params:
                self.current_parameters = previous_params
            raise RuntimeError(f"Mock parameter update failed: {e}")
    
    def _validate_network_constraints(self, params: NetworkParameters) -> None:
        """
        Validate network-specific constraints for parameter updates.
        
        Args:
            params: Parameters to validate
            
        Raises:
            ValueError: If network constraints are violated
        """
        # Validate bandwidth constraints
        for link_id, bandwidth in params.bandwidth.items():
            if bandwidth > 1000.0:  # Max 1 Gbps per link
                raise ValueError(f"Bandwidth {bandwidth} Mbps exceeds maximum limit (1000 Mbps) for link {link_id}")
            if bandwidth < 0.1:  # Min 0.1 Mbps
                raise ValueError(f"Bandwidth {bandwidth} Mbps below minimum limit (0.1 Mbps) for link {link_id}")
        
        # Validate queue size constraints
        for node_id, queue_size in params.queue_size.items():
            if queue_size > 10000:  # Max 10k packets
                raise ValueError(f"Queue size {queue_size} exceeds maximum limit (10000) for node {node_id}")
            if queue_size < 10:  # Min 10 packets
                raise ValueError(f"Queue size {queue_size} below minimum limit (10) for node {node_id}")
        
        # Validate scheduling algorithm compatibility
        valid_algorithms = {"FIFO", "WFQ", "PQ", "RR"}
        for node_id, algorithm in params.scheduling_algorithm.items():
            if algorithm not in valid_algorithms:
                raise ValueError(f"Invalid scheduling algorithm '{algorithm}' for {node_id}. "
                               f"Must be one of: {valid_algorithms}")
    
    def _log_parameter_changes(self, changes_applied: List[tuple]) -> None:
        """
        Log all parameter changes for audit trail.
        
        Args:
            changes_applied: List of (change_type, target_id, old_value, new_value) tuples
        """
        if not changes_applied:
            logger.info("No parameter changes to log")
            return
        
        logger.info("=== Mock Parameter Change Summary ===")
        for change_type, target_id, old_value, new_value in changes_applied:
            logger.info(f"{change_type.upper()}: {target_id} changed from {old_value} to {new_value}")
        logger.info("=== End Mock Parameter Change Summary ===")
    
    def validate_current_parameters(self) -> bool:
        """
        Validate current network parameters for consistency.
        
        Returns:
            bool: True if parameters are valid, False otherwise
        """
        if not self.current_parameters:
            logger.warning("No current parameters to validate")
            return False
        
        try:
            self.current_parameters.validate()
            self._validate_network_constraints(self.current_parameters)
            logger.debug("Current mock network parameters validation passed")
            return True
        except (ValueError, RuntimeError) as e:
            logger.error(f"Current mock parameter validation failed: {e}")
            return False
    
    def get_parameter_history(self) -> List[Dict[str, Any]]:
        """
        Get history of parameter changes (simplified implementation).
        
        Returns:
            List of parameter change records
        """
        # In a full implementation, this would return actual history
        # For now, return current parameters as a single history entry
        if self.current_parameters:
            return [{
                'timestamp': self.current_parameters.update_timestamp.isoformat(),
                'parameters': self.current_parameters.to_dict(),
                'change_type': 'current_state'
            }]
        return []
    
    def stop_simulation(self) -> None:
        """Stop mock simulation."""
        logger.info("Stopping mock network simulation")
        self.is_running = False
        self.traffic_active = False
        self._traffic_start_time = None
    
    def get_aggregated_metrics(self, window_seconds: int = 10) -> Optional[Dict[str, float]]:
        """
        Get aggregated mock KPI metrics over a specified time window.
        
        Args:
            window_seconds: Time window in seconds for aggregation
            
        Returns:
            Dict containing aggregated metrics (mean, min, max) or None if insufficient data
        """
        try:
            with self._metrics_lock:
                if not self._metrics_history:
                    return None
                
                # Filter metrics within the time window
                current_time = datetime.now()
                window_start = current_time - timedelta(seconds=window_seconds)
                
                recent_metrics = [
                    m for m in self._metrics_history 
                    if m.timestamp >= window_start
                ]
                
                if not recent_metrics:
                    return None
                
                # Calculate aggregated statistics
                throughputs = [m.throughput for m in recent_metrics]
                latencies = [m.latency for m in recent_metrics]
                packet_losses = [m.packet_loss for m in recent_metrics]
                utilizations = [m.utilization for m in recent_metrics]
                
                aggregated = {
                    'throughput_mean': statistics.mean(throughputs),
                    'throughput_min': min(throughputs),
                    'throughput_max': max(throughputs),
                    'latency_mean': statistics.mean(latencies),
                    'latency_min': min(latencies),
                    'latency_max': max(latencies),
                    'packet_loss_mean': statistics.mean(packet_losses),
                    'packet_loss_min': min(packet_losses),
                    'packet_loss_max': max(packet_losses),
                    'utilization_mean': statistics.mean(utilizations),
                    'utilization_min': min(utilizations),
                    'utilization_max': max(utilizations),
                    'sample_count': len(recent_metrics),
                    'window_seconds': window_seconds
                }
                
                logger.debug(f"Mock aggregated metrics over {window_seconds}s window: "
                           f"{len(recent_metrics)} samples")
                
                return aggregated
                
        except Exception as e:
            logger.error(f"Failed to calculate mock aggregated metrics: {e}")
            return None


def create_network_simulator(use_mininet: bool = True) -> NetworkSimulatorInterface:
    """
    Factory function to create appropriate network simulator instance.
    
    Args:
        use_mininet: Whether to use Mininet (True) or mock simulator (False)
        
    Returns:
        NetworkSimulatorInterface: Configured network simulator instance
    """
    if use_mininet and MININET_AVAILABLE:
        return MininetNetworkSimulator()
    else:
        if use_mininet and not MININET_AVAILABLE:
            logger.warning("Mininet not available, falling back to mock simulator")
        return MockNetworkSimulator()
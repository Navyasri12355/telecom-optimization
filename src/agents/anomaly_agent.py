"""
Anomaly detection agent for AI-driven telecom network optimization system.

This module implements the AnomalyAgent class that uses machine learning algorithms
to detect network anomalies including packet loss spikes, latency threshold violations,
throughput drops, and node failures.
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# ML algorithm imports
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# TensorFlow for Autoencoder (optional, fallback to sklearn if not available)
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    # Note: TensorFlow not available, Autoencoder detection will use fallback method

from src.interfaces import AnomalyAgentInterface
from src.models import KPIMetrics, Anomaly, AnomalyType, SeverityLevel


class AnomalyAgent(AnomalyAgentInterface):
    """
    Anomaly detection agent using multiple ML algorithms.
    
    Implements Isolation Forest, One-Class SVM, and Autoencoder methods
    for detecting network anomalies in real-time KPI data.
    """
    
    def __init__(self, 
                 detection_algorithms: Optional[List[str]] = None,
                 detection_interval: float = 5.0,
                 historical_window: int = 100,
                 contamination_rate: float = 0.05):  # lowered from 0.1 to reduce false positives
        """
        Initialize the anomaly detection agent.
        
        Args:
            detection_algorithms: List of algorithms to use ['IsolationForest', 'OneClassSVM', 'Autoencoder']
            detection_interval: Detection interval in seconds
            historical_window: Number of historical samples to maintain
            contamination_rate: Expected proportion of anomalies in data
        """
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.detection_algorithms = detection_algorithms or ['IsolationForest', 'OneClassSVM']
        if TENSORFLOW_AVAILABLE and 'Autoencoder' not in self.detection_algorithms:
            self.detection_algorithms.append('Autoencoder')
        
        self.detection_interval = detection_interval
        self.historical_window = historical_window
        self.contamination_rate = contamination_rate
        
        # Historical data storage
        self.historical_data = deque(maxlen=self.historical_window)
        self.scaler = StandardScaler()
        
        # Detection models
        self.models = {}
        self._initialize_models()
        
        # Anomaly tracking
        self.detected_anomalies = []
        self.last_detection_time = datetime.now()
        
        # Thresholds for different anomaly types
        self.thresholds = {
            'packet_loss_spike': 5.0,   # percentage (0-100)
            'latency_threshold': 100.0, # milliseconds
            'throughput_drop': 0.3,     # 30% relative drop
            'utilization_spike': 90.0   # percentage (0-100) — was wrongly 0.9 (fraction)
        }
        
        self.logger.info(f"AnomalyAgent initialized with algorithms: {self.detection_algorithms}")
    
    def _initialize_models(self):
        """Initialize the ML models for anomaly detection."""
        try:
            if 'IsolationForest' in self.detection_algorithms:
                self.models['IsolationForest'] = IsolationForest(
                    contamination=self.contamination_rate,
                    random_state=42,
                    n_estimators=100
                )
                
            if 'OneClassSVM' in self.detection_algorithms:
                self.models['OneClassSVM'] = OneClassSVM(
                    nu=self.contamination_rate,
                    kernel='rbf',
                    gamma='scale'
                )
                
            if 'Autoencoder' in self.detection_algorithms and TENSORFLOW_AVAILABLE:
                self.models['Autoencoder'] = self._create_autoencoder()
                
            self.logger.info(f"Initialized {len(self.models)} detection models")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            raise
    
    def _create_autoencoder(self) -> Optional[keras.Model]:
        """Create and compile an autoencoder model for anomaly detection."""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        try:
            # Simple autoencoder architecture for 4 KPI features
            input_dim = 4  # throughput, latency, packet_loss, utilization
            
            model = keras.Sequential([
                keras.layers.Dense(8, activation='relu', input_shape=(input_dim,)),
                keras.layers.Dense(4, activation='relu'),
                keras.layers.Dense(2, activation='relu'),  # Bottleneck
                keras.layers.Dense(4, activation='relu'),
                keras.layers.Dense(8, activation='relu'),
                keras.layers.Dense(input_dim, activation='linear')
            ])
            
            model.compile(optimizer='adam', loss='mse')
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating autoencoder: {e}")
            return None
    
    def detect_anomalies(self, kpis: KPIMetrics) -> List[Anomaly]:
        """
        Detect anomalies in the provided KPI metrics.
        
        Args:
            kpis: Current KPI metrics to analyze
            
        Returns:
            List of detected anomalies
        """
        try:
            # Add current metrics to historical data
            self.historical_data.append(kpis)
            
            # Need sufficient historical data for detection
            if len(self.historical_data) < 10:
                return []
            
            # Prepare feature matrix
            features = self._extract_features()
            if features is None or len(features) == 0:
                return []
            
            # Detect anomalies using multiple algorithms
            anomalies = []
            
            # Rule-based detection (fast, specific thresholds)
            rule_anomalies = self._detect_rule_based_anomalies(kpis)
            anomalies.extend(rule_anomalies)
            
            # ML-based detection (require more data to avoid warmup false positives)
            if len(self.historical_data) >= 50:
                ml_anomalies = self._detect_ml_based_anomalies(features, kpis)
                anomalies.extend(ml_anomalies)
            
            # Update detection time
            self.last_detection_time = datetime.now()
            
            # Store detected anomalies
            self.detected_anomalies.extend(anomalies)
            
            if anomalies:
                self.logger.warning(f"Detected {len(anomalies)} anomalies")
                for anomaly in anomalies:
                    self.logger.warning(f"  - {anomaly.anomaly_type}: {anomaly.severity}")
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {e}")
            return []
    
    def _extract_features(self) -> Optional[np.ndarray]:
        """Extract feature matrix from historical KPI data."""
        try:
            if len(self.historical_data) < 2:
                return None
                
            features = []
            for kpi in self.historical_data:
                feature_vector = [
                    kpi.throughput,
                    kpi.latency,
                    kpi.packet_loss,
                    kpi.utilization
                ]
                features.append(feature_vector)
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return None
    
    def _detect_rule_based_anomalies(self, kpis: KPIMetrics) -> List[Anomaly]:
        """Detect anomalies using rule-based thresholds."""
        anomalies = []
        current_time = datetime.now()
        bottleneck = kpis.node_id or 'core_router'

        try:
            # Packet loss spike — queue overflow at the bottleneck node
            if kpis.packet_loss > self.thresholds['packet_loss_spike']:
                anomaly = Anomaly(
                    anomaly_type=AnomalyType.PACKET_LOSS,
                    severity=self._calculate_severity('packet_loss', kpis.packet_loss),
                    affected_nodes=[bottleneck],
                    detection_time=current_time,
                    confidence_score=min(0.9, kpis.packet_loss / 10.0)
                )
                anomalies.append(anomaly)

            # Latency spike — congestion builds up through enodeb and core path
            if kpis.latency > self.thresholds['latency_threshold']:
                # Latency affects the full path from access to core
                latency_nodes = [bottleneck]
                if bottleneck not in ('enodeb', 'core_router'):
                    latency_nodes = [bottleneck, 'enodeb']
                anomaly = Anomaly(
                    anomaly_type=AnomalyType.LATENCY_SPIKE,
                    severity=self._calculate_severity('latency', kpis.latency),
                    affected_nodes=latency_nodes,
                    detection_time=current_time,
                    confidence_score=min(0.9, kpis.latency / 200.0)
                )
                anomalies.append(anomaly)

            # Throughput drop — bottleneck node + its upstream input link
            if len(self.historical_data) >= 5:
                recent_throughput = [kpi.throughput for kpi in list(self.historical_data)[-5:]]
                avg_throughput = np.mean(recent_throughput)

                if avg_throughput > 0 and kpis.throughput < avg_throughput * (1 - self.thresholds['throughput_drop']):
                    # Throughput drop is visible at the bottleneck and the server end
                    drop_nodes = list({bottleneck, 'server'})
                    anomaly = Anomaly(
                        anomaly_type=AnomalyType.THROUGHPUT_DROP,
                        severity=self._calculate_severity('throughput_drop',
                                                         (avg_throughput - kpis.throughput) / avg_throughput),
                        affected_nodes=drop_nodes,
                        detection_time=current_time,
                        confidence_score=0.8
                    )
                    anomalies.append(anomaly)

            # Utilization spike — the saturated bottleneck node itself
            if kpis.utilization > self.thresholds['utilization_spike']:
                anomaly = Anomaly(
                    anomaly_type=AnomalyType.UTILIZATION_SPIKE,
                    severity=self._calculate_severity('utilization', kpis.utilization),
                    affected_nodes=[bottleneck],
                    detection_time=current_time,
                    confidence_score=0.85
                )
                anomalies.append(anomaly)

        except Exception as e:
            self.logger.error(f"Error in rule-based detection: {e}")


        return anomalies
    
    def _detect_ml_based_anomalies(self, features: np.ndarray, current_kpis: KPIMetrics) -> List[Anomaly]:
        """Detect anomalies using ML algorithms."""
        anomalies = []
        current_time = datetime.now()
        
        try:
            # Normalize features
            if len(features) > 1:
                features_scaled = self.scaler.fit_transform(features)
                current_features = features_scaled[-1].reshape(1, -1)
            else:
                return []
            
            # Test each ML model
            for model_name, model in self.models.items():
                try:
                    if model_name == 'Autoencoder' and TENSORFLOW_AVAILABLE:
                        anomaly_score = self._detect_with_autoencoder(model, features_scaled, current_features)
                    else:
                        # Fit model on historical data (excluding current point)
                        if len(features_scaled) > 1:
                            model.fit(features_scaled[:-1])
                            prediction = model.predict(current_features)
                            anomaly_score = 1.0 if prediction[0] == -1 else 0.0
                        else:
                            continue
                    
                    # If anomaly detected by this model
                    if anomaly_score > 0.5:
                        anomaly = Anomaly(
                            anomaly_type=AnomalyType.GENERAL_ANOMALY,
                            severity=self._calculate_ml_severity(anomaly_score),
                            affected_nodes=[current_kpis.node_id],
                            detection_time=current_time,
                            confidence_score=anomaly_score
                        )
                        anomalies.append(anomaly)
                        
                except Exception as e:
                    self.logger.warning(f"Error with {model_name} model: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error in ML-based detection: {e}")
        
        return anomalies
    
    def _detect_with_autoencoder(self, model: keras.Model, features_scaled: np.ndarray, 
                               current_features: np.ndarray) -> float:
        """Detect anomalies using autoencoder reconstruction error."""
        try:
            # Train autoencoder on historical data (excluding current point)
            if len(features_scaled) > 10:
                training_data = features_scaled[:-1]
                model.fit(training_data, training_data, epochs=50, batch_size=32, 
                         verbose=0, validation_split=0.2)
                
                # Calculate reconstruction error for current point
                reconstruction = model.predict(current_features, verbose=0)
                mse = np.mean(np.square(current_features - reconstruction))
                
                # Normalize error to 0-1 range (threshold-based)
                threshold = 0.1  # Adjust based on typical reconstruction errors
                anomaly_score = min(1.0, mse / threshold)
                
                return anomaly_score
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error in autoencoder detection: {e}")
            return 0.0
    
    def _calculate_severity(self, anomaly_type: str, value: float) -> SeverityLevel:
        """Calculate severity level based on anomaly type and value."""
        try:
            if anomaly_type == 'packet_loss':
                if value > 20.0:
                    return SeverityLevel.CRITICAL
                elif value > 10.0:
                    return SeverityLevel.HIGH
                elif value > 5.0:
                    return SeverityLevel.MEDIUM
                else:
                    return SeverityLevel.LOW
                    
            elif anomaly_type == 'latency':
                if value > 500.0:
                    return SeverityLevel.CRITICAL
                elif value > 200.0:
                    return SeverityLevel.HIGH
                elif value > 100.0:
                    return SeverityLevel.MEDIUM
                else:
                    return SeverityLevel.LOW
                    
            elif anomaly_type == 'throughput_drop':
                if value > 0.7:  # 70% drop
                    return SeverityLevel.CRITICAL
                elif value > 0.5:  # 50% drop
                    return SeverityLevel.HIGH
                elif value > 0.3:  # 30% drop
                    return SeverityLevel.MEDIUM
                else:
                    return SeverityLevel.LOW
                    
            elif anomaly_type == 'utilization':
                # value is a percentage (0-100), not a fraction
                if value > 98.0:
                    return SeverityLevel.CRITICAL
                elif value > 95.0:
                    return SeverityLevel.HIGH
                elif value > 90.0:
                    return SeverityLevel.MEDIUM
                else:
                    return SeverityLevel.LOW
                    
            else:
                return SeverityLevel.MEDIUM
                
        except Exception:
            return SeverityLevel.MEDIUM
    
    def _calculate_ml_severity(self, anomaly_score: float) -> SeverityLevel:
        """Calculate severity based on ML anomaly score."""
        if anomaly_score > 0.9:
            return SeverityLevel.CRITICAL
        elif anomaly_score > 0.7:
            return SeverityLevel.HIGH
        elif anomaly_score > 0.5:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
    
    def classify_severity(self, anomaly: Anomaly) -> SeverityLevel:
        """
        Classify the severity of a detected anomaly.
        
        Args:
            anomaly: The anomaly to classify
            
        Returns:
            Severity level of the anomaly
        """
        # Severity is already calculated during detection
        return anomaly.severity
    
    def trigger_alerts(self, anomalies: List[Anomaly]) -> None:
        """
        Trigger alerts for detected anomalies.
        
        Args:
            anomalies: List of anomalies to alert on
        """
        try:
            for anomaly in anomalies:
                alert_message = (
                    f"ANOMALY ALERT: {anomaly.anomaly_type} detected "
                    f"on nodes {anomaly.affected_nodes} "
                    f"with {anomaly.severity} severity "
                    f"(confidence: {anomaly.confidence_score:.2f})"
                )
                
                # Log based on severity
                if anomaly.severity == SeverityLevel.CRITICAL:
                    self.logger.critical(alert_message)
                elif anomaly.severity == SeverityLevel.HIGH:
                    self.logger.error(alert_message)
                elif anomaly.severity == SeverityLevel.MEDIUM:
                    self.logger.warning(alert_message)
                else:
                    self.logger.info(alert_message)
                    
        except Exception as e:
            self.logger.error(f"Error triggering alerts: {e}")
    
    def detect_node_failure(self, node_id: str, kpis: KPIMetrics) -> Optional[Anomaly]:
        """
        Detect node failure based on connectivity and performance metrics.
        
        Args:
            node_id: ID of the node to check
            kpis: Current KPI metrics
            
        Returns:
            Node failure anomaly if detected, None otherwise
        """
        try:
            # Check for signs of node failure
            failure_indicators = 0
            
            # Complete throughput loss
            if kpis.throughput == 0.0:
                failure_indicators += 1
                
            # Extreme packet loss
            if kpis.packet_loss > 90.0:
                failure_indicators += 1
                
            # Extreme latency (timeout-like behavior)
            if kpis.latency > 1000.0:
                failure_indicators += 1
                
            # If multiple indicators suggest failure
            if failure_indicators >= 2:
                anomaly = Anomaly(
                    anomaly_type=AnomalyType.NODE_FAILURE,
                    severity=SeverityLevel.CRITICAL,
                    affected_nodes=[node_id],
                    detection_time=datetime.now(),
                    confidence_score=min(0.95, failure_indicators / 3.0)
                )
                
                self.logger.critical(f"Node failure detected: {node_id}")
                return anomaly
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error in node failure detection: {e}")
            return None
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about anomaly detection performance.
        
        Returns:
            Dictionary containing detection statistics
        """
        try:
            if not self.detected_anomalies:
                return {
                    'total_anomalies': 0,
                    'anomalies_by_type': {},
                    'anomalies_by_severity': {},
                    'detection_rate': 0.0,
                    'last_detection': None
                }
            
            # Count anomalies by type
            type_counts = {}
            for anomaly in self.detected_anomalies:
                anomaly_type = anomaly.anomaly_type.value
                type_counts[anomaly_type] = type_counts.get(anomaly_type, 0) + 1
            
            # Count anomalies by severity
            severity_counts = {}
            for anomaly in self.detected_anomalies:
                severity = anomaly.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Calculate detection rate (anomalies per minute)
            if len(self.detected_anomalies) > 1:
                time_span = (self.detected_anomalies[-1].detection_time - 
                           self.detected_anomalies[0].detection_time).total_seconds()
                detection_rate = len(self.detected_anomalies) / max(time_span / 60.0, 1.0)
            else:
                detection_rate = 0.0
            
            return {
                'total_anomalies': len(self.detected_anomalies),
                'anomalies_by_type': type_counts,
                'anomalies_by_severity': severity_counts,
                'detection_rate': detection_rate,
                'last_detection': self.last_detection_time.isoformat() if self.detected_anomalies else None,
                'algorithms_used': self.detection_algorithms,
                'historical_window_size': len(self.historical_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting detection statistics: {e}")
            return {'error': str(e)}
    
    def reset_detection_history(self) -> None:
        """Reset the anomaly detection history and models."""
        try:
            self.historical_data.clear()
            self.detected_anomalies.clear()
            self._initialize_models()
            self.logger.info("Anomaly detection history reset")
            
        except Exception as e:
            self.logger.error(f"Error resetting detection history: {e}")
    
    def update_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """
        Update anomaly detection thresholds.
        
        Args:
            new_thresholds: Dictionary of threshold updates
        """
        try:
            for threshold_name, value in new_thresholds.items():
                if threshold_name in self.thresholds:
                    old_value = self.thresholds[threshold_name]
                    self.thresholds[threshold_name] = value
                    self.logger.info(f"Updated threshold {threshold_name}: {old_value} -> {value}")
                else:
                    self.logger.warning(f"Unknown threshold: {threshold_name}")
                    
        except Exception as e:
            self.logger.error(f"Error updating thresholds: {e}")
    
    def update_detection_models(self, feedback_data: List[KPIMetrics]) -> None:
        """
        Update anomaly detection models based on feedback data.
        
        Args:
            feedback_data: List of KPI metrics for model retraining
        """
        try:
            if len(feedback_data) < 10:
                self.logger.warning("Insufficient feedback data for model update")
                return
            
            # Add feedback data to historical data
            for kpi in feedback_data:
                self.historical_data.append(kpi)
            
            # Retrain models if we have sufficient data
            if len(self.historical_data) >= 50:
                self.logger.info("Retraining anomaly detection models with feedback data")
                self._initialize_models()
                
        except Exception as e:
            self.logger.error(f"Error updating detection models: {e}")
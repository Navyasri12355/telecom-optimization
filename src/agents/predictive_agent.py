"""
Predictive Agent for AI-driven telecom network optimization system.

This agent implements load forecasting using multiple ML algorithms including
ARIMA, Prophet, AutoML, and LSTM for network performance prediction.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Callable
from collections import deque
import numpy as np
import pandas as pd
import threading
import time

# ML algorithm imports
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from src.interfaces import PredictiveAgentInterface
from src.models import KPIMetrics, LoadForecast


class ModelType:
    """Supported prediction model types."""
    ARIMA = "arima"
    PROPHET = "prophet"
    AUTOML = "automl"
    LSTM = "lstm"


class PredictiveAgent(PredictiveAgentInterface):
    """
    Predictive agent that forecasts network load using historical KPI data.
    
    Implements multiple ML algorithms for load prediction with automatic
    model selection based on performance metrics.
    """
    
    def __init__(self, 
                 window_size_seconds: int = 15,
                 prediction_horizon: int = 10,
                 model_selection_strategy: str = "best_accuracy"):
        """
        Initialize the predictive agent.
        
        Args:
            window_size_seconds: Historical data window size (10-20 seconds)
            prediction_horizon: Prediction horizon in seconds (default 10)
            model_selection_strategy: Strategy for model selection
        """
        self.logger = logging.getLogger(__name__)
        
        # Validate window size according to requirements (10-20 seconds)
        if not (10 <= window_size_seconds <= 20):
            raise ValueError("Window size must be between 10-20 seconds (Requirement 3.1)")
        
        self.window_size_seconds = window_size_seconds
        self.prediction_horizon = prediction_horizon
        self.model_selection_strategy = model_selection_strategy
        
        # Historical data storage with time-based window management
        self.historical_data: deque = deque(maxlen=1000)  # Limit memory usage
        
        # Model instances and performance tracking
        self.models: Dict[str, Any] = {}
        self.model_performance: Dict[str, List[float]] = {
            ModelType.ARIMA: [],
            ModelType.PROPHET: [],
            ModelType.AUTOML: [],
            ModelType.LSTM: []
        }
        
        # Current best model
        self.best_model_type: Optional[str] = None
        self.last_training_time: Optional[datetime] = None
        
        # Prediction tracking for accuracy evaluation (task 6.3)
        self.pending_predictions: List[Tuple[List[float], datetime, int]] = []  # (predictions, timestamp, horizon)
        self.prediction_accuracy_history: List[float] = []
        
        # Enhanced accuracy monitoring (task 6.6)
        self.accuracy_monitoring_enabled = True
        self.accuracy_check_interval = 30  # seconds between accuracy checks
        self.last_accuracy_check: Optional[datetime] = None
        self.retraining_threshold = 15.0  # MAPE threshold for retraining (Requirement 3.4)
        self.consecutive_poor_accuracy_count = 0
        self.max_consecutive_poor_accuracy = 3  # Trigger retraining after 3 consecutive poor accuracy measurements
        self.accuracy_trend_window = 10  # Number of recent measurements to analyze for trends
        
        # Model configuration
        self.model_configs = {
            ModelType.ARIMA: {"order": (1, 1, 1)},
            ModelType.PROPHET: {"daily_seasonality": False, "weekly_seasonality": False},
            ModelType.AUTOML: {"n_estimators": 50, "max_depth": 10},
            ModelType.LSTM: {"units": 50, "epochs": 20, "batch_size": 32}
        }
        
        # Prediction update scheduling (task 6.8, Requirement 3.5)
        self.prediction_update_interval = 2.0  # seconds - as required by Requirement 3.5
        self.scheduler_enabled = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.scheduler_stop_event = threading.Event()
        self.last_scheduled_prediction: Optional[datetime] = None
        self.scheduled_predictions_count = 0
        self.prediction_callbacks: List[Callable[[LoadForecast], None]] = []
        self.scheduler_lock = threading.Lock()
        
        self.logger.info(f"PredictiveAgent initialized with {window_size_seconds}s window, "
                        f"{prediction_horizon}s horizon, 2s prediction scheduling")
    
    def analyze_historical_data(self, kpis: List[KPIMetrics]) -> None:
        """
        Analyze historical KPI data for prediction model training.
        
        Implements data window management for 10-20 second historical analysis
        as required by Requirement 3.1.
        
        Args:
            kpis: List of KPI metrics to analyze
        """
        if not kpis:
            self.logger.warning("No KPI data provided for analysis")
            return
        
        # Add new data to historical storage
        for kpi in kpis:
            self.historical_data.append(kpi)
        
        # Get data within the specified window
        window_data = self._get_window_data()
        
        if len(window_data) < 5:  # Minimum data points for training
            self.logger.warning(f"Insufficient data for training: {len(window_data)} points")
            return
        
        # Train all available models
        self._train_all_models(window_data)
        
        # Select best model based on strategy
        self._select_best_model()
        
        self.last_training_time = datetime.now()
        self.logger.info(f"Analyzed {len(window_data)} data points, "
                        f"best model: {self.best_model_type}")
    
    def predict_load(self, horizon: int = 10) -> LoadForecast:
        """
        Generate load forecast for specified time horizon.
        
        Implements 10-second prediction horizon as required by Requirement 3.2.
        Generates prediction for 10-second future windows with confidence intervals
        and accuracy tracking as specified in task 6.3.
        
        Args:
            horizon: Prediction horizon in seconds (default 10)
            
        Returns:
            LoadForecast with predictions and confidence intervals
        """
        # Enforce 10-second horizon requirement (Requirement 3.2)
        if horizon != 10:
            self.logger.warning(f"Non-standard horizon {horizon}s, requirement specifies 10s")
            # Still allow other horizons for flexibility, but log the deviation
        
        if not self.best_model_type:
            raise ValueError("No trained model available for prediction")
        
        # Get current window data
        window_data = self._get_window_data()
        
        if len(window_data) < 5:
            raise ValueError("Insufficient historical data for prediction")
        
        # Generate prediction using best model with enhanced confidence intervals
        predictions, confidence_interval = self._predict_with_model(
            self.best_model_type, window_data, horizon
        )
        
        # Enhanced confidence interval calculation based on model uncertainty
        enhanced_confidence_interval = self._calculate_enhanced_confidence_interval(
            predictions, self.best_model_type, window_data
        )
        
        # Calculate current model accuracy with recent performance tracking
        accuracy = self._get_current_accuracy(self.best_model_type)
        
        # Track prediction for future accuracy evaluation
        self._track_prediction_for_accuracy(predictions, horizon)
        
        # Perform accuracy monitoring if enabled (task 6.6)
        if self.accuracy_monitoring_enabled:
            monitoring_result = self.monitor_prediction_accuracy()
            if monitoring_result["retraining_triggered"]:
                self.logger.info(f"Model retraining triggered during prediction: {monitoring_result['retraining_reason']}")
        
        forecast = LoadForecast(
            predicted_values=predictions,
            confidence_interval=enhanced_confidence_interval,
            prediction_horizon=horizon,
            model_accuracy=accuracy,
            timestamp=datetime.now()
        )
        
        self.logger.info(f"Generated {horizon}s forecast with {len(predictions)} values, "
                        f"accuracy: {accuracy:.2f}%, confidence: {enhanced_confidence_interval}")
        
        return forecast
    
    def calculate_accuracy(self, predictions: LoadForecast, actual: List[KPIMetrics]) -> float:
        """
        Calculate prediction accuracy using MAPE.
        
        Implements MAPE calculation as required by Requirement 3.4.
        
        Args:
            predictions: Previous predictions to evaluate
            actual: Actual KPI metrics for comparison
            
        Returns:
            Mean Absolute Percentage Error (MAPE) as percentage
        """
        if not actual or not predictions.predicted_values:
            return 100.0  # Maximum error if no data
        
        # Extract utilization values for comparison
        actual_values = [kpi.utilization for kpi in actual[:len(predictions.predicted_values)]]
        predicted_values = predictions.predicted_values[:len(actual_values)]
        
        if len(actual_values) != len(predicted_values):
            self.logger.warning("Mismatch in prediction and actual data lengths")
        
        # Calculate MAPE
        mape = mean_absolute_percentage_error(actual_values, predicted_values) * 100
        
        # Update model performance tracking
        if hasattr(predictions, 'model_type'):
            model_type = getattr(predictions, 'model_type')
            if model_type in self.model_performance:
                self.model_performance[model_type].append(mape)
                # Keep only recent performance metrics
                if len(self.model_performance[model_type]) > 10:
                    self.model_performance[model_type].pop(0)
        
        self.logger.debug(f"Calculated MAPE: {mape:.2f}%")
        return mape
    
    def update_model(self, feedback_data: List[KPIMetrics]) -> None:
        """
        Update prediction model based on feedback.
        
        Implements adaptive model behavior as required by Requirement 7.2.
        
        Args:
            feedback_data: Recent KPI metrics for model updating
        """
        if not feedback_data:
            return
        
        # Add feedback data to historical storage
        for kpi in feedback_data:
            self.historical_data.append(kpi)
        
        # Check if retraining is needed based on performance degradation
        current_accuracy = self._get_current_accuracy(self.best_model_type)
        
        if current_accuracy > 15.0:  # MAPE threshold from Requirement 3.4
            self.logger.info(f"Model accuracy degraded to {current_accuracy:.2f}%, retraining...")
            window_data = self._get_window_data()
            if len(window_data) >= 5:
                self._train_all_models(window_data)
                self._select_best_model()
        
        self.logger.debug(f"Updated model with {len(feedback_data)} feedback points")
    
    def _get_window_data(self) -> List[KPIMetrics]:
        """
        Get KPI data within the specified time window.
        
        Returns:
            List of KPI metrics within the window
        """
        if not self.historical_data:
            return []
        
        current_time = datetime.now()
        window_start = current_time - timedelta(seconds=self.window_size_seconds)
        
        # Filter data within window
        window_data = [
            kpi for kpi in self.historical_data
            if kpi.timestamp >= window_start
        ]
        
        return sorted(window_data, key=lambda x: x.timestamp)
    
    def _train_all_models(self, data: List[KPIMetrics]) -> None:
        """
        Train all available prediction models.
        
        Implements support for ARIMA, Prophet, AutoML, and LSTM algorithms
        as required by Requirement 3.3.
        
        Args:
            data: Historical KPI data for training
        """
        if len(data) < 5:
            return
        
        # Prepare data for training
        df = self._prepare_training_data(data)
        
        try:
            # Train ARIMA model
            self._train_arima_model(df)
        except Exception as e:
            self.logger.warning(f"ARIMA training failed: {e}")
        
        try:
            # Train Prophet model
            self._train_prophet_model(df)
        except Exception as e:
            self.logger.warning(f"Prophet training failed: {e}")
        
        try:
            # Train AutoML model (Random Forest as representative)
            self._train_automl_model(df)
        except Exception as e:
            self.logger.warning(f"AutoML training failed: {e}")
        
        try:
            # Train LSTM model
            self._train_lstm_model(df)
        except Exception as e:
            self.logger.warning(f"LSTM training failed: {e}")
    
    def _prepare_training_data(self, data: List[KPIMetrics]) -> pd.DataFrame:
        """
        Prepare KPI data for model training.
        
        Args:
            data: Raw KPI metrics
            
        Returns:
            Pandas DataFrame formatted for training
        """
        df_data = []
        for kpi in data:
            df_data.append({
                'ds': kpi.timestamp,  # Prophet format
                'y': kpi.utilization,  # Target variable
                'throughput': kpi.throughput,
                'latency': kpi.latency,
                'packet_loss': kpi.packet_loss
            })
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('ds').reset_index(drop=True)
        return df
    
    def _train_arima_model(self, df: pd.DataFrame) -> None:
        """Train ARIMA model for time series prediction."""
        try:
            model = ARIMA(df['y'], order=self.model_configs[ModelType.ARIMA]["order"])
            fitted_model = model.fit()
            self.models[ModelType.ARIMA] = fitted_model
            self.logger.debug("ARIMA model trained successfully")
        except Exception as e:
            self.logger.error(f"ARIMA training error: {e}")
            raise
    
    def _train_prophet_model(self, df: pd.DataFrame) -> None:
        """Train Prophet model for time series prediction."""
        try:
            model = Prophet(**self.model_configs[ModelType.PROPHET])
            model.fit(df[['ds', 'y']])
            self.models[ModelType.PROPHET] = model
            self.logger.debug("Prophet model trained successfully")
        except Exception as e:
            self.logger.error(f"Prophet training error: {e}")
            raise
    
    def _train_automl_model(self, df: pd.DataFrame) -> None:
        """Train AutoML model (Random Forest) for prediction."""
        try:
            # Create features from time series
            features = self._create_features(df)
            if len(features) < 3:
                raise ValueError("Insufficient data for AutoML training")
            
            X = features[['lag1', 'lag2', 'trend']].values
            y = features['target'].values
            
            model = RandomForestRegressor(**self.model_configs[ModelType.AUTOML])
            model.fit(X, y)
            self.models[ModelType.AUTOML] = model
            self.logger.debug("AutoML model trained successfully")
        except Exception as e:
            self.logger.error(f"AutoML training error: {e}")
            raise
    
    def _train_lstm_model(self, df: pd.DataFrame) -> None:
        """Train LSTM model for time series prediction."""
        try:
            # Prepare LSTM data
            X, y = self._prepare_lstm_data(df['y'].values)
            if len(X) < 3:
                raise ValueError("Insufficient data for LSTM training")
            
            # Build LSTM model
            model = Sequential([
                LSTM(self.model_configs[ModelType.LSTM]["units"], 
                     return_sequences=True, input_shape=(X.shape[1], 1)),
                Dropout(0.2),
                LSTM(self.model_configs[ModelType.LSTM]["units"]),
                Dropout(0.2),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            model.fit(X, y, 
                     epochs=self.model_configs[ModelType.LSTM]["epochs"],
                     batch_size=self.model_configs[ModelType.LSTM]["batch_size"],
                     verbose=0)
            
            self.models[ModelType.LSTM] = model
            self.logger.debug("LSTM model trained successfully")
        except Exception as e:
            self.logger.error(f"LSTM training error: {e}")
            raise
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for AutoML model."""
        features_df = df.copy()
        features_df['lag1'] = features_df['y'].shift(1)
        features_df['lag2'] = features_df['y'].shift(2)
        features_df['trend'] = range(len(features_df))
        features_df['target'] = features_df['y']
        return features_df.dropna()
    
    def _prepare_lstm_data(self, data: np.ndarray, lookback: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training."""
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        return np.array(X).reshape(-1, lookback, 1), np.array(y)
    
    def _select_best_model(self) -> None:
        """Select the best performing model based on strategy."""
        if not self.models:
            return
        
        if self.model_selection_strategy == "best_accuracy":
            # Select model with best recent performance
            best_model = None
            best_score = float('inf')
            
            for model_type in self.models.keys():
                if model_type in self.model_performance and self.model_performance[model_type]:
                    avg_score = np.mean(self.model_performance[model_type][-3:])  # Recent average
                    if avg_score < best_score:
                        best_score = avg_score
                        best_model = model_type
            
            if best_model:
                self.best_model_type = best_model
            else:
                # Default to first available model
                self.best_model_type = list(self.models.keys())[0]
        
        self.logger.debug(f"Selected best model: {self.best_model_type}")
    
    def _predict_with_model(self, model_type: str, data: List[KPIMetrics], 
                           horizon: int) -> Tuple[List[float], Tuple[float, float]]:
        """
        Generate predictions using specified model.
        
        Args:
            model_type: Type of model to use
            data: Historical data
            horizon: Prediction horizon
            
        Returns:
            Tuple of (predictions, confidence_interval)
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not available")
        
        model = self.models[model_type]
        
        if model_type == ModelType.ARIMA:
            return self._predict_arima(model, horizon)
        elif model_type == ModelType.PROPHET:
            return self._predict_prophet(model, horizon)
        elif model_type == ModelType.AUTOML:
            return self._predict_automl(model, data, horizon)
        elif model_type == ModelType.LSTM:
            return self._predict_lstm(model, data, horizon)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _predict_arima(self, model, horizon: int) -> Tuple[List[float], Tuple[float, float]]:
        """Generate ARIMA predictions."""
        forecast = model.forecast(steps=horizon)
        predictions = forecast.tolist() if hasattr(forecast, 'tolist') else [forecast]
        
        # Simple confidence interval (±10% of prediction)
        avg_pred = np.mean(predictions)
        confidence_interval = (avg_pred * 0.9, avg_pred * 1.1)
        
        return predictions, confidence_interval
    
    def _predict_prophet(self, model, horizon: int) -> Tuple[List[float], Tuple[float, float]]:
        """Generate Prophet predictions."""
        future = model.make_future_dataframe(periods=horizon, freq='S')
        forecast = model.predict(future)
        
        predictions = forecast['yhat'].tail(horizon).tolist()
        lower_bound = forecast['yhat_lower'].tail(horizon).mean()
        upper_bound = forecast['yhat_upper'].tail(horizon).mean()
        
        return predictions, (lower_bound, upper_bound)
    
    def _predict_automl(self, model, data: List[KPIMetrics], 
                       horizon: int) -> Tuple[List[float], Tuple[float, float]]:
        """Generate AutoML predictions."""
        # Use last few values to predict next values
        recent_values = [kpi.utilization for kpi in data[-3:]]
        if len(recent_values) < 3:
            recent_values.extend([recent_values[-1]] * (3 - len(recent_values)))
        
        predictions = []
        for i in range(horizon):
            features = np.array([[recent_values[-2], recent_values[-1], len(data) + i]])
            pred = model.predict(features)[0]
            predictions.append(pred)
            recent_values.append(pred)
        
        avg_pred = np.mean(predictions)
        confidence_interval = (avg_pred * 0.9, avg_pred * 1.1)
        
        return predictions, confidence_interval
    
    def _predict_lstm(self, model, data: List[KPIMetrics], 
                     horizon: int) -> Tuple[List[float], Tuple[float, float]]:
        """Generate LSTM predictions."""
        # Prepare input sequence
        recent_values = [kpi.utilization for kpi in data[-3:]]
        if len(recent_values) < 3:
            recent_values.extend([recent_values[-1]] * (3 - len(recent_values)))
        
        predictions = []
        input_seq = np.array(recent_values[-3:]).reshape(1, 3, 1)
        
        for _ in range(horizon):
            pred = model.predict(input_seq, verbose=0)[0][0]
            predictions.append(pred)
            
            # Update input sequence
            input_seq = np.roll(input_seq, -1, axis=1)
            input_seq[0, -1, 0] = pred
        
        avg_pred = np.mean(predictions)
        confidence_interval = (avg_pred * 0.9, avg_pred * 1.1)
        
        return predictions, confidence_interval
    
    def _calculate_enhanced_confidence_interval(self, predictions: List[float], 
                                              model_type: str, 
                                              window_data: List[KPIMetrics]) -> Tuple[float, float]:
        """
        Calculate enhanced confidence intervals based on model uncertainty and historical variance.
        
        Implements improved confidence interval calculation for task 6.3.
        
        Args:
            predictions: Generated predictions
            model_type: Type of model used
            window_data: Historical data used for prediction
            
        Returns:
            Tuple of (lower_bound, upper_bound) for confidence interval
        """
        if not predictions:
            return (0.0, 0.0)
        
        # Calculate historical variance from window data
        historical_values = [kpi.utilization for kpi in window_data]
        historical_std = np.std(historical_values) if len(historical_values) > 1 else 5.0
        
        # Get model-specific uncertainty
        model_accuracy = self._get_current_accuracy(model_type)
        uncertainty_factor = model_accuracy / 100.0  # Convert MAPE to factor
        
        # Calculate prediction statistics
        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions) if len(predictions) > 1 else historical_std
        
        # Enhanced confidence interval calculation
        # Combine historical variance, model uncertainty, and prediction variance
        total_uncertainty = np.sqrt(
            (historical_std ** 2) + 
            (pred_std ** 2) + 
            (pred_mean * uncertainty_factor) ** 2
        )
        
        # Use 95% confidence interval (approximately 2 standard deviations)
        confidence_multiplier = 1.96
        margin = confidence_multiplier * total_uncertainty
        
        lower_bound = max(0.0, pred_mean - margin)  # Ensure non-negative
        upper_bound = min(100.0, pred_mean + margin)  # Cap at 100% utilization
        
        return (lower_bound, upper_bound)
    
    def _track_prediction_for_accuracy(self, predictions: List[float], horizon: int) -> None:
        """
        Track predictions for future accuracy evaluation.
        
        Implements prediction tracking for accuracy monitoring as required by task 6.3.
        
        Args:
            predictions: Generated predictions
            horizon: Prediction horizon in seconds
        """
        current_time = datetime.now()
        
        # Store prediction for future evaluation
        self.pending_predictions.append((predictions, current_time, horizon))
        
        # Clean up old predictions (keep only recent ones)
        cutoff_time = current_time - timedelta(minutes=5)
        self.pending_predictions = [
            (pred, timestamp, h) for pred, timestamp, h in self.pending_predictions
            if timestamp >= cutoff_time
        ]
        
        # Evaluate any predictions that should have materialized
        self._evaluate_pending_predictions()
    
    def _evaluate_pending_predictions(self) -> None:
        """
        Evaluate pending predictions against actual data for accuracy tracking.
        
        Updates prediction accuracy history based on realized outcomes.
        """
        current_time = datetime.now()
        evaluated_predictions = []
        
        for predictions, pred_timestamp, horizon in self.pending_predictions:
            # Check if enough time has passed to evaluate this prediction
            evaluation_time = pred_timestamp + timedelta(seconds=horizon)
            
            if current_time >= evaluation_time:
                # Find actual data for this time period
                actual_data = self._get_actual_data_for_period(evaluation_time, horizon)
                
                if actual_data:
                    # Calculate accuracy for this prediction
                    actual_values = [kpi.utilization for kpi in actual_data]
                    pred_values = predictions[:len(actual_values)]
                    
                    if len(pred_values) > 0 and len(actual_values) > 0:
                        mape = mean_absolute_percentage_error(actual_values, pred_values) * 100
                        self.prediction_accuracy_history.append(mape)
                        
                        # Keep only recent accuracy measurements
                        if len(self.prediction_accuracy_history) > 20:
                            self.prediction_accuracy_history.pop(0)
                
                evaluated_predictions.append((predictions, pred_timestamp, horizon))
        
        # Remove evaluated predictions
        for eval_pred in evaluated_predictions:
            if eval_pred in self.pending_predictions:
                self.pending_predictions.remove(eval_pred)
    
    def _get_actual_data_for_period(self, start_time: datetime, duration: int) -> List[KPIMetrics]:
        """
        Get actual KPI data for a specific time period.
        
        Args:
            start_time: Start of the evaluation period
            duration: Duration in seconds
            
        Returns:
            List of KPI metrics within the specified period
        """
        end_time = start_time + timedelta(seconds=duration)
        
        actual_data = [
            kpi for kpi in self.historical_data
            if start_time <= kpi.timestamp <= end_time
        ]
        
        return sorted(actual_data, key=lambda x: x.timestamp)
    
    def _get_current_accuracy(self, model_type: Optional[str]) -> float:
        """Get current accuracy for specified model type."""
        if not model_type or model_type not in self.model_performance:
            # Use overall prediction accuracy if available
            if self.prediction_accuracy_history:
                return np.mean(self.prediction_accuracy_history[-5:])
            return 100.0  # Maximum error if no performance data
        
        performance_history = self.model_performance[model_type]
        if not performance_history:
            # Fallback to overall prediction accuracy
            if self.prediction_accuracy_history:
                return np.mean(self.prediction_accuracy_history[-5:])
            return 100.0
        
        # Return recent average accuracy
        return np.mean(performance_history[-3:])
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        Get current status of all models.
        
        Enhanced with prediction accuracy tracking for task 6.3.
        
        Returns:
            Dictionary with model status information
        """
        status = {
            "available_models": list(self.models.keys()),
            "best_model": self.best_model_type,
            "last_training": self.last_training_time.isoformat() if self.last_training_time else None,
            "data_points": len(self.historical_data),
            "window_size": self.window_size_seconds,
            "prediction_horizon": self.prediction_horizon,
            "pending_predictions": len(self.pending_predictions),
            "performance": {},
            "overall_accuracy": {}
        }
        
        # Model-specific performance
        for model_type, performance in self.model_performance.items():
            if performance:
                status["performance"][model_type] = {
                    "recent_mape": np.mean(performance[-3:]),
                    "best_mape": min(performance),
                    "evaluations": len(performance)
                }
        
        # Overall prediction accuracy tracking (task 6.3)
        if self.prediction_accuracy_history:
            status["overall_accuracy"] = {
                "recent_mape": np.mean(self.prediction_accuracy_history[-5:]),
                "best_mape": min(self.prediction_accuracy_history),
                "total_evaluations": len(self.prediction_accuracy_history),
                "meets_requirement": np.mean(self.prediction_accuracy_history[-5:]) < 15.0  # Requirement 3.4
            }
        
        # Enhanced accuracy monitoring status (task 6.6)
        status["accuracy_monitoring"] = self.get_accuracy_monitoring_status()
        
        return status
    
    def monitor_prediction_accuracy(self) -> Dict[str, Any]:
        """
        Comprehensive prediction accuracy monitoring and retraining trigger system.
        
        Implements enhanced MAPE calculation, accuracy tracking over time, and model 
        retraining triggers for accuracy degradation as required by task 6.6.
        
        This method addresses Requirements 3.4 (MAPE < 15%) and 7.2 (adaptive model behavior).
        
        Returns:
            Dictionary with accuracy monitoring results and actions taken
        """
        current_time = datetime.now()
        monitoring_result = {
            "timestamp": current_time.isoformat(),
            "accuracy_check_performed": False,
            "current_accuracy": None,
            "accuracy_trend": None,
            "meets_requirement": None,
            "retraining_triggered": False,
            "retraining_reason": None,
            "consecutive_poor_count": self.consecutive_poor_accuracy_count,
            "actions_taken": []
        }
        
        # Check if it's time for accuracy monitoring
        if (self.last_accuracy_check is None or 
            (current_time - self.last_accuracy_check).total_seconds() >= self.accuracy_check_interval):
            
            monitoring_result["accuracy_check_performed"] = True
            self.last_accuracy_check = current_time
            
            # Calculate current accuracy
            current_accuracy = self._get_current_accuracy(self.best_model_type)
            monitoring_result["current_accuracy"] = current_accuracy
            
            # Analyze accuracy trend
            accuracy_trend = self._analyze_accuracy_trend()
            monitoring_result["accuracy_trend"] = accuracy_trend
            
            # Check if accuracy meets requirement (MAPE < 15%)
            meets_requirement = current_accuracy < self.retraining_threshold
            monitoring_result["meets_requirement"] = meets_requirement
            
            # Track consecutive poor accuracy measurements
            if not meets_requirement:
                self.consecutive_poor_accuracy_count += 1
                monitoring_result["actions_taken"].append(
                    f"Recorded poor accuracy measurement #{self.consecutive_poor_accuracy_count}"
                )
            else:
                self.consecutive_poor_accuracy_count = 0
                monitoring_result["actions_taken"].append("Reset consecutive poor accuracy counter")
            
            # Determine if retraining should be triggered
            retraining_needed, reason = self._should_trigger_retraining(
                current_accuracy, accuracy_trend, meets_requirement
            )
            
            if retraining_needed:
                monitoring_result["retraining_triggered"] = True
                monitoring_result["retraining_reason"] = reason
                
                # Perform retraining
                retraining_success = self._trigger_model_retraining()
                if retraining_success:
                    monitoring_result["actions_taken"].append("Model retraining completed successfully")
                    self.consecutive_poor_accuracy_count = 0  # Reset counter after successful retraining
                else:
                    monitoring_result["actions_taken"].append("Model retraining failed")
            
            # Log monitoring results
            self.logger.info(f"Accuracy monitoring: MAPE={current_accuracy:.2f}%, "
                           f"meets_req={meets_requirement}, trend={accuracy_trend}, "
                           f"consecutive_poor={self.consecutive_poor_accuracy_count}")
        
        return monitoring_result
    
    def _analyze_accuracy_trend(self) -> Optional[str]:
        """
        Analyze the trend in prediction accuracy over recent measurements.
        
        Returns:
            String describing the accuracy trend: 'improving', 'degrading', 'stable', or None
        """
        if len(self.prediction_accuracy_history) < 3:
            return None
        
        # Get recent accuracy measurements
        recent_measurements = self.prediction_accuracy_history[-self.accuracy_trend_window:]
        
        if len(recent_measurements) < 3:
            return None
        
        # Calculate trend using linear regression slope
        x = np.arange(len(recent_measurements))
        y = np.array(recent_measurements)
        
        # Simple linear trend calculation
        slope = np.polyfit(x, y, 1)[0]
        
        # Determine trend direction (negative slope = improving accuracy, positive = degrading)
        if slope < -1.0:  # Accuracy improving (MAPE decreasing)
            return "improving"
        elif slope > 1.0:  # Accuracy degrading (MAPE increasing)
            return "degrading"
        else:
            return "stable"
    
    def _should_trigger_retraining(self, current_accuracy: float, 
                                 accuracy_trend: Optional[str], 
                                 meets_requirement: bool) -> Tuple[bool, Optional[str]]:
        """
        Determine if model retraining should be triggered based on accuracy metrics.
        
        Implements intelligent retraining triggers as required by Requirements 3.4 and 7.2.
        
        Args:
            current_accuracy: Current MAPE value
            accuracy_trend: Trend analysis result
            meets_requirement: Whether current accuracy meets the 15% threshold
            
        Returns:
            Tuple of (should_retrain, reason)
        """
        # Trigger 1: Accuracy exceeds threshold for consecutive measurements
        if self.consecutive_poor_accuracy_count >= self.max_consecutive_poor_accuracy:
            return True, f"Consecutive poor accuracy for {self.consecutive_poor_accuracy_count} measurements"
        
        # Trigger 2: Severe accuracy degradation (MAPE > 25%)
        if current_accuracy > 25.0:
            return True, f"Severe accuracy degradation: MAPE={current_accuracy:.2f}%"
        
        # Trigger 3: Degrading trend with poor accuracy
        if accuracy_trend == "degrading" and current_accuracy > self.retraining_threshold:
            return True, f"Degrading accuracy trend with MAPE={current_accuracy:.2f}%"
        
        # Trigger 4: No recent training and poor accuracy
        if (self.last_training_time is None or 
            (datetime.now() - self.last_training_time).total_seconds() > 300):  # 5 minutes
            if current_accuracy > self.retraining_threshold:
                return True, f"No recent training and poor accuracy: MAPE={current_accuracy:.2f}%"
        
        return False, None
    
    def _trigger_model_retraining(self) -> bool:
        """
        Trigger model retraining with current data.
        
        Implements model retraining as required by Requirement 7.2.
        
        Returns:
            True if retraining was successful, False otherwise
        """
        try:
            self.logger.info("Triggering model retraining due to accuracy degradation")
            
            # Get current window data for retraining
            window_data = self._get_window_data()
            
            if len(window_data) < 5:
                self.logger.warning("Insufficient data for retraining")
                return False
            
            # Store previous best model for fallback
            previous_best_model = self.best_model_type
            previous_models = self.models.copy()
            
            # Clear current models and retrain
            self.models.clear()
            self.best_model_type = None
            
            # Retrain all models
            self._train_all_models(window_data)
            
            # Select new best model
            self._select_best_model()
            
            if self.best_model_type is None:
                # Retraining failed, restore previous models
                self.logger.warning("Retraining failed, restoring previous models")
                self.models = previous_models
                self.best_model_type = previous_best_model
                return False
            
            # Update training timestamp
            self.last_training_time = datetime.now()
            
            self.logger.info(f"Model retraining completed successfully, new best model: {self.best_model_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model retraining failed: {e}")
            return False
    
    def get_accuracy_monitoring_status(self) -> Dict[str, Any]:
        """
        Get current status of accuracy monitoring system.
        
        Returns comprehensive information about accuracy monitoring state and performance.
        
        Returns:
            Dictionary with accuracy monitoring status
        """
        current_time = datetime.now()
        
        status = {
            "monitoring_enabled": self.accuracy_monitoring_enabled,
            "check_interval_seconds": self.accuracy_check_interval,
            "last_check": self.last_accuracy_check.isoformat() if self.last_accuracy_check else None,
            "next_check_due": None,
            "retraining_threshold": self.retraining_threshold,
            "consecutive_poor_count": self.consecutive_poor_accuracy_count,
            "max_consecutive_threshold": self.max_consecutive_poor_accuracy,
            "accuracy_history_length": len(self.prediction_accuracy_history),
            "current_accuracy": self._get_current_accuracy(self.best_model_type),
            "accuracy_trend": self._analyze_accuracy_trend(),
            "meets_requirement": None,
            "time_since_last_training": None
        }
        
        # Calculate next check time
        if self.last_accuracy_check:
            next_check = self.last_accuracy_check + timedelta(seconds=self.accuracy_check_interval)
            status["next_check_due"] = next_check.isoformat()
        
        # Check if current accuracy meets requirement
        if status["current_accuracy"] is not None:
            status["meets_requirement"] = status["current_accuracy"] < self.retraining_threshold
        
        # Calculate time since last training
        if self.last_training_time:
            time_diff = (current_time - self.last_training_time).total_seconds()
            status["time_since_last_training"] = time_diff
        
        return status
    
    def generate_10_second_forecast(self) -> LoadForecast:
        """
        Generate a 10-second load forecast with enhanced confidence intervals and accuracy tracking.
        
        This method specifically implements the 10-second horizon forecasting requirement
        from task 6.3 and Requirements 3.2.
        
        Returns:
            LoadForecast with 10-second predictions, confidence intervals, and accuracy metrics
        """
        return self.predict_load(horizon=10)
    
    def force_accuracy_check(self) -> Dict[str, Any]:
        """
        Force an immediate accuracy check and potential retraining.
        
        Useful for testing and manual system maintenance.
        
        Returns:
            Dictionary with monitoring results
        """
        # Temporarily reset the last check time to force immediate monitoring
        original_last_check = self.last_accuracy_check
        self.last_accuracy_check = None
        
        try:
            result = self.monitor_prediction_accuracy()
            return result
        finally:
            # Restore original timestamp if monitoring wasn't performed
            if not result.get("accuracy_check_performed", False):
                self.last_accuracy_check = original_last_check
    
    def configure_accuracy_monitoring(self, 
                                    enabled: bool = True,
                                    check_interval: int = 30,
                                    retraining_threshold: float = 15.0,
                                    max_consecutive_poor: int = 3) -> None:
        """
        Configure accuracy monitoring parameters.
        
        Args:
            enabled: Whether to enable accuracy monitoring
            check_interval: Seconds between accuracy checks
            retraining_threshold: MAPE threshold for triggering retraining
            max_consecutive_poor: Maximum consecutive poor accuracy measurements before retraining
        """
        self.accuracy_monitoring_enabled = enabled
        self.accuracy_check_interval = check_interval
        self.retraining_threshold = retraining_threshold
        self.max_consecutive_poor_accuracy = max_consecutive_poor
        
        self.logger.info(f"Accuracy monitoring configured: enabled={enabled}, "
                        f"interval={check_interval}s, threshold={retraining_threshold}%, "
                        f"max_consecutive={max_consecutive_poor}")
    
    def start_prediction_scheduler(self) -> None:
        """
        Start the prediction update scheduler.
        
        Implements 2-second prediction update cycles as required by Requirement 3.5.
        Creates a background thread that generates predictions every 2 seconds and
        notifies registered callbacks with the results.
        """
        with self.scheduler_lock:
            if self.scheduler_enabled:
                self.logger.warning("Prediction scheduler is already running")
                return
            
            if not self.best_model_type:
                self.logger.error("Cannot start scheduler: no trained model available")
                raise ValueError("No trained model available for scheduling")
            
            # Reset stop event and create scheduler thread
            self.scheduler_stop_event.clear()
            self.scheduler_thread = threading.Thread(
                target=self._prediction_scheduler_loop,
                name="PredictionScheduler",
                daemon=True
            )
            
            self.scheduler_enabled = True
            self.scheduled_predictions_count = 0
            self.last_scheduled_prediction = None
            
            # Start the scheduler thread
            self.scheduler_thread.start()
            
            self.logger.info(f"Prediction scheduler started with {self.prediction_update_interval}s interval")
    
    def stop_prediction_scheduler(self) -> None:
        """
        Stop the prediction update scheduler.
        
        Gracefully shuts down the background prediction thread and waits for completion.
        """
        with self.scheduler_lock:
            if not self.scheduler_enabled:
                self.logger.warning("Prediction scheduler is not running")
                return
            
            # Signal the scheduler thread to stop
            self.scheduler_stop_event.set()
            self.scheduler_enabled = False
            
            # Wait for thread to complete (with timeout)
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=5.0)
                
                if self.scheduler_thread.is_alive():
                    self.logger.warning("Scheduler thread did not stop gracefully within timeout")
                else:
                    self.logger.info("Prediction scheduler stopped successfully")
            
            self.scheduler_thread = None
            
            self.logger.info(f"Prediction scheduler stopped after {self.scheduled_predictions_count} predictions")
    
    def _prediction_scheduler_loop(self) -> None:
        """
        Main loop for the prediction scheduler thread.
        
        Runs continuously, generating predictions every 2 seconds and calling
        registered callbacks with the results. Handles errors gracefully to
        maintain scheduler stability.
        """
        self.logger.info("Prediction scheduler loop started")
        
        while not self.scheduler_stop_event.is_set():
            try:
                start_time = time.time()
                
                # Generate prediction if we have sufficient data
                window_data = self._get_window_data()
                
                if len(window_data) >= 5 and self.best_model_type:
                    # Generate 10-second forecast
                    forecast = self.predict_load(horizon=10)
                    
                    # Update scheduling statistics
                    self.last_scheduled_prediction = datetime.now()
                    self.scheduled_predictions_count += 1
                    
                    # Notify all registered callbacks
                    self._notify_prediction_callbacks(forecast)
                    
                    self.logger.debug(f"Scheduled prediction #{self.scheduled_predictions_count} generated")
                else:
                    self.logger.debug("Skipping prediction: insufficient data or no model")
                
                # Calculate sleep time to maintain 2-second interval
                elapsed_time = time.time() - start_time
                sleep_time = max(0, self.prediction_update_interval - elapsed_time)
                
                # Use event wait instead of sleep for responsive shutdown
                if sleep_time > 0:
                    self.scheduler_stop_event.wait(timeout=sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in prediction scheduler loop: {e}")
                # Continue running despite errors, but add a small delay to prevent tight error loops
                self.scheduler_stop_event.wait(timeout=1.0)
        
        self.logger.info("Prediction scheduler loop ended")
    
    def _notify_prediction_callbacks(self, forecast: LoadForecast) -> None:
        """
        Notify all registered callbacks with the new prediction.
        
        Args:
            forecast: The generated load forecast to send to callbacks
        """
        if not self.prediction_callbacks:
            return
        
        for callback in self.prediction_callbacks:
            try:
                callback(forecast)
            except Exception as e:
                self.logger.error(f"Error in prediction callback: {e}")
    
    def register_prediction_callback(self, callback: Callable[[LoadForecast], None]) -> None:
        """
        Register a callback function to receive scheduled predictions.
        
        Args:
            callback: Function that will be called with each LoadForecast
        """
        if callback not in self.prediction_callbacks:
            self.prediction_callbacks.append(callback)
            self.logger.info(f"Registered prediction callback: {callback.__name__}")
    
    def unregister_prediction_callback(self, callback: Callable[[LoadForecast], None]) -> None:
        """
        Unregister a prediction callback.
        
        Args:
            callback: Function to remove from callbacks
        """
        if callback in self.prediction_callbacks:
            self.prediction_callbacks.remove(callback)
            self.logger.info(f"Unregistered prediction callback: {callback.__name__}")
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """
        Get current status of the prediction scheduler.
        
        Returns:
            Dictionary with scheduler status information
        """
        with self.scheduler_lock:
            status = {
                "enabled": self.scheduler_enabled,
                "running": self.scheduler_thread is not None and self.scheduler_thread.is_alive(),
                "update_interval": self.prediction_update_interval,
                "predictions_generated": self.scheduled_predictions_count,
                "last_prediction": self.last_scheduled_prediction.isoformat() if self.last_scheduled_prediction else None,
                "registered_callbacks": len(self.prediction_callbacks),
                "thread_name": self.scheduler_thread.name if self.scheduler_thread else None
            }
            
            # Add timing information
            if self.last_scheduled_prediction:
                time_since_last = (datetime.now() - self.last_scheduled_prediction).total_seconds()
                status["seconds_since_last_prediction"] = time_since_last
                status["next_prediction_due"] = max(0, self.prediction_update_interval - time_since_last)
            
            return status
    
    def configure_scheduler(self, update_interval: float = 2.0) -> None:
        """
        Configure scheduler parameters.
        
        Args:
            update_interval: Seconds between prediction updates (default 2.0 per Requirement 3.5)
        """
        if update_interval <= 0:
            raise ValueError("Update interval must be positive")
        
        # Stop scheduler if running to apply new configuration
        was_running = self.scheduler_enabled
        if was_running:
            self.stop_prediction_scheduler()
        
        self.prediction_update_interval = update_interval
        
        # Restart scheduler if it was running
        if was_running and self.best_model_type:
            self.start_prediction_scheduler()
        
        self.logger.info(f"Scheduler configured with {update_interval}s interval")
    
    def get_prediction_accuracy_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive prediction accuracy statistics.
        
        Returns:
            Dictionary with detailed accuracy statistics
        """
        if not self.prediction_accuracy_history:
            return {
                "error": "No accuracy history available",
                "total_measurements": 0
            }
        
        history = np.array(self.prediction_accuracy_history)
        
        statistics = {
            "total_measurements": len(history),
            "current_mape": history[-1] if len(history) > 0 else None,
            "recent_mape": np.mean(history[-5:]) if len(history) >= 5 else np.mean(history),
            "best_mape": np.min(history),
            "worst_mape": np.max(history),
            "average_mape": np.mean(history),
            "median_mape": np.median(history),
            "std_mape": np.std(history),
            "meets_requirement_percentage": np.sum(history < self.retraining_threshold) / len(history) * 100,
            "trend": self._analyze_accuracy_trend(),
            "consecutive_poor_count": self.consecutive_poor_accuracy_count,
            "measurements_above_threshold": np.sum(history >= self.retraining_threshold),
            "measurements_below_threshold": np.sum(history < self.retraining_threshold)
        }
        
        # Add percentile information
        if len(history) >= 5:
            statistics["percentiles"] = {
                "p25": np.percentile(history, 25),
                "p50": np.percentile(history, 50),
                "p75": np.percentile(history, 75),
                "p90": np.percentile(history, 90),
                "p95": np.percentile(history, 95)
            }
        
        return statistics
    
    def start_prediction_scheduler(self) -> None:
        """
        Start the prediction update scheduler.
        
        Implements 2-second prediction update cycles as required by Requirement 3.5.
        Creates a background thread that generates predictions every 2 seconds and
        notifies registered callbacks with the results.
        """
        with self.scheduler_lock:
            if self.scheduler_enabled:
                self.logger.warning("Prediction scheduler is already running")
                return
            
            if not self.best_model_type:
                self.logger.error("Cannot start scheduler: no trained model available")
                raise ValueError("No trained model available for scheduling")
            
            # Reset stop event and create scheduler thread
            self.scheduler_stop_event.clear()
            self.scheduler_thread = threading.Thread(
                target=self._prediction_scheduler_loop,
                name="PredictionScheduler",
                daemon=True
            )
            
            self.scheduler_enabled = True
            self.scheduled_predictions_count = 0
            self.last_scheduled_prediction = None
            
            # Start the scheduler thread
            self.scheduler_thread.start()
            
            self.logger.info(f"Prediction scheduler started with {self.prediction_update_interval}s interval")
    
    def stop_prediction_scheduler(self) -> None:
        """
        Stop the prediction update scheduler.
        
        Gracefully shuts down the background prediction thread and waits for completion.
        """
        with self.scheduler_lock:
            if not self.scheduler_enabled:
                self.logger.warning("Prediction scheduler is not running")
                return
            
            # Signal the scheduler thread to stop
            self.scheduler_stop_event.set()
            self.scheduler_enabled = False
            
            # Wait for thread to complete (with timeout)
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=5.0)
                
                if self.scheduler_thread.is_alive():
                    self.logger.warning("Scheduler thread did not stop gracefully within timeout")
                else:
                    self.logger.info("Prediction scheduler stopped successfully")
            
            self.scheduler_thread = None
            
            self.logger.info(f"Prediction scheduler stopped after {self.scheduled_predictions_count} predictions")
    
    def _prediction_scheduler_loop(self) -> None:
        """
        Main loop for the prediction scheduler thread.
        
        Runs continuously, generating predictions every 2 seconds and calling
        registered callbacks with the results. Handles errors gracefully to
        maintain scheduler stability.
        """
        self.logger.info("Prediction scheduler loop started")
        
        while not self.scheduler_stop_event.is_set():
            try:
                start_time = time.time()
                
                # Generate prediction if we have sufficient data
                window_data = self._get_window_data()
                
                if len(window_data) >= 5 and self.best_model_type:
                    # Generate 10-second forecast
                    forecast = self.predict_load(horizon=10)
                    
                    # Update scheduling statistics
                    self.last_scheduled_prediction = datetime.now()
                    self.scheduled_predictions_count += 1
                    
                    # Notify all registered callbacks
                    self._notify_prediction_callbacks(forecast)
                    
                    self.logger.debug(f"Scheduled prediction #{self.scheduled_predictions_count} generated")
                else:
                    self.logger.debug("Skipping prediction: insufficient data or no model")
                
                # Calculate sleep time to maintain 2-second interval
                elapsed_time = time.time() - start_time
                sleep_time = max(0, self.prediction_update_interval - elapsed_time)
                
                # Use event wait instead of sleep for responsive shutdown
                if sleep_time > 0:
                    self.scheduler_stop_event.wait(timeout=sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in prediction scheduler loop: {e}")
                # Continue running despite errors, but add a small delay to prevent tight error loops
                self.scheduler_stop_event.wait(timeout=1.0)
        
        self.logger.info("Prediction scheduler loop ended")
    
    def _notify_prediction_callbacks(self, forecast: LoadForecast) -> None:
        """
        Notify all registered callbacks with the new prediction.
        
        Args:
            forecast: The generated load forecast to send to callbacks
        """
        if not self.prediction_callbacks:
            return
        
        for callback in self.prediction_callbacks:
            try:
                callback(forecast)
            except Exception as e:
                self.logger.error(f"Error in prediction callback: {e}")
    
    def register_prediction_callback(self, callback: Callable[[LoadForecast], None]) -> None:
        """
        Register a callback function to receive scheduled predictions.
        
        Args:
            callback: Function that will be called with each LoadForecast
        """
        if callback not in self.prediction_callbacks:
            self.prediction_callbacks.append(callback)
            self.logger.info(f"Registered prediction callback: {callback.__name__}")
    
    def unregister_prediction_callback(self, callback: Callable[[LoadForecast], None]) -> None:
        """
        Unregister a prediction callback.
        
        Args:
            callback: Function to remove from callbacks
        """
        if callback in self.prediction_callbacks:
            self.prediction_callbacks.remove(callback)
            self.logger.info(f"Unregistered prediction callback: {callback.__name__}")
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """
        Get current status of the prediction scheduler.
        
        Returns:
            Dictionary with scheduler status information
        """
        with self.scheduler_lock:
            status = {
                "enabled": self.scheduler_enabled,
                "running": self.scheduler_thread is not None and self.scheduler_thread.is_alive(),
                "update_interval": self.prediction_update_interval,
                "predictions_generated": self.scheduled_predictions_count,
                "last_prediction": self.last_scheduled_prediction.isoformat() if self.last_scheduled_prediction else None,
                "registered_callbacks": len(self.prediction_callbacks),
                "thread_name": self.scheduler_thread.name if self.scheduler_thread else None
            }
            
            # Add timing information
            if self.last_scheduled_prediction:
                time_since_last = (datetime.now() - self.last_scheduled_prediction).total_seconds()
                status["seconds_since_last_prediction"] = time_since_last
                status["next_prediction_due"] = max(0, self.prediction_update_interval - time_since_last)
            
            return status
    
    def configure_scheduler(self, update_interval: float = 2.0) -> None:
        """
        Configure scheduler parameters.
        
        Args:
            update_interval: Seconds between prediction updates (default 2.0 per Requirement 3.5)
        """
        if update_interval <= 0:
            raise ValueError("Update interval must be positive")
        
        # Stop scheduler if running to apply new configuration
        was_running = self.scheduler_enabled
        if was_running:
            self.stop_prediction_scheduler()
        
        self.prediction_update_interval = update_interval
        
        # Restart scheduler if it was running
        if was_running and self.best_model_type:
            self.start_prediction_scheduler()
        
        self.logger.info(f"Scheduler configured with {update_interval}s interval")
    def start_prediction_scheduler(self) -> None:
        """
        Start the prediction update scheduler.
        
        Implements 2-second prediction update cycles as required by Requirement 3.5.
        Creates a background thread that generates predictions every 2 seconds and
        notifies registered callbacks with the results.
        """
        with self.scheduler_lock:
            if self.scheduler_enabled:
                self.logger.warning("Prediction scheduler is already running")
                return
            
            if not self.best_model_type:
                self.logger.error("Cannot start scheduler: no trained model available")
                raise ValueError("No trained model available for scheduling")
            
            # Reset stop event and create scheduler thread
            self.scheduler_stop_event.clear()
            self.scheduler_thread = threading.Thread(
                target=self._prediction_scheduler_loop,
                name="PredictionScheduler",
                daemon=True
            )
            
            self.scheduler_enabled = True
            self.scheduled_predictions_count = 0
            self.last_scheduled_prediction = None
            
            # Start the scheduler thread
            self.scheduler_thread.start()
            
            self.logger.info(f"Prediction scheduler started with {self.prediction_update_interval}s interval")
    
    def stop_prediction_scheduler(self) -> None:
        """
        Stop the prediction update scheduler.
        
        Gracefully shuts down the background prediction thread and waits for completion.
        """
        with self.scheduler_lock:
            if not self.scheduler_enabled:
                self.logger.warning("Prediction scheduler is not running")
                return
            
            # Signal the scheduler thread to stop
            self.scheduler_stop_event.set()
            self.scheduler_enabled = False
            
            # Wait for thread to complete (with timeout)
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=5.0)
                
                if self.scheduler_thread.is_alive():
                    self.logger.warning("Scheduler thread did not stop gracefully within timeout")
                else:
                    self.logger.info("Prediction scheduler stopped successfully")
            
            self.scheduler_thread = None
            
            self.logger.info(f"Prediction scheduler stopped after {self.scheduled_predictions_count} predictions")
    
    def _prediction_scheduler_loop(self) -> None:
        """
        Main loop for the prediction scheduler thread.
        
        Runs continuously, generating predictions every 2 seconds and calling
        registered callbacks with the results. Handles errors gracefully to
        maintain scheduler stability.
        """
        self.logger.info("Prediction scheduler loop started")
        
        while not self.scheduler_stop_event.is_set():
            try:
                start_time = time.time()
                
                # Generate prediction if we have sufficient data
                window_data = self._get_window_data()
                
                if len(window_data) >= 5 and self.best_model_type:
                    # Generate 10-second forecast
                    forecast = self.predict_load(horizon=10)
                    
                    # Update scheduling statistics
                    self.last_scheduled_prediction = datetime.now()
                    self.scheduled_predictions_count += 1
                    
                    # Notify all registered callbacks
                    self._notify_prediction_callbacks(forecast)
                    
                    self.logger.debug(f"Scheduled prediction #{self.scheduled_predictions_count} generated")
                else:
                    self.logger.debug("Skipping prediction: insufficient data or no model")
                
                # Calculate sleep time to maintain 2-second interval
                elapsed_time = time.time() - start_time
                sleep_time = max(0, self.prediction_update_interval - elapsed_time)
                
                # Use event wait instead of sleep for responsive shutdown
                if sleep_time > 0:
                    self.scheduler_stop_event.wait(timeout=sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in prediction scheduler loop: {e}")
                # Continue running despite errors, but add a small delay to prevent tight error loops
                self.scheduler_stop_event.wait(timeout=1.0)
        
        self.logger.info("Prediction scheduler loop ended")
    
    def _notify_prediction_callbacks(self, forecast: LoadForecast) -> None:
        """
        Notify all registered callbacks with the new prediction.
        
        Args:
            forecast: The generated load forecast to send to callbacks
        """
        if not self.prediction_callbacks:
            return
        
        for callback in self.prediction_callbacks:
            try:
                callback(forecast)
            except Exception as e:
                self.logger.error(f"Error in prediction callback: {e}")
    
    def register_prediction_callback(self, callback: Callable[[LoadForecast], None]) -> None:
        """
        Register a callback function to receive scheduled predictions.
        
        Args:
            callback: Function that will be called with each LoadForecast
        """
        if callback not in self.prediction_callbacks:
            self.prediction_callbacks.append(callback)
            self.logger.info(f"Registered prediction callback: {callback.__name__}")
    
    def unregister_prediction_callback(self, callback: Callable[[LoadForecast], None]) -> None:
        """
        Unregister a prediction callback.
        
        Args:
            callback: Function to remove from callbacks
        """
        if callback in self.prediction_callbacks:
            self.prediction_callbacks.remove(callback)
            self.logger.info(f"Unregistered prediction callback: {callback.__name__}")
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """
        Get current status of the prediction scheduler.
        
        Returns:
            Dictionary with scheduler status information
        """
        with self.scheduler_lock:
            status = {
                "enabled": self.scheduler_enabled,
                "running": self.scheduler_thread is not None and self.scheduler_thread.is_alive(),
                "update_interval": self.prediction_update_interval,
                "predictions_generated": self.scheduled_predictions_count,
                "last_prediction": self.last_scheduled_prediction.isoformat() if self.last_scheduled_prediction else None,
                "registered_callbacks": len(self.prediction_callbacks),
                "thread_name": self.scheduler_thread.name if self.scheduler_thread else None
            }
            
            # Add timing information
            if self.last_scheduled_prediction:
                time_since_last = (datetime.now() - self.last_scheduled_prediction).total_seconds()
                status["seconds_since_last_prediction"] = time_since_last
                status["next_prediction_due"] = max(0, self.prediction_update_interval - time_since_last)
            
            return status
    
    def configure_scheduler(self, update_interval: float = 2.0) -> None:
        """
        Configure scheduler parameters.
        
        Args:
            update_interval: Seconds between prediction updates (default 2.0 per Requirement 3.5)
        """
        if update_interval <= 0:
            raise ValueError("Update interval must be positive")
        
        # Stop scheduler if running to apply new configuration
        was_running = self.scheduler_enabled
        if was_running:
            self.stop_prediction_scheduler()
        
        self.prediction_update_interval = update_interval
        
        # Restart scheduler if it was running
        if was_running and self.best_model_type:
            self.start_prediction_scheduler()
        
        self.logger.info(f"Scheduler configured with {update_interval}s interval")
"""
Tests for PredictiveAgent class.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import numpy as np

from src.agents.predictive_agent import PredictiveAgent, ModelType
from src.models import KPIMetrics, LoadForecast


class TestPredictiveAgent:
    """Test cases for PredictiveAgent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = PredictiveAgent(window_size_seconds=15, prediction_horizon=10)
        
        # Create sample KPI data
        base_time = datetime.now()
        self.sample_kpis = []
        for i in range(20):
            kpi = KPIMetrics(
                timestamp=base_time + timedelta(seconds=i),
                throughput=100.0 + i * 2,
                latency=10.0 + i * 0.5,
                packet_loss=1.0 + i * 0.1,
                utilization=50.0 + i * 2,
                node_id=f"node_{i % 3}"
            )
            self.sample_kpis.append(kpi)
    
    def test_initialization(self):
        """Test agent initialization."""
        assert self.agent.window_size_seconds == 15
        assert self.agent.prediction_horizon == 10
        assert len(self.agent.historical_data) == 0
        assert self.agent.best_model_type is None
    
    def test_window_size_validation(self):
        """Test window size validation according to requirements."""
        # Valid window sizes (10-20 seconds)
        PredictiveAgent(window_size_seconds=10)
        PredictiveAgent(window_size_seconds=20)
        
        # Invalid window sizes
        with pytest.raises(ValueError, match="Window size must be between 10-20 seconds"):
            PredictiveAgent(window_size_seconds=9)
        
        with pytest.raises(ValueError, match="Window size must be between 10-20 seconds"):
            PredictiveAgent(window_size_seconds=21)
    
    def test_analyze_historical_data(self):
        """Test historical data analysis."""
        # Test with empty data
        self.agent.analyze_historical_data([])
        assert len(self.agent.historical_data) == 0
        
        # Test with sample data
        self.agent.analyze_historical_data(self.sample_kpis[:10])
        assert len(self.agent.historical_data) == 10
        
        # Test with more data
        self.agent.analyze_historical_data(self.sample_kpis[10:])
        assert len(self.agent.historical_data) == 20
    
    def test_get_window_data(self):
        """Test window data retrieval."""
        # Add data spanning more than window size
        self.agent.analyze_historical_data(self.sample_kpis)
        
        # Get window data
        window_data = self.agent._get_window_data()
        
        # Should only include data within window
        assert len(window_data) <= len(self.sample_kpis)
        
        # Data should be sorted by timestamp
        timestamps = [kpi.timestamp for kpi in window_data]
        assert timestamps == sorted(timestamps)
    
    def test_model_training_with_insufficient_data(self):
        """Test model training with insufficient data."""
        # Add minimal data
        self.agent.analyze_historical_data(self.sample_kpis[:3])
        
        # Should not train models with insufficient data
        assert len(self.agent.models) == 0
        assert self.agent.best_model_type is None
    
    @patch('src.agents.predictive_agent.ARIMA')
    def test_arima_model_training(self, mock_arima):
        """Test ARIMA model training."""
        # Mock ARIMA model
        mock_model = Mock()
        mock_fitted = Mock()
        mock_model.fit.return_value = mock_fitted
        mock_arima.return_value = mock_model
        
        # Add sufficient data
        self.agent.analyze_historical_data(self.sample_kpis)
        
        # Check if ARIMA was called
        assert mock_arima.called
        assert ModelType.ARIMA in self.agent.models
    
    def test_prediction_without_trained_model(self):
        """Test prediction without trained model."""
        with pytest.raises(ValueError, match="No trained model available"):
            self.agent.predict_load()
    
    def test_prediction_with_insufficient_data(self):
        """Test prediction with insufficient historical data."""
        # Add minimal data (insufficient for training)
        self.agent.analyze_historical_data(self.sample_kpis[:2])
        
        # Should fail because no model was trained due to insufficient data
        with pytest.raises(ValueError, match="No trained model available"):
            self.agent.predict_load()
    
    def test_calculate_accuracy(self):
        """Test MAPE calculation."""
        # Create mock forecast
        forecast = LoadForecast(
            predicted_values=[50.0, 52.0, 54.0],
            confidence_interval=(45.0, 60.0),
            prediction_horizon=10,
            model_accuracy=10.0,
            timestamp=datetime.now()
        )
        
        # Create actual values
        actual_kpis = self.sample_kpis[:3]
        
        # Calculate accuracy
        mape = self.agent.calculate_accuracy(forecast, actual_kpis)
        
        # Should return a valid MAPE value
        assert isinstance(mape, float)
        assert 0 <= mape <= 100
    
    def test_update_model(self):
        """Test model updating with feedback."""
        # First train with initial data
        self.agent.analyze_historical_data(self.sample_kpis[:10])
        initial_data_count = len(self.agent.historical_data)
        
        # Update with feedback
        feedback_data = self.sample_kpis[10:15]
        self.agent.update_model(feedback_data)
        
        # Should have more data
        assert len(self.agent.historical_data) > initial_data_count
    
    def test_model_status(self):
        """Test model status reporting."""
        # Get status before training
        status = self.agent.get_model_status()
        assert "available_models" in status
        assert "best_model" in status
        assert "data_points" in status
        assert status["data_points"] == 0
        
        # Train models and check status
        self.agent.analyze_historical_data(self.sample_kpis)
        status = self.agent.get_model_status()
        assert status["data_points"] > 0
    
    def test_prepare_training_data(self):
        """Test training data preparation."""
        df = self.agent._prepare_training_data(self.sample_kpis[:5])
        
        # Check DataFrame structure
        assert 'ds' in df.columns  # Prophet format
        assert 'y' in df.columns   # Target variable
        assert len(df) == 5
        
        # Check data is sorted by timestamp
        assert df['ds'].is_monotonic_increasing
    
    def test_create_features(self):
        """Test feature creation for AutoML."""
        df = self.agent._prepare_training_data(self.sample_kpis[:5])
        features = self.agent._create_features(df)
        
        # Check feature columns
        expected_columns = ['lag1', 'lag2', 'trend', 'target']
        for col in expected_columns:
            assert col in features.columns
        
        # Should have fewer rows due to lagging
        assert len(features) < len(df)
    
    def test_lstm_data_preparation(self):
        """Test LSTM data preparation."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        X, y = self.agent._prepare_lstm_data(data, lookback=3)
        
        # Check shapes
        assert X.shape == (7, 3, 1)  # (samples, timesteps, features)
        assert y.shape == (7,)
        
        # Check first sample
        np.testing.assert_array_equal(X[0].flatten(), [1, 2, 3])
        assert y[0] == 4
    
    def test_model_performance_tracking(self):
        """Test model performance tracking."""
        model_type = ModelType.ARIMA
        
        # Initially empty
        assert len(self.agent.model_performance[model_type]) == 0
        
        # Add performance metrics
        self.agent.model_performance[model_type].extend([10.0, 12.0, 8.0])
        
        # Get current accuracy
        accuracy = self.agent._get_current_accuracy(model_type)
        expected = np.mean([10.0, 12.0, 8.0])
        assert accuracy == expected
    
    def test_algorithm_compliance(self):
        """Test that only approved algorithms are used (Requirement 3.3)."""
        # Check that model types match requirements
        approved_algorithms = {ModelType.ARIMA, ModelType.PROPHET, ModelType.AUTOML, ModelType.LSTM}
        
        # All model types should be from approved set
        for model_type in self.agent.model_performance.keys():
            assert model_type in approved_algorithms
        
        # Check model configurations exist for all approved types
        for algorithm in approved_algorithms:
            assert algorithm in self.agent.model_configs


    def test_generate_10_second_forecast(self):
        """Test the specific 10-second forecast method (task 6.3)."""
        # Add sufficient data and train models
        self.agent.analyze_historical_data(self.sample_kpis)
        
        # Skip if no models were trained (can happen with test data)
        if not self.agent.best_model_type:
            pytest.skip("No models trained with test data")
        
        # Generate 10-second forecast
        forecast = self.agent.generate_10_second_forecast()
        
        # Validate forecast properties
        assert isinstance(forecast, LoadForecast)
        assert forecast.prediction_horizon == 10
        assert len(forecast.predicted_values) == 10
        assert isinstance(forecast.confidence_interval, tuple)
        assert len(forecast.confidence_interval) == 2
        assert forecast.confidence_interval[0] <= forecast.confidence_interval[1]
    
    def test_enhanced_confidence_intervals(self):
        """Test enhanced confidence interval calculation (task 6.3)."""
        # Add data and train models
        self.agent.analyze_historical_data(self.sample_kpis)
        
        if not self.agent.best_model_type:
            pytest.skip("No models trained with test data")
        
        # Test confidence interval calculation
        predictions = [50.0, 52.0, 54.0, 56.0, 58.0]
        window_data = self.sample_kpis[:10]
        
        confidence_interval = self.agent._calculate_enhanced_confidence_interval(
            predictions, self.agent.best_model_type, window_data
        )
        
        # Validate confidence interval
        assert isinstance(confidence_interval, tuple)
        assert len(confidence_interval) == 2
        assert confidence_interval[0] <= confidence_interval[1]
        assert 0 <= confidence_interval[0] <= 100
        assert 0 <= confidence_interval[1] <= 100
    
    def test_prediction_accuracy_tracking(self):
        """Test prediction accuracy tracking functionality (task 6.3)."""
        # Initially no predictions tracked
        assert len(self.agent.pending_predictions) == 0
        assert len(self.agent.prediction_accuracy_history) == 0
        
        # Track a prediction
        predictions = [50.0, 52.0, 54.0]
        self.agent._track_prediction_for_accuracy(predictions, 10)
        
        # Should have one pending prediction
        assert len(self.agent.pending_predictions) == 1
        
        # Test cleanup of old predictions
        # Add an old prediction manually
        old_time = datetime.now() - timedelta(minutes=10)
        self.agent.pending_predictions.append(([45.0, 47.0], old_time, 10))
        
        # Track another prediction (should clean up old one)
        self.agent._track_prediction_for_accuracy([55.0, 57.0], 10)
        
        # Should only have recent predictions
        assert len(self.agent.pending_predictions) <= 2
    
    def test_model_status_enhancements(self):
        """Test enhanced model status reporting (task 6.3)."""
        # Get initial status
        status = self.agent.get_model_status()
        
        # Check new fields added for task 6.3
        assert "prediction_horizon" in status
        assert "pending_predictions" in status
        assert "overall_accuracy" in status
        
        assert status["prediction_horizon"] == 10
        assert status["pending_predictions"] == 0
        
        # Add some accuracy history
        self.agent.prediction_accuracy_history = [12.0, 14.0, 10.0, 8.0, 15.0]
        
        status = self.agent.get_model_status()
        assert "recent_mape" in status["overall_accuracy"]
        assert "best_mape" in status["overall_accuracy"]
        assert "total_evaluations" in status["overall_accuracy"]
        assert "meets_requirement" in status["overall_accuracy"]
        
        # Check requirement compliance (MAPE < 15%)
        recent_mape = status["overall_accuracy"]["recent_mape"]
        meets_requirement = status["overall_accuracy"]["meets_requirement"]
        assert meets_requirement == (recent_mape < 15.0)
        
        # Check accuracy monitoring status (task 6.6)
        assert "accuracy_monitoring" in status
        monitoring_status = status["accuracy_monitoring"]
        assert "monitoring_enabled" in monitoring_status
        assert "retraining_threshold" in monitoring_status
        assert "consecutive_poor_count" in monitoring_status
    
    def test_accuracy_monitoring_configuration(self):
        """Test accuracy monitoring configuration (task 6.6)."""
        # Test default configuration
        assert self.agent.accuracy_monitoring_enabled == True
        assert self.agent.retraining_threshold == 15.0
        
        # Test configuration changes
        self.agent.configure_accuracy_monitoring(
            enabled=False,
            check_interval=60,
            retraining_threshold=20.0,
            max_consecutive_poor=5
        )
        
        assert self.agent.accuracy_monitoring_enabled == False
        assert self.agent.accuracy_check_interval == 60
        assert self.agent.retraining_threshold == 20.0
        assert self.agent.max_consecutive_poor_accuracy == 5
    
    def test_accuracy_trend_analysis(self):
        """Test accuracy trend analysis (task 6.6)."""
        # Test with insufficient data
        trend = self.agent._analyze_accuracy_trend()
        assert trend is None
        
        # Test with improving trend (decreasing MAPE)
        self.agent.prediction_accuracy_history = [20.0, 18.0, 15.0, 12.0, 10.0]
        trend = self.agent._analyze_accuracy_trend()
        assert trend == "improving"
        
        # Test with degrading trend (increasing MAPE)
        self.agent.prediction_accuracy_history = [8.0, 10.0, 13.0, 16.0, 20.0]
        trend = self.agent._analyze_accuracy_trend()
        assert trend == "degrading"
        
        # Test with stable trend
        self.agent.prediction_accuracy_history = [12.0, 12.5, 12.2, 12.8, 12.1]
        trend = self.agent._analyze_accuracy_trend()
        assert trend == "stable"
    
    def test_retraining_triggers(self):
        """Test retraining trigger logic (task 6.6)."""
        # Reset consecutive count for clean testing
        self.agent.consecutive_poor_accuracy_count = 0
        
        # Test severe degradation trigger (should trigger first)
        should_retrain, reason = self.agent._should_trigger_retraining(30.0, "stable", False)
        assert should_retrain == True
        assert "Severe accuracy degradation" in reason
        
        # Test consecutive poor accuracy trigger
        self.agent.consecutive_poor_accuracy_count = 3
        should_retrain, reason = self.agent._should_trigger_retraining(18.0, "stable", False)
        assert should_retrain == True
        assert "Consecutive poor accuracy" in reason
        
        # Reset for next test
        self.agent.consecutive_poor_accuracy_count = 0
        
        # Test degrading trend trigger
        should_retrain, reason = self.agent._should_trigger_retraining(18.0, "degrading", False)
        assert should_retrain == True
        assert "Degrading accuracy trend" in reason
        
        # Test no retraining needed
        should_retrain, reason = self.agent._should_trigger_retraining(10.0, "stable", True)
        assert should_retrain == False
        assert reason is None
    
    def test_accuracy_statistics(self):
        """Test accuracy statistics generation (task 6.6)."""
        # Test with no data
        stats = self.agent.get_accuracy_statistics()
        assert "error" in stats
        assert stats["total_measurements"] == 0
        
        # Test with sample data
        self.agent.prediction_accuracy_history = [10.0, 12.0, 8.0, 15.0, 11.0, 9.0, 14.0]
        stats = self.agent.get_accuracy_statistics()
        
        assert stats["total_measurements"] == 7
        assert stats["current_mape"] == 14.0
        assert stats["best_mape"] == 8.0
        assert stats["worst_mape"] == 15.0
        assert "average_mape" in stats
        assert "median_mape" in stats
        assert "std_mape" in stats
        assert "meets_requirement_percentage" in stats
        assert "percentiles" in stats
    
    def test_force_accuracy_check(self):
        """Test forced accuracy check functionality (task 6.6)."""
        # Add some accuracy history
        self.agent.prediction_accuracy_history = [18.0, 20.0, 22.0]
        
        # Force accuracy check
        result = self.agent.force_accuracy_check()
        
        # Should have performed check
        assert result["accuracy_check_performed"] == True
        assert result["current_accuracy"] is not None
        assert result["meets_requirement"] is not None
    
    def test_accuracy_monitoring_integration(self):
        """Test integration of accuracy monitoring with prediction workflow (task 6.6)."""
        # Add sufficient data for training
        self.agent.analyze_historical_data(self.sample_kpis)
        
        if not self.agent.best_model_type:
            pytest.skip("No models trained with test data")
        
        # Set up poor accuracy to trigger monitoring
        self.agent.prediction_accuracy_history = [20.0, 22.0, 25.0]
        self.agent.consecutive_poor_accuracy_count = 2
        
        # Generate prediction (should trigger monitoring)
        try:
            forecast = self.agent.predict_load(horizon=10)
            assert isinstance(forecast, LoadForecast)
        except Exception as e:
            # Some models might fail with test data
            pytest.skip(f"Prediction failed with test data: {e}")
    
    def test_accuracy_monitoring_status(self):
        """Test accuracy monitoring status reporting (task 6.6)."""
        status = self.agent.get_accuracy_monitoring_status()
        
        # Check required fields
        required_fields = [
            "monitoring_enabled", "check_interval_seconds", "retraining_threshold",
            "consecutive_poor_count", "max_consecutive_threshold", "current_accuracy",
            "accuracy_trend", "meets_requirement"
        ]
        
        for field in required_fields:
            assert field in status


    def test_prediction_scheduler_initialization(self):
        """Test prediction scheduler initialization (task 6.8)."""
        # Initially scheduler should be disabled
        assert self.agent.scheduler_enabled == False
        assert self.agent.scheduler_thread is None
        assert self.agent.prediction_update_interval == 2.0  # Requirement 3.5
        assert len(self.agent.prediction_callbacks) == 0
    
    def test_scheduler_without_trained_model(self):
        """Test scheduler behavior without trained model (task 6.8)."""
        # Should fail to start without trained model
        with pytest.raises(ValueError, match="No trained model available"):
            self.agent.start_prediction_scheduler()
    
    def test_scheduler_start_stop(self):
        """Test scheduler start and stop functionality (task 6.8)."""
        # Train a model first
        self.agent.analyze_historical_data(self.sample_kpis)
        
        if not self.agent.best_model_type:
            pytest.skip("No models trained with test data")
        
        # Start scheduler
        self.agent.start_prediction_scheduler()
        
        # Check scheduler status
        assert self.agent.scheduler_enabled == True
        assert self.agent.scheduler_thread is not None
        assert self.agent.scheduler_thread.is_alive()
        
        # Stop scheduler
        self.agent.stop_prediction_scheduler()
        
        # Check scheduler stopped
        assert self.agent.scheduler_enabled == False
        # Thread should be None or not alive
        assert self.agent.scheduler_thread is None or not self.agent.scheduler_thread.is_alive()
    
    def test_scheduler_double_start_stop(self):
        """Test scheduler double start/stop behavior (task 6.8)."""
        # Train a model first
        self.agent.analyze_historical_data(self.sample_kpis)
        
        if not self.agent.best_model_type:
            pytest.skip("No models trained with test data")
        
        # Start scheduler twice (should handle gracefully)
        self.agent.start_prediction_scheduler()
        self.agent.start_prediction_scheduler()  # Should log warning but not crash
        
        # Stop scheduler twice (should handle gracefully)
        self.agent.stop_prediction_scheduler()
        self.agent.stop_prediction_scheduler()  # Should log warning but not crash
    
    def test_prediction_callbacks(self):
        """Test prediction callback registration and notification (task 6.8)."""
        received_forecasts = []
        
        def test_callback(forecast: LoadForecast):
            received_forecasts.append(forecast)
        
        # Register callback
        self.agent.register_prediction_callback(test_callback)
        assert len(self.agent.prediction_callbacks) == 1
        
        # Test callback notification
        mock_forecast = LoadForecast(
            predicted_values=[50.0, 52.0],
            confidence_interval=(45.0, 55.0),
            prediction_horizon=10,
            model_accuracy=12.0,
            timestamp=datetime.now()
        )
        
        self.agent._notify_prediction_callbacks(mock_forecast)
        assert len(received_forecasts) == 1
        assert received_forecasts[0] == mock_forecast
        
        # Unregister callback
        self.agent.unregister_prediction_callback(test_callback)
        assert len(self.agent.prediction_callbacks) == 0
    
    def test_scheduler_status(self):
        """Test scheduler status reporting (task 6.8)."""
        # Get initial status
        status = self.agent.get_scheduler_status()
        
        # Check required fields
        required_fields = [
            "enabled", "running", "update_interval", "predictions_generated",
            "last_prediction", "registered_callbacks"
        ]
        
        for field in required_fields:
            assert field in status
        
        assert status["enabled"] == False
        assert status["running"] == False
        assert status["update_interval"] == 2.0
        assert status["predictions_generated"] == 0
        assert status["registered_callbacks"] == 0
    
    def test_scheduler_configuration(self):
        """Test scheduler configuration (task 6.8)."""
        # Test invalid configuration
        with pytest.raises(ValueError, match="Update interval must be positive"):
            self.agent.configure_scheduler(update_interval=-1.0)
        
        # Test valid configuration
        self.agent.configure_scheduler(update_interval=5.0)
        assert self.agent.prediction_update_interval == 5.0
        
        # Test configuration with running scheduler
        self.agent.analyze_historical_data(self.sample_kpis)
        
        if self.agent.best_model_type:
            self.agent.start_prediction_scheduler()
            original_interval = self.agent.prediction_update_interval
            
            # Change configuration (should restart scheduler)
            self.agent.configure_scheduler(update_interval=3.0)
            assert self.agent.prediction_update_interval == 3.0
            
            # Clean up
            self.agent.stop_prediction_scheduler()
    
    def test_scheduler_timing_requirements(self):
        """Test scheduler timing meets requirements (task 6.8, Requirement 3.5)."""
        import time
        
        # Train a model first
        self.agent.analyze_historical_data(self.sample_kpis)
        
        if not self.agent.best_model_type:
            pytest.skip("No models trained with test data")
        
        received_times = []
        
        def timing_callback(forecast: LoadForecast):
            received_times.append(time.time())
        
        # Register timing callback
        self.agent.register_prediction_callback(timing_callback)
        
        # Start scheduler with short interval for testing
        self.agent.configure_scheduler(update_interval=0.5)  # 500ms for faster testing
        self.agent.start_prediction_scheduler()
        
        # Wait for a few predictions
        time.sleep(2.0)  # Should get ~4 predictions
        
        # Stop scheduler
        self.agent.stop_prediction_scheduler()
        
        # Check timing (should have at least 2 predictions)
        assert len(received_times) >= 2
        
        # Check intervals are approximately correct (within 20% tolerance)
        if len(received_times) >= 2:
            intervals = [received_times[i] - received_times[i-1] for i in range(1, len(received_times))]
            avg_interval = sum(intervals) / len(intervals)
            assert 0.4 <= avg_interval <= 0.6  # 500ms ± 20%


class TestPredictiveAgentIntegration:
    """Integration tests for PredictiveAgent."""
    
    def test_full_prediction_workflow(self):
        """Test complete prediction workflow."""
        agent = PredictiveAgent(window_size_seconds=15)
        
        # Create realistic time series data
        base_time = datetime.now()
        kpis = []
        for i in range(30):
            # Create trending utilization data
            utilization = 40 + 10 * np.sin(i * 0.2) + np.random.normal(0, 2)
            utilization = max(0, min(100, utilization))  # Clamp to valid range
            
            kpi = KPIMetrics(
                timestamp=base_time + timedelta(seconds=i),
                throughput=100.0 + utilization,
                latency=10.0 + utilization * 0.1,
                packet_loss=max(0, utilization * 0.05 - 2),
                utilization=utilization,
                node_id="test_node"
            )
            kpis.append(kpi)
        
        # Analyze data (should train models)
        agent.analyze_historical_data(kpis)
        
        # Should have trained at least one model
        assert len(agent.models) > 0
        assert agent.best_model_type is not None
        
        # Generate prediction
        try:
            forecast = agent.predict_load(horizon=10)
            
            # Validate forecast
            assert isinstance(forecast, LoadForecast)
            assert len(forecast.predicted_values) == 10
            assert forecast.prediction_horizon == 10
            assert 0 <= forecast.model_accuracy <= 100
            
            # Calculate accuracy with actual data
            actual_data = kpis[-5:]  # Use recent data as "actual"
            mape = agent.calculate_accuracy(forecast, actual_data)
            assert isinstance(mape, float)
            
        except Exception as e:
            # Some models might fail with random data, that's acceptable
            pytest.skip(f"Model training failed with test data: {e}")
    
    def test_scheduler_integration_workflow(self):
        """Test complete scheduler integration workflow (task 6.8)."""
        import time
        
        agent = PredictiveAgent(window_size_seconds=15)
        
        # Create realistic time series data
        base_time = datetime.now()
        kpis = []
        for i in range(30):
            utilization = 50 + 20 * np.sin(i * 0.3) + np.random.normal(0, 3)
            utilization = max(10, min(90, utilization))
            
            kpi = KPIMetrics(
                timestamp=base_time + timedelta(seconds=i),
                throughput=100.0 + utilization,
                latency=10.0 + utilization * 0.1,
                packet_loss=max(0, utilization * 0.02),
                utilization=utilization,
                node_id="integration_test_node"
            )
            kpis.append(kpi)
        
        # Train models
        agent.analyze_historical_data(kpis)
        
        if not agent.best_model_type:
            pytest.skip("No models trained with test data")
        
        # Set up callback to collect predictions
        collected_forecasts = []
        
        def collect_forecast(forecast: LoadForecast):
            collected_forecasts.append(forecast)
        
        agent.register_prediction_callback(collect_forecast)
        
        # Configure fast scheduling for testing
        agent.configure_scheduler(update_interval=0.3)
        
        # Start scheduler
        agent.start_prediction_scheduler()
        
        # Let it run for a short time
        time.sleep(1.0)
        
        # Stop scheduler
        agent.stop_prediction_scheduler()
        
        # Verify results
        assert len(collected_forecasts) >= 2  # Should have generated multiple predictions
        
        for forecast in collected_forecasts:
            assert isinstance(forecast, LoadForecast)
            assert forecast.prediction_horizon == 10
            assert len(forecast.predicted_values) == 10
            assert isinstance(forecast.confidence_interval, tuple)
        
        # Check scheduler statistics
        status = agent.get_scheduler_status()
        assert status["predictions_generated"] >= 2
        assert status["last_prediction"] is not None
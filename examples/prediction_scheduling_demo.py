#!/usr/bin/env python3
"""
Prediction Scheduling Demo

Demonstrates the prediction update scheduling functionality implemented in task 6.8.
Shows how to use the 2-second prediction update cycles as required by Requirement 3.5.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import List
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.predictive_agent import PredictiveAgent
from src.models import KPIMetrics, LoadForecast


def create_sample_kpi_data(num_points: int = 50) -> List[KPIMetrics]:
    """
    Create sample KPI data with realistic network patterns.
    
    Args:
        num_points: Number of data points to generate
        
    Returns:
        List of KPI metrics with time series patterns
    """
    base_time = datetime.now() - timedelta(seconds=num_points)
    kpis = []
    
    for i in range(num_points):
        # Create realistic network utilization pattern
        # Base load with daily pattern and some noise
        base_utilization = 40
        daily_pattern = 20 * np.sin(i * 0.1)  # Slow oscillation
        traffic_burst = 15 * np.sin(i * 0.5)  # Faster oscillation
        noise = np.random.normal(0, 3)
        
        utilization = base_utilization + daily_pattern + traffic_burst + noise
        utilization = max(5, min(95, utilization))  # Clamp to realistic range
        
        # Derive other metrics from utilization
        throughput = 100 + utilization * 2 + np.random.normal(0, 5)
        latency = 10 + utilization * 0.3 + np.random.normal(0, 1)
        packet_loss = max(0, (utilization - 70) * 0.1 + np.random.normal(0, 0.2))
        
        kpi = KPIMetrics(
            timestamp=base_time + timedelta(seconds=i),
            throughput=max(0, throughput),
            latency=max(1, latency),
            packet_loss=max(0, packet_loss),
            utilization=utilization,
            node_id=f"demo_node_{i % 3}"
        )
        kpis.append(kpi)
    
    return kpis


def prediction_callback(forecast: LoadForecast) -> None:
    """
    Callback function to handle scheduled predictions.
    
    Args:
        forecast: The generated load forecast
    """
    print(f"\n📊 Scheduled Prediction at {forecast.timestamp.strftime('%H:%M:%S')}")
    print(f"   Horizon: {forecast.prediction_horizon} seconds")
    print(f"   Predicted values: {[f'{v:.1f}' for v in forecast.predicted_values[:5]]}... (showing first 5)")
    print(f"   Confidence interval: ({forecast.confidence_interval[0]:.1f}, {forecast.confidence_interval[1]:.1f})")
    print(f"   Model accuracy (MAPE): {forecast.model_accuracy:.2f}%")
    print(f"   Meets requirement (<15%): {'✅' if forecast.model_accuracy < 15.0 else '❌'}")


def main():
    """
    Main demonstration of prediction scheduling functionality.
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Prediction Scheduling Demo - Task 6.8")
    print("=" * 50)
    
    # Create predictive agent
    print("\n1. Initializing Predictive Agent...")
    agent = PredictiveAgent(
        window_size_seconds=15,  # 10-20 second window (Requirement 3.1)
        prediction_horizon=10    # 10-second horizon (Requirement 3.2)
    )
    
    # Generate sample data
    print("2. Generating sample KPI data...")
    sample_data = create_sample_kpi_data(40)
    print(f"   Created {len(sample_data)} KPI data points")
    
    # Train models
    print("3. Training prediction models...")
    agent.analyze_historical_data(sample_data)
    
    if not agent.best_model_type:
        print("❌ No models were successfully trained with the sample data")
        print("   This can happen with synthetic data. Try running with real network data.")
        return
    
    print(f"   ✅ Best model selected: {agent.best_model_type}")
    
    # Show model status
    status = agent.get_model_status()
    print(f"   Available models: {status['available_models']}")
    print(f"   Data points: {status['data_points']}")
    
    # Register prediction callback
    print("\n4. Registering prediction callback...")
    agent.register_prediction_callback(prediction_callback)
    
    # Configure scheduler (using default 2-second interval per Requirement 3.5)
    print("5. Configuring prediction scheduler...")
    print(f"   Update interval: {agent.prediction_update_interval} seconds (Requirement 3.5)")
    
    # Start scheduler
    print("\n6. Starting prediction scheduler...")
    try:
        agent.start_prediction_scheduler()
        
        # Show scheduler status
        scheduler_status = agent.get_scheduler_status()
        print(f"   Scheduler enabled: {scheduler_status['enabled']}")
        print(f"   Scheduler running: {scheduler_status['running']}")
        print(f"   Update interval: {scheduler_status['update_interval']} seconds")
        
        print("\n7. Running scheduled predictions for 10 seconds...")
        print("   (Press Ctrl+C to stop early)")
        
        # Let scheduler run for demonstration
        start_time = time.time()
        try:
            while time.time() - start_time < 10.0:
                time.sleep(0.5)
                
                # Show live status every few seconds
                if int(time.time() - start_time) % 3 == 0:
                    current_status = agent.get_scheduler_status()
                    print(f"\n📈 Status Update:")
                    print(f"   Predictions generated: {current_status['predictions_generated']}")
                    if current_status['last_prediction']:
                        print(f"   Last prediction: {current_status['last_prediction']}")
                    
                    # Show accuracy monitoring status
                    accuracy_status = agent.get_accuracy_monitoring_status()
                    print(f"   Current accuracy: {accuracy_status['current_accuracy']:.2f}%")
                    print(f"   Meets requirement: {accuracy_status['meets_requirement']}")
        
        except KeyboardInterrupt:
            print("\n⏹️  Stopping early due to user interrupt...")
        
        # Stop scheduler
        print("\n8. Stopping prediction scheduler...")
        agent.stop_prediction_scheduler()
        
        # Final statistics
        final_status = agent.get_scheduler_status()
        print(f"\n📊 Final Statistics:")
        print(f"   Total predictions generated: {final_status['predictions_generated']}")
        print(f"   Scheduler enabled: {final_status['enabled']}")
        print(f"   Scheduler running: {final_status['running']}")
        
        # Show accuracy statistics if available
        accuracy_stats = agent.get_prediction_accuracy_statistics()
        if "error" not in accuracy_stats:
            print(f"\n🎯 Accuracy Statistics:")
            print(f"   Total measurements: {accuracy_stats['total_measurements']}")
            print(f"   Recent MAPE: {accuracy_stats['recent_mape']:.2f}%")
            print(f"   Best MAPE: {accuracy_stats['best_mape']:.2f}%")
            print(f"   Meets requirement: {accuracy_stats['recent_mape'] < 15.0}")
        
        print("\n✅ Prediction scheduling demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during scheduling demo: {e}")
        # Ensure scheduler is stopped
        try:
            agent.stop_prediction_scheduler()
        except:
            pass


if __name__ == "__main__":
    main()
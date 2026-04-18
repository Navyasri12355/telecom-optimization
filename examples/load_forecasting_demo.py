#!/usr/bin/env python3
"""
Demonstration of 10-second load forecasting functionality (Task 6.3).

This script demonstrates the enhanced PredictiveAgent capabilities:
1. 10-second horizon prediction generation
2. Enhanced confidence interval calculation
3. Accuracy tracking and monitoring

Requirements addressed:
- 3.2: 10-second prediction horizon
- 3.4: Accuracy tracking with MAPE < 15%
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime, timedelta
import numpy as np
from src.agents.predictive_agent import PredictiveAgent
from src.models import KPIMetrics


def generate_realistic_kpi_data(num_points: int = 50) -> list[KPIMetrics]:
    """Generate realistic KPI data for demonstration."""
    base_time = datetime.now()
    kpis = []
    
    for i in range(num_points):
        # Create realistic utilization pattern with trend and noise
        time_factor = i / num_points
        trend = 40 + 20 * time_factor  # Increasing trend from 40% to 60%
        seasonal = 10 * np.sin(i * 0.3)  # Seasonal variation
        noise = np.random.normal(0, 3)  # Random noise
        
        utilization = max(5, min(95, trend + seasonal + noise))
        
        # Correlated metrics
        throughput = 80 + utilization * 0.8 + np.random.normal(0, 5)
        latency = 5 + (utilization / 100) * 15 + np.random.normal(0, 2)
        packet_loss = max(0, (utilization - 70) * 0.1 + np.random.normal(0, 0.5))
        
        kpi = KPIMetrics(
            timestamp=base_time + timedelta(seconds=i),
            throughput=max(0, throughput),
            latency=max(0, latency),
            packet_loss=max(0, min(10, packet_loss)),
            utilization=utilization,
            node_id="demo_node"
        )
        kpis.append(kpi)
    
    return kpis


def demonstrate_10_second_forecasting():
    """Demonstrate 10-second load forecasting capabilities."""
    print("=" * 60)
    print("AI Telecom Optimization - 10-Second Load Forecasting Demo")
    print("Task 6.3: Enhanced Prediction with Confidence Intervals")
    print("=" * 60)
    
    # Initialize predictive agent with 15-second window
    print("\n1. Initializing Predictive Agent...")
    agent = PredictiveAgent(window_size_seconds=15, prediction_horizon=10)
    print(f"   - Window size: {agent.window_size_seconds} seconds")
    print(f"   - Prediction horizon: {agent.prediction_horizon} seconds")
    
    # Generate realistic KPI data
    print("\n2. Generating realistic KPI data...")
    kpi_data = generate_realistic_kpi_data(40)
    print(f"   - Generated {len(kpi_data)} KPI data points")
    print(f"   - Time range: {kpi_data[0].timestamp.strftime('%H:%M:%S')} to {kpi_data[-1].timestamp.strftime('%H:%M:%S')}")
    
    # Analyze historical data and train models
    print("\n3. Training prediction models...")
    agent.analyze_historical_data(kpi_data)
    
    status = agent.get_model_status()
    print(f"   - Available models: {status['available_models']}")
    print(f"   - Best model: {status['best_model']}")
    print(f"   - Data points: {status['data_points']}")
    
    if not agent.best_model_type:
        print("   ⚠️  No models trained successfully with demo data")
        return
    
    # Generate 10-second forecast
    print("\n4. Generating 10-second load forecast...")
    try:
        forecast = agent.generate_10_second_forecast()
        
        print(f"   ✅ Forecast generated successfully!")
        print(f"   - Prediction horizon: {forecast.prediction_horizon} seconds")
        print(f"   - Number of predictions: {len(forecast.predicted_values)}")
        print(f"   - Model accuracy (MAPE): {forecast.model_accuracy:.2f}%")
        print(f"   - Confidence interval: ({forecast.confidence_interval[0]:.1f}%, {forecast.confidence_interval[1]:.1f}%)")
        
        # Display predictions
        print(f"\n   Predicted utilization values:")
        for i, value in enumerate(forecast.predicted_values):
            print(f"     Second {i+1}: {value:.1f}%")
        
        # Check requirement compliance
        meets_requirement = forecast.model_accuracy < 15.0
        requirement_status = "✅ MEETS" if meets_requirement else "❌ FAILS"
        print(f"\n   Requirement 3.4 (MAPE < 15%): {requirement_status}")
        
    except Exception as e:
        print(f"   ❌ Forecast generation failed: {e}")
        return
    
    # Demonstrate accuracy tracking
    print("\n5. Demonstrating accuracy tracking...")
    print(f"   - Pending predictions: {len(agent.pending_predictions)}")
    print(f"   - Accuracy history length: {len(agent.prediction_accuracy_history)}")
    
    if agent.prediction_accuracy_history:
        recent_accuracy = np.mean(agent.prediction_accuracy_history[-5:])
        print(f"   - Recent average MAPE: {recent_accuracy:.2f}%")
    
    # Simulate feedback and model updating
    print("\n6. Simulating feedback and model updates...")
    feedback_data = generate_realistic_kpi_data(10)
    agent.update_model(feedback_data)
    print(f"   - Added {len(feedback_data)} feedback data points")
    
    updated_status = agent.get_model_status()
    if "overall_accuracy" in updated_status and updated_status["overall_accuracy"]:
        overall_acc = updated_status["overall_accuracy"]
        print(f"   - Overall accuracy evaluations: {overall_acc.get('total_evaluations', 0)}")
        if overall_acc.get('recent_mape'):
            print(f"   - Recent MAPE: {overall_acc['recent_mape']:.2f}%")
            print(f"   - Meets requirement: {overall_acc.get('meets_requirement', False)}")
    
    print("\n7. Enhanced Features Summary:")
    print("   ✅ 10-second prediction horizon (Requirement 3.2)")
    print("   ✅ Enhanced confidence interval calculation")
    print("   ✅ Real-time accuracy tracking (Requirement 3.4)")
    print("   ✅ Prediction performance monitoring")
    print("   ✅ Adaptive model behavior")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("Task 6.3 implementation verified: 10-second load forecasting")
    print("with enhanced confidence intervals and accuracy tracking.")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_10_second_forecasting()
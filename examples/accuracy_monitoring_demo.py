#!/usr/bin/env python3
"""
Accuracy Monitoring Demo for AI-driven Telecom Network Optimization

This demo showcases the enhanced prediction accuracy monitoring and model 
retraining capabilities implemented in task 6.6.

Features demonstrated:
1. MAPE calculation and accuracy tracking over time
2. Model retraining triggers for accuracy degradation
3. Accuracy trend analysis
4. Comprehensive accuracy statistics

Requirements addressed:
- 3.4: MAPE calculation with < 15% threshold
- 7.2: Adaptive model behavior with retraining triggers
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datetime import datetime, timedelta
from src.agents.predictive_agent import PredictiveAgent
from src.models import KPIMetrics
import time


def create_sample_kpi_data(num_points: int = 50, base_utilization: float = 50.0, 
                          noise_level: float = 5.0, trend: float = 0.0) -> list:
    """Create sample KPI data with configurable characteristics."""
    base_time = datetime.now() - timedelta(seconds=num_points)
    kpis = []
    
    for i in range(num_points):
        # Create utilization with trend and noise
        utilization = base_utilization + (trend * i) + np.random.normal(0, noise_level)
        utilization = max(0, min(100, utilization))  # Clamp to valid range
        
        kpi = KPIMetrics(
            timestamp=base_time + timedelta(seconds=i),
            throughput=100.0 + utilization * 2,
            latency=10.0 + utilization * 0.2,
            packet_loss=max(0, (utilization - 50) * 0.1),
            utilization=utilization,
            node_id=f"demo_node_{i % 3}"
        )
        kpis.append(kpi)
    
    return kpis


def demonstrate_accuracy_monitoring():
    """Demonstrate comprehensive accuracy monitoring functionality."""
    print("=" * 80)
    print("AI-DRIVEN TELECOM NETWORK OPTIMIZATION")
    print("Prediction Accuracy Monitoring Demo (Task 6.6)")
    print("=" * 80)
    
    # Initialize predictive agent with accuracy monitoring
    print("\n1. Initializing Predictive Agent with Accuracy Monitoring")
    print("-" * 60)
    
    agent = PredictiveAgent(
        window_size_seconds=15,
        prediction_horizon=10,
        model_selection_strategy="best_accuracy"
    )
    
    # Configure accuracy monitoring
    agent.configure_accuracy_monitoring(
        enabled=True,
        check_interval=10,  # Check every 10 seconds for demo
        retraining_threshold=15.0,  # MAPE threshold (Requirement 3.4)
        max_consecutive_poor=2  # Trigger retraining after 2 poor measurements
    )
    
    print(f"   ✅ Agent initialized with {agent.window_size_seconds}s window")
    print(f"   ✅ Accuracy monitoring enabled (threshold: {agent.retraining_threshold}%)")
    print(f"   ✅ Retraining triggers configured")
    
    # Create initial training data
    print("\n2. Training Initial Models")
    print("-" * 60)
    
    training_data = create_sample_kpi_data(
        num_points=30, 
        base_utilization=45.0, 
        noise_level=3.0,
        trend=0.2  # Slight upward trend
    )
    
    agent.analyze_historical_data(training_data)
    
    if agent.best_model_type:
        print(f"   ✅ Models trained successfully")
        print(f"   ✅ Best model selected: {agent.best_model_type}")
        print(f"   ✅ Available models: {list(agent.models.keys())}")
    else:
        print("   ❌ Model training failed - using synthetic accuracy data for demo")
        # Add synthetic accuracy history for demonstration
        agent.prediction_accuracy_history = [12.0, 14.0, 11.0, 13.0, 10.0]
    
    # Demonstrate accuracy monitoring status
    print("\n3. Accuracy Monitoring Status")
    print("-" * 60)
    
    monitoring_status = agent.get_accuracy_monitoring_status()
    print(f"   Monitoring Enabled: {monitoring_status['monitoring_enabled']}")
    print(f"   Check Interval: {monitoring_status['check_interval_seconds']}s")
    print(f"   Retraining Threshold: {monitoring_status['retraining_threshold']}%")
    print(f"   Current Accuracy: {monitoring_status['current_accuracy']:.2f}%" if monitoring_status['current_accuracy'] else "   Current Accuracy: Not available")
    print(f"   Meets Requirement: {monitoring_status['meets_requirement']}")
    print(f"   Consecutive Poor Count: {monitoring_status['consecutive_poor_count']}")
    
    # Simulate accuracy degradation scenario
    print("\n4. Simulating Accuracy Degradation Scenario")
    print("-" * 60)
    
    # Add poor accuracy measurements to trigger retraining
    poor_accuracy_measurements = [18.0, 20.0, 22.0, 25.0, 28.0]
    
    for i, mape in enumerate(poor_accuracy_measurements):
        print(f"\n   Simulation Step {i+1}: Adding MAPE measurement of {mape:.1f}%")
        
        # Add the measurement to history
        agent.prediction_accuracy_history.append(mape)
        
        # Force accuracy check
        monitoring_result = agent.force_accuracy_check()
        
        print(f"   Current Accuracy: {monitoring_result['current_accuracy']:.2f}%")
        print(f"   Meets Requirement (< 15%): {monitoring_result['meets_requirement']}")
        print(f"   Accuracy Trend: {monitoring_result['accuracy_trend']}")
        print(f"   Consecutive Poor Count: {monitoring_result['consecutive_poor_count']}")
        
        if monitoring_result['retraining_triggered']:
            print(f"   🔄 RETRAINING TRIGGERED: {monitoring_result['retraining_reason']}")
            print(f"   Actions Taken: {monitoring_result['actions_taken']}")
            break
        else:
            print(f"   ⏳ No retraining triggered yet")
    
    # Demonstrate accuracy trend analysis
    print("\n5. Accuracy Trend Analysis")
    print("-" * 60)
    
    # Test different trend scenarios
    trend_scenarios = [
        ("Improving Trend", [25.0, 22.0, 18.0, 15.0, 12.0, 10.0]),
        ("Degrading Trend", [8.0, 10.0, 13.0, 16.0, 20.0, 25.0]),
        ("Stable Trend", [12.0, 12.5, 11.8, 12.2, 11.9, 12.3])
    ]
    
    for scenario_name, measurements in trend_scenarios:
        # Temporarily set accuracy history
        original_history = agent.prediction_accuracy_history.copy()
        agent.prediction_accuracy_history = measurements
        
        trend = agent._analyze_accuracy_trend()
        print(f"   {scenario_name}: {trend}")
        
        # Restore original history
        agent.prediction_accuracy_history = original_history
    
    # Demonstrate comprehensive accuracy statistics
    print("\n6. Comprehensive Accuracy Statistics")
    print("-" * 60)
    
    stats = agent.get_accuracy_statistics()
    
    if "error" not in stats:
        print(f"   Total Measurements: {stats['total_measurements']}")
        print(f"   Current MAPE: {stats['current_mape']:.2f}%")
        print(f"   Recent MAPE (avg): {stats['recent_mape']:.2f}%")
        print(f"   Best MAPE: {stats['best_mape']:.2f}%")
        print(f"   Worst MAPE: {stats['worst_mape']:.2f}%")
        print(f"   Average MAPE: {stats['average_mape']:.2f}%")
        print(f"   Standard Deviation: {stats['std_mape']:.2f}%")
        print(f"   Meets Requirement %: {stats['meets_requirement_percentage']:.1f}%")
        print(f"   Accuracy Trend: {stats['trend']}")
        
        if "percentiles" in stats:
            print(f"   Percentiles:")
            print(f"     P25: {stats['percentiles']['p25']:.2f}%")
            print(f"     P50: {stats['percentiles']['p50']:.2f}%")
            print(f"     P75: {stats['percentiles']['p75']:.2f}%")
            print(f"     P90: {stats['percentiles']['p90']:.2f}%")
            print(f"     P95: {stats['percentiles']['p95']:.2f}%")
    else:
        print(f"   {stats['error']}")
    
    # Demonstrate retraining trigger conditions
    print("\n7. Retraining Trigger Conditions")
    print("-" * 60)
    
    trigger_scenarios = [
        ("Consecutive Poor Accuracy", 18.0, "stable", False, 3),
        ("Severe Degradation", 30.0, "stable", False, 0),
        ("Degrading Trend", 18.0, "degrading", False, 0),
        ("Good Performance", 10.0, "stable", True, 0)
    ]
    
    for scenario_name, accuracy, trend, meets_req, consecutive_count in trigger_scenarios:
        # Temporarily set consecutive count
        original_count = agent.consecutive_poor_accuracy_count
        agent.consecutive_poor_accuracy_count = consecutive_count
        
        should_retrain, reason = agent._should_trigger_retraining(accuracy, trend, meets_req)
        
        print(f"   {scenario_name}:")
        print(f"     MAPE: {accuracy}%, Trend: {trend}, Consecutive: {consecutive_count}")
        print(f"     Should Retrain: {should_retrain}")
        if reason:
            print(f"     Reason: {reason}")
        
        # Restore original count
        agent.consecutive_poor_accuracy_count = original_count
    
    # Final model status
    print("\n8. Final Model Status")
    print("-" * 60)
    
    final_status = agent.get_model_status()
    
    print(f"   Available Models: {final_status['available_models']}")
    print(f"   Best Model: {final_status['best_model']}")
    print(f"   Data Points: {final_status['data_points']}")
    print(f"   Pending Predictions: {final_status['pending_predictions']}")
    
    if "overall_accuracy" in final_status and final_status["overall_accuracy"]:
        accuracy_info = final_status["overall_accuracy"]
        print(f"   Overall Accuracy:")
        print(f"     Recent MAPE: {accuracy_info['recent_mape']:.2f}%")
        print(f"     Best MAPE: {accuracy_info['best_mape']:.2f}%")
        print(f"     Total Evaluations: {accuracy_info['total_evaluations']}")
        print(f"     Meets Requirement: {accuracy_info['meets_requirement']}")
    
    if "accuracy_monitoring" in final_status:
        monitoring_info = final_status["accuracy_monitoring"]
        print(f"   Accuracy Monitoring:")
        print(f"     Enabled: {monitoring_info['monitoring_enabled']}")
        print(f"     Threshold: {monitoring_info['retraining_threshold']}%")
        print(f"     Current Accuracy: {monitoring_info['current_accuracy']:.2f}%" if monitoring_info['current_accuracy'] else "     Current Accuracy: Not available")
    
    # Summary
    print("\n" + "=" * 80)
    print("ACCURACY MONITORING DEMO SUMMARY")
    print("=" * 80)
    print("\nImplemented Features (Task 6.6):")
    print("   ✅ MAPE calculation and accuracy tracking over time")
    print("   ✅ Model retraining triggers for accuracy degradation")
    print("   ✅ Accuracy trend analysis (improving/degrading/stable)")
    print("   ✅ Comprehensive accuracy statistics and reporting")
    print("   ✅ Configurable monitoring parameters")
    print("   ✅ Multiple retraining trigger conditions")
    print("   ✅ Integration with prediction workflow")
    
    print("\nRequirements Addressed:")
    print("   ✅ Requirement 3.4: MAPE calculation with < 15% threshold")
    print("   ✅ Requirement 7.2: Adaptive model behavior with retraining")
    print("   ✅ Enhanced accuracy monitoring and reporting")
    print("   ✅ Automated model maintenance and optimization")
    
    print("\nKey Benefits:")
    print("   • Proactive model quality management")
    print("   • Automated retraining for sustained performance")
    print("   • Comprehensive accuracy analytics")
    print("   • Configurable monitoring and alerting")
    print("   • Integration with existing prediction pipeline")


if __name__ == "__main__":
    try:
        demonstrate_accuracy_monitoring()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
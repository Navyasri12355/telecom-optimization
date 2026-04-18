#!/usr/bin/env python3
"""
Demonstration of enhanced conflict resolution mechanisms in OptimizationAgent.

This script shows how the OptimizationAgent handles conflicting decisions using
sophisticated business rules, priority boosting, and decision override logic.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime
from src.agents.optimization_agent import OptimizationAgent
from src.models import OptimizationDecision, ActionType


def demonstrate_conflict_resolution():
    """Demonstrate various conflict resolution scenarios."""
    
    print("=== AI Telecom Optimization: Enhanced Conflict Resolution Demo ===\n")
    
    # Initialize optimization agent
    agent = OptimizationAgent()
    
    print("1. EMERGENCY OVERRIDE SCENARIO")
    print("-" * 40)
    
    # Create emergency decision
    emergency_decision = OptimizationDecision(
        action_type=ActionType.INCREASE_CAPACITY,
        target_parameters={
            "emergency_threshold": 95.0,
            "bandwidth_multiplier": 2.0,
            "confidence_lower": 90.0,
            "confidence_upper": 98.0
        },
        rationale="Critical load spike detected - emergency capacity increase required",
        priority=100,  # Emergency priority
        timestamp=datetime.now()
    )
    
    # Create normal conflicting decisions
    normal_increase = OptimizationDecision(
        action_type=ActionType.INCREASE_CAPACITY,
        target_parameters={"bandwidth_multiplier": 1.3},
        rationale="Moderate load increase predicted",
        priority=60,
        timestamp=datetime.now()
    )
    
    normal_decrease = OptimizationDecision(
        action_type=ActionType.DECREASE_CAPACITY,
        target_parameters={"bandwidth_multiplier": 0.8},
        rationale="Energy saving opportunity detected",
        priority=30,
        timestamp=datetime.now()
    )
    
    # Resolve emergency conflict
    decisions = [normal_decrease, normal_increase, emergency_decision]
    resolved = agent.resolve_conflicts(decisions)
    
    print(f"Input decisions: {len(decisions)}")
    print(f"Resolved action: {resolved.action_type.value}")
    print(f"Resolved priority: {resolved.priority}")
    print(f"Rationale: {resolved.rationale[:100]}...")
    print(f"Emergency override: {'EMERGENCY OVERRIDE' in resolved.rationale}")
    
    print("\n2. PRIORITY BOOSTING SCENARIO")
    print("-" * 40)
    
    # Create high-confidence decision with lower base priority
    high_confidence = OptimizationDecision(
        action_type=ActionType.INCREASE_CAPACITY,
        target_parameters={
            "bandwidth_multiplier": 1.4,
            "confidence_lower": 88.0,
            "confidence_upper": 92.0  # Small range = high confidence
        },
        rationale="High confidence prediction of load increase",
        priority=45,  # Lower base priority
        timestamp=datetime.now()
    )
    
    # Create low-confidence decision with higher base priority
    low_confidence = OptimizationDecision(
        action_type=ActionType.DECREASE_CAPACITY,
        target_parameters={
            "bandwidth_multiplier": 0.7,
            "confidence_lower": 30.0,
            "confidence_upper": 70.0  # Large range = low confidence
        },
        rationale="Uncertain prediction suggests possible decrease",
        priority=65,  # Higher base priority
        timestamp=datetime.now()
    )
    
    decisions = [low_confidence, high_confidence]
    resolved = agent.resolve_conflicts(decisions)
    
    print(f"Input decisions: {len(decisions)}")
    print(f"Resolved action: {resolved.action_type.value}")
    print(f"Priority boosted: {'Priority boosted' in resolved.rationale}")
    print(f"Rationale: {resolved.rationale[:100]}...")
    
    print("\n3. RESOURCE CONSTRAINT FILTERING")
    print("-" * 40)
    
    # Create decision that violates resource constraints
    violating_decision = OptimizationDecision(
        action_type=ActionType.INCREASE_CAPACITY,
        target_parameters={
            "param1": 1.0, "param2": 2.0, "param3": 3.0, 
            "param4": 4.0, "param5": 5.0  # Too many concurrent adjustments
        },
        rationale="Aggressive capacity increase with many parameters",
        priority=80,
        timestamp=datetime.now()
    )
    
    # Create compliant decision
    compliant_decision = OptimizationDecision(
        action_type=ActionType.DECREASE_CAPACITY,
        target_parameters={"bandwidth_multiplier": 0.8},
        rationale="Simple, compliant capacity adjustment",
        priority=40,  # Lower priority
        timestamp=datetime.now()
    )
    
    decisions = [violating_decision, compliant_decision]
    resolved = agent.resolve_conflicts(decisions)
    
    print(f"Input decisions: {len(decisions)}")
    print(f"Resolved action: {resolved.action_type.value}")
    print(f"Resource constraint applied: {resolved.priority < violating_decision.priority}")
    print(f"Rationale: {resolved.rationale[:100]}...")
    
    print("\n4. DECISION OVERRIDE FUNCTIONALITY")
    print("-" * 40)
    
    # Create a normal decision
    original_decision = OptimizationDecision(
        action_type=ActionType.INCREASE_CAPACITY,
        target_parameters={"bandwidth_multiplier": 1.5},
        rationale="Standard capacity increase based on load prediction",
        priority=60,
        timestamp=datetime.now()
    )
    
    # Override the decision
    override_decision = agent.override_decision(
        original_decision,
        "System maintenance window active - preventing capacity changes"
    )
    
    print(f"Original action: {original_decision.action_type.value}")
    print(f"Override action: {override_decision.action_type.value}")
    print(f"Override priority: {override_decision.priority}")
    print(f"Override rationale: {override_decision.rationale[:100]}...")
    
    print("\n5. BUSINESS RULE COMPLIANCE EVALUATION")
    print("-" * 40)
    
    # Test compliant decision
    compliant = OptimizationDecision(
        action_type=ActionType.INCREASE_CAPACITY,
        target_parameters={"bandwidth_multiplier": 1.3},  # Within limits
        rationale="Compliant capacity increase",
        priority=60,
        timestamp=datetime.now()
    )
    
    compliance = agent.evaluate_business_rule_compliance(compliant)
    print(f"Compliant decision: {compliance['compliant']}")
    print(f"Violations: {len(compliance['violations'])}")
    
    # Test non-compliant decision
    non_compliant = OptimizationDecision(
        action_type=ActionType.INCREASE_CAPACITY,
        target_parameters={"bandwidth_multiplier": 3.0},  # Exceeds max of 2.0
        rationale="Excessive capacity increase",
        priority=60,
        timestamp=datetime.now()
    )
    
    compliance = agent.evaluate_business_rule_compliance(non_compliant)
    print(f"Non-compliant decision: {compliance['compliant']}")
    print(f"Violations: {len(compliance['violations'])}")
    if compliance['violations']:
        print(f"First violation: {compliance['violations'][0]}")
    
    print("\n6. CONFLICT RESOLUTION METRICS")
    print("-" * 40)
    
    # Generate some conflict resolution activity
    test_decisions = [
        OptimizationDecision(ActionType.INCREASE_CAPACITY, {"bandwidth_multiplier": 1.2}, "Test 1", 50, datetime.now()),
        OptimizationDecision(ActionType.DECREASE_CAPACITY, {"bandwidth_multiplier": 0.8}, "Test 2", 40, datetime.now())
    ]
    
    resolved = agent.resolve_conflicts(test_decisions)
    agent.log_actions(
        resolved, 
        "Demo conflict resolution",
        execution_context={
            "conflict_resolution": {
                "conflict_type": "opposite_type_conflicts",
                "resolution_strategy": "priority_order",
                "original_decision_count": 2
            }
        }
    )
    
    metrics = agent.get_conflict_resolution_metrics()
    print(f"Total decisions logged: {metrics['total_decisions']}")
    print(f"Conflict resolutions: {metrics['conflict_resolutions']}")
    print(f"Conflict resolution rate: {metrics['conflict_resolution_rate']:.2%}")
    
    print("\n7. BUSINESS RULES CONFIGURATION")
    print("-" * 40)
    
    status = agent.get_agent_status()
    print("Current business rules:")
    for rule_category, rules in status['business_rules'].items():
        print(f"  {rule_category}:")
        if isinstance(rules, dict):
            for key, value in rules.items():
                print(f"    {key}: {value}")
        else:
            print(f"    {rules}")
    
    print("\n=== Demo Complete ===")
    print("\nKey Features Demonstrated:")
    print("✓ Emergency decision override")
    print("✓ Priority boosting based on confidence")
    print("✓ Resource constraint filtering")
    print("✓ Manual decision override")
    print("✓ Business rule compliance evaluation")
    print("✓ Conflict resolution metrics")
    print("✓ Comprehensive business rules configuration")


if __name__ == "__main__":
    demonstrate_conflict_resolution()
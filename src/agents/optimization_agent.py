"""
Optimization Agent for AI-driven telecom network optimization system.

This agent implements load-based capacity adjustment decisions using predictions
from the PredictiveAgent to optimize network performance through dynamic parameter
adjustments.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import deque
import json

from src.interfaces import OptimizationAgentInterface
from src.models import (
    LoadForecast, OptimizationDecision, NetworkParameters, ActionType
)


class OptimizationAgent(OptimizationAgentInterface):
    """
    Optimization agent that makes load-based capacity adjustment decisions.
    
    Implements Requirements 4.1, 4.2, 4.4, and 4.5 for automated capacity
    adjustment based on predicted load thresholds.
    """
    
    def __init__(self, 
                 high_load_threshold: float = 80.0,
                 low_load_threshold: float = 30.0,
                 capacity_adjustment_factor: float = 0.2):
        """
        Initialize the optimization agent.
        
        Args:
            high_load_threshold: Load percentage threshold for capacity increase (default 80%)
            low_load_threshold: Load percentage threshold for capacity decrease (default 30%)
            capacity_adjustment_factor: Factor for capacity adjustments (default 20%)
        """
        self.logger = logging.getLogger(__name__)
        
        # Load thresholds as per Requirements 4.1 and 4.2
        self.high_load_threshold = high_load_threshold
        self.low_load_threshold = low_load_threshold
        self.capacity_adjustment_factor = capacity_adjustment_factor
        
        # Action logging storage (Requirement 4.4)
        self.action_log: deque = deque(maxlen=1000)  # Keep recent actions
        
        # Business rules for conflict resolution (Requirement 4.5)
        self.business_rules = {
            "priority_order": [ActionType.INCREASE_CAPACITY, ActionType.DECREASE_CAPACITY, ActionType.NO_ACTION],
            "max_capacity_increase": 2.0,  # Maximum 2x capacity increase
            "min_capacity_decrease": 0.5,  # Minimum 50% capacity retention
            "cooling_period_seconds": 30,  # Minimum time between major adjustments
            "emergency_threshold": 95.0,   # Emergency capacity increase threshold
            
            # Enhanced conflict resolution rules
            "priority_boost_factors": {
                "emergency_multiplier": 2.0,  # Boost priority for emergency conditions
                "confidence_multiplier": 1.5,  # Boost priority for high-confidence predictions
                "trend_multiplier": 1.3,  # Boost priority for consistent trend predictions
            },
            "decision_override_rules": {
                "emergency_overrides_all": True,  # Emergency decisions override all others
                "high_confidence_threshold": 0.8,  # Confidence threshold for priority boost
                "trend_consistency_threshold": 3,  # Number of consistent predictions for trend boost
                "resource_constraint_override": True,  # Override decisions that violate resource constraints
            },
            "conflict_resolution_strategies": {
                "same_type_conflicts": "highest_priority",  # How to resolve conflicts of same action type
                "opposite_type_conflicts": "priority_order",  # How to resolve opposite action conflicts
                "mixed_type_conflicts": "weighted_priority",  # How to resolve mixed conflicts
                "tie_breaking": "most_recent",  # How to break priority ties
            },
            "resource_constraints": {
                "max_concurrent_adjustments": 3,  # Maximum number of simultaneous parameter adjustments
                "min_stability_period": 10,  # Minimum seconds between conflicting actions
                "resource_utilization_limit": 0.9,  # Maximum resource utilization before override
            }
        }
        
        # Track recent decisions for cooling period enforcement
        self.recent_decisions: deque = deque(maxlen=10)
        
        # Default network parameter ranges
        self.parameter_ranges = {
            "bandwidth": {"min": 1.0, "max": 1000.0, "unit": "Mbps"},
            "queue_size": {"min": 10, "max": 10000, "unit": "packets"},
            "scheduling_weight": {"min": 0.1, "max": 10.0, "unit": "weight"}
        }
        
        self.logger.info(f"OptimizationAgent initialized with thresholds: "
                        f"high={high_load_threshold}%, low={low_load_threshold}%, "
                        f"adjustment_factor={capacity_adjustment_factor}")
    
    def evaluate_predictions(self, forecast: LoadForecast) -> OptimizationDecision:
        """
        Evaluate load predictions and make optimization decisions.
        
        Implements load threshold evaluation as required by Requirements 4.1 and 4.2:
        - When predicted load exceeds 80%: increase capacity
        - When predicted load falls below 30%: decrease capacity
        - Otherwise: no action
        
        Args:
            forecast: Load forecast from PredictiveAgent
            
        Returns:
            OptimizationDecision with action type and rationale
        """
        if not forecast or not forecast.predicted_values:
            return self._create_no_action_decision("No forecast data available")
        
        # Calculate average predicted load over the forecast horizon
        avg_predicted_load = sum(forecast.predicted_values) / len(forecast.predicted_values)
        max_predicted_load = max(forecast.predicted_values)
        
        # Check for emergency conditions first
        if max_predicted_load >= self.business_rules["emergency_threshold"]:
            return self._create_emergency_decision(max_predicted_load, forecast)
        
        # Apply load-based decision logic (Requirements 4.1 and 4.2)
        if avg_predicted_load > self.high_load_threshold:
            return self._create_increase_capacity_decision(avg_predicted_load, forecast)
        elif avg_predicted_load < self.low_load_threshold:
            return self._create_decrease_capacity_decision(avg_predicted_load, forecast)
        else:
            return self._create_no_action_decision(
                f"Load {avg_predicted_load:.1f}% within acceptable range "
                f"({self.low_load_threshold}%-{self.high_load_threshold}%)"
            )
    
    def adjust_capacity(self, decision: OptimizationDecision) -> NetworkParameters:
        """
        Generate network parameter adjustments based on optimization decision.
        
        Implements capacity adjustment logic for bandwidth, queue size, and 
        scheduling parameters as required by Requirement 4.3.
        
        Args:
            decision: Optimization decision to implement
            
        Returns:
            NetworkParameters with adjusted values
        """
        current_time = datetime.now()
        
        # Extract target parameters from decision
        target_params = decision.target_parameters
        
        # Create network parameters based on action type
        if decision.action_type == ActionType.INCREASE_CAPACITY:
            network_params = self._generate_capacity_increase_parameters(target_params)
        elif decision.action_type == ActionType.DECREASE_CAPACITY:
            network_params = self._generate_capacity_decrease_parameters(target_params)
        else:  # NO_ACTION
            network_params = self._generate_no_change_parameters()
        
        # Set update timestamp
        network_params.update_timestamp = current_time
        
        # Validate parameters are within acceptable ranges
        self._validate_network_parameters(network_params)
        
        self.logger.info(f"Generated network parameters for {decision.action_type.value}: "
                        f"bandwidth adjustments={len(network_params.bandwidth)}, "
                        f"queue adjustments={len(network_params.queue_size)}")
        
        return network_params
    
    def log_actions(self, decision: OptimizationDecision, rationale: str, 
                   network_params: Optional[NetworkParameters] = None,
                   execution_context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log optimization actions with comprehensive details.
        
        Implements comprehensive action logging as required by Requirements 4.4 and 6.4.
        Logs all parameter changes with timestamps and rationale as specified in Property 12.
        
        Args:
            decision: The optimization decision that was made
            rationale: Additional rationale for the decision
            network_params: The actual network parameters that were applied (optional)
            execution_context: Additional context about the execution environment (optional)
        """
        current_time = datetime.now()
        
        # Build comprehensive log entry
        log_entry = {
            # Core action information (Property 12 requirements)
            "log_timestamp": current_time.isoformat(),
            "action_type": decision.action_type.value,
            "decision_timestamp": decision.timestamp.isoformat(),
            "priority": decision.priority,
            
            # Rationale and reasoning
            "original_rationale": decision.rationale,
            "additional_rationale": rationale,
            
            # Parameter information (Requirement 4.4)
            "target_parameters": decision.target_parameters.copy(),
            "parameters_changed": self._extract_parameter_changes(decision.target_parameters),
            
            # Execution context (Requirement 6.4 - comprehensive logging)
            "execution_context": execution_context or {},
            
            # Agent state at time of action
            "agent_state": {
                "high_load_threshold": self.high_load_threshold,
                "low_load_threshold": self.low_load_threshold,
                "capacity_adjustment_factor": self.capacity_adjustment_factor,
                "recent_decisions_count": len(self.recent_decisions),
                "action_log_size": len(self.action_log)
            }
        }
        
        # Add actual network parameters if provided
        if network_params:
            log_entry["applied_network_parameters"] = {
                "bandwidth_changes": network_params.bandwidth.copy(),
                "queue_size_changes": network_params.queue_size.copy(),
                "scheduling_changes": network_params.scheduling_algorithm.copy(),
                "parameter_update_timestamp": network_params.update_timestamp.isoformat()
            }
            
            # Calculate parameter deltas if we have previous parameters
            if hasattr(self, '_last_network_params') and self._last_network_params:
                log_entry["parameter_deltas"] = self._calculate_parameter_deltas(
                    self._last_network_params, network_params
                )
            
            # Store for next delta calculation
            self._last_network_params = network_params
        
        # Add performance metrics if available in execution context
        if execution_context and "performance_metrics" in execution_context:
            log_entry["performance_impact"] = execution_context["performance_metrics"]
        
        # Add conflict resolution information if this was a resolved decision
        if "conflict_resolution" in (execution_context or {}):
            log_entry["conflict_resolution"] = execution_context["conflict_resolution"]
        
        # Generate unique action ID for tracking
        action_id = f"{decision.action_type.value}_{current_time.strftime('%Y%m%d_%H%M%S_%f')}"
        log_entry["action_id"] = action_id
        
        # Add to action log with size management
        self.action_log.append(log_entry)
        
        # Structured logging to system logger (multiple levels for different audiences)
        self.logger.info(
            f"OPTIMIZATION_ACTION [{action_id}]: {decision.action_type.value} - {rationale}"
        )
        
        # Detailed debug logging with full context
        self.logger.debug(
            f"OPTIMIZATION_ACTION_DETAIL [{action_id}]: "
            f"Priority={decision.priority}, "
            f"Target_Params={len(decision.target_parameters)}, "
            f"Applied_Params={'Yes' if network_params else 'No'}, "
            f"Context={'Yes' if execution_context else 'No'}"
        )
        
        # Parameter-specific logging for audit trails (Requirement 4.4)
        if network_params:
            for param_type, changes in [
                ("bandwidth", network_params.bandwidth),
                ("queue_size", network_params.queue_size),
                ("scheduling", network_params.scheduling_algorithm)
            ]:
                if changes:
                    self.logger.info(
                        f"PARAMETER_CHANGE [{action_id}]: {param_type.upper()} - "
                        f"{len(changes)} nodes affected"
                    )
                    for node_id, value in changes.items():
                        self.logger.debug(
                            f"PARAMETER_DETAIL [{action_id}]: {param_type}[{node_id}] = {value}"
                        )
        
        # Track for cooling period enforcement
        self.recent_decisions.append({
            "timestamp": current_time,
            "action_type": decision.action_type,
            "priority": decision.priority,
            "action_id": action_id
        })
        
        # Log action completion
        self.logger.debug(f"OPTIMIZATION_ACTION_COMPLETE [{action_id}]: Logged successfully")
    
    def _extract_parameter_changes(self, target_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and categorize parameter changes from target parameters.
        
        Args:
            target_parameters: Target parameters from optimization decision
            
        Returns:
            Dictionary categorizing the types of changes
        """
        changes = {
            "capacity_changes": {},
            "performance_targets": {},
            "prediction_context": {},
            "business_rules_applied": []
        }
        
        # Categorize different types of parameter changes
        for key, value in target_parameters.items():
            if "multiplier" in key:
                changes["capacity_changes"][key] = value
            elif key in ["predicted_load", "confidence_lower", "confidence_upper"]:
                changes["prediction_context"][key] = value
            elif key in ["load_excess", "load_deficit", "emergency_threshold"]:
                changes["performance_targets"][key] = value
        
        # Identify which business rules were applied
        if "emergency_threshold" in target_parameters:
            changes["business_rules_applied"].append("emergency_response")
        if "bandwidth_multiplier" in target_parameters:
            if target_parameters["bandwidth_multiplier"] > 1.0:
                changes["business_rules_applied"].append("capacity_increase")
            elif target_parameters["bandwidth_multiplier"] < 1.0:
                changes["business_rules_applied"].append("capacity_decrease")
        
        return changes
    
    def _calculate_parameter_deltas(self, previous_params: NetworkParameters, 
                                  current_params: NetworkParameters) -> Dict[str, Any]:
        """
        Calculate the differences between previous and current network parameters.
        
        Args:
            previous_params: Previous network parameters
            current_params: Current network parameters
            
        Returns:
            Dictionary with parameter deltas
        """
        deltas = {
            "bandwidth_deltas": {},
            "queue_size_deltas": {},
            "scheduling_changes": {},
            "summary": {
                "bandwidth_nodes_changed": 0,
                "queue_nodes_changed": 0,
                "scheduling_nodes_changed": 0,
                "total_changes": 0
            }
        }
        
        # Calculate bandwidth deltas
        for node_id in set(previous_params.bandwidth.keys()) | set(current_params.bandwidth.keys()):
            prev_bw = previous_params.bandwidth.get(node_id, 0)
            curr_bw = current_params.bandwidth.get(node_id, 0)
            if prev_bw != curr_bw:
                deltas["bandwidth_deltas"][node_id] = {
                    "previous": prev_bw,
                    "current": curr_bw,
                    "delta": curr_bw - prev_bw,
                    "percent_change": ((curr_bw - prev_bw) / prev_bw * 100) if prev_bw > 0 else 0
                }
                deltas["summary"]["bandwidth_nodes_changed"] += 1
        
        # Calculate queue size deltas
        for node_id in set(previous_params.queue_size.keys()) | set(current_params.queue_size.keys()):
            prev_queue = previous_params.queue_size.get(node_id, 0)
            curr_queue = current_params.queue_size.get(node_id, 0)
            if prev_queue != curr_queue:
                deltas["queue_size_deltas"][node_id] = {
                    "previous": prev_queue,
                    "current": curr_queue,
                    "delta": curr_queue - prev_queue,
                    "percent_change": ((curr_queue - prev_queue) / prev_queue * 100) if prev_queue > 0 else 0
                }
                deltas["summary"]["queue_nodes_changed"] += 1
        
        # Calculate scheduling changes
        for node_id in set(previous_params.scheduling_algorithm.keys()) | set(current_params.scheduling_algorithm.keys()):
            prev_sched = previous_params.scheduling_algorithm.get(node_id, "")
            curr_sched = current_params.scheduling_algorithm.get(node_id, "")
            if prev_sched != curr_sched:
                deltas["scheduling_changes"][node_id] = {
                    "previous": prev_sched,
                    "current": curr_sched
                }
                deltas["summary"]["scheduling_nodes_changed"] += 1
        
        # Calculate total changes
        deltas["summary"]["total_changes"] = (
            deltas["summary"]["bandwidth_nodes_changed"] +
            deltas["summary"]["queue_nodes_changed"] +
            deltas["summary"]["scheduling_nodes_changed"]
        )
        
        return deltas
    
    def resolve_conflicts(self, decisions: List[OptimizationDecision]) -> OptimizationDecision:
        """
        Resolve conflicts between multiple optimization decisions using enhanced business rules.
        
        Implements sophisticated conflict resolution using predefined business rules as 
        required by Requirement 4.5, including decision override logic and priority boosting.
        
        Args:
            decisions: List of conflicting optimization decisions
            
        Returns:
            Single resolved OptimizationDecision with conflict resolution context
        """
        if not decisions:
            return self._create_no_action_decision("No decisions to resolve")
        
        if len(decisions) == 1:
            return decisions[0]
        
        self.logger.info(f"Resolving conflicts between {len(decisions)} decisions using enhanced business rules")
        
        # Step 1: Apply decision override rules for emergency conditions
        emergency_decisions = self._filter_emergency_decisions(decisions)
        if emergency_decisions and self.business_rules["decision_override_rules"]["emergency_overrides_all"]:
            self.logger.info("Emergency override activated - selecting emergency decision")
            return self._resolve_emergency_conflicts(emergency_decisions, decisions)
        
        # Step 2: Apply priority boosting based on confidence and trends
        enhanced_decisions = self._apply_priority_boosting(decisions)
        
        # Step 3: Check resource constraints and apply overrides if needed
        if self.business_rules["decision_override_rules"]["resource_constraint_override"]:
            constraint_filtered_decisions = self._apply_resource_constraint_overrides(enhanced_decisions)
        else:
            constraint_filtered_decisions = enhanced_decisions
        
        # Step 4: Apply conflict resolution strategy based on decision types
        conflict_type = self._classify_conflict_type(constraint_filtered_decisions)
        resolution_strategy = self.business_rules["conflict_resolution_strategies"][conflict_type]
        
        # Step 5: Resolve conflicts using the determined strategy
        resolved_decision = self._apply_resolution_strategy(
            constraint_filtered_decisions, resolution_strategy, conflict_type
        )
        
        # Step 6: Check cooling period for major adjustments
        if self._is_in_cooling_period(resolved_decision.action_type):
            return self._create_no_action_decision(
                f"Cooling period active for {resolved_decision.action_type.value}, "
                f"original decision: {resolved_decision.rationale}"
            )
        
        # Step 7: Create final resolved decision with comprehensive context
        final_decision = self._create_resolved_decision(
            resolved_decision, decisions, conflict_type, resolution_strategy
        )
        
        self.logger.info(f"Conflict resolved using {resolution_strategy} strategy: "
                        f"selected {final_decision.action_type.value} from {len(decisions)} decisions")
        
        return final_decision
    
    def _filter_emergency_decisions(self, decisions: List[OptimizationDecision]) -> List[OptimizationDecision]:
        """Filter decisions that are emergency-level (priority >= 100)."""
        return [d for d in decisions if d.priority >= 100]
    
    def _resolve_emergency_conflicts(self, emergency_decisions: List[OptimizationDecision], 
                                   all_decisions: List[OptimizationDecision]) -> OptimizationDecision:
        """Resolve conflicts when emergency decisions are present."""
        # Among emergency decisions, select the one with highest priority
        best_emergency = max(emergency_decisions, key=lambda d: d.priority)
        
        # Create resolved decision with emergency override context
        return OptimizationDecision(
            action_type=best_emergency.action_type,
            target_parameters=best_emergency.target_parameters.copy(),
            rationale=f"EMERGENCY OVERRIDE: {best_emergency.rationale} "
                     f"(overrode {len(all_decisions) - 1} other decisions)",
            priority=best_emergency.priority + 10,  # Boost emergency priority further
            timestamp=datetime.now()
        )
    
    def _apply_priority_boosting(self, decisions: List[OptimizationDecision]) -> List[OptimizationDecision]:
        """Apply priority boosting based on confidence, trends, and other factors."""
        enhanced_decisions = []
        boost_factors = self.business_rules["priority_boost_factors"]
        override_rules = self.business_rules["decision_override_rules"]
        
        for decision in decisions:
            enhanced_priority = decision.priority
            boost_reasons = []
            
            # Boost for high confidence predictions
            if "confidence_upper" in decision.target_parameters:
                confidence_range = (decision.target_parameters.get("confidence_upper", 0) - 
                                  decision.target_parameters.get("confidence_lower", 0))
                if confidence_range > 0:
                    confidence_level = 1.0 - (confidence_range / 100.0)  # Higher confidence = smaller range
                    if confidence_level >= override_rules["high_confidence_threshold"]:
                        enhanced_priority *= boost_factors["confidence_multiplier"]
                        boost_reasons.append(f"high_confidence({confidence_level:.2f})")
            
            # Boost for emergency conditions
            if "emergency_threshold" in decision.target_parameters:
                enhanced_priority *= boost_factors["emergency_multiplier"]
                boost_reasons.append("emergency_condition")
            
            # Boost for trend consistency (simulated - would use historical data in real implementation)
            if decision.action_type != ActionType.NO_ACTION:
                # Check if recent decisions show consistent trend
                consistent_trend = self._check_trend_consistency(decision.action_type)
                if consistent_trend >= override_rules["trend_consistency_threshold"]:
                    enhanced_priority *= boost_factors["trend_multiplier"]
                    boost_reasons.append(f"consistent_trend({consistent_trend})")
            
            # Create enhanced decision
            enhanced_decision = OptimizationDecision(
                action_type=decision.action_type,
                target_parameters=decision.target_parameters.copy(),
                rationale=decision.rationale + (f" [Priority boosted: {', '.join(boost_reasons)}]" if boost_reasons else ""),
                priority=int(enhanced_priority),
                timestamp=decision.timestamp
            )
            
            enhanced_decisions.append(enhanced_decision)
        
        return enhanced_decisions
    
    def _check_trend_consistency(self, action_type: ActionType) -> int:
        """Check how many recent decisions were of the same action type."""
        if not self.recent_decisions:
            return 0
        
        consistent_count = 0
        for recent in reversed(list(self.recent_decisions)):  # Check most recent first
            if recent["action_type"] == action_type:
                consistent_count += 1
            else:
                break  # Stop at first different action type
        
        return consistent_count
    
    def _apply_resource_constraint_overrides(self, decisions: List[OptimizationDecision]) -> List[OptimizationDecision]:
        """Apply resource constraint overrides to filter out decisions that violate constraints."""
        constraints = self.business_rules["resource_constraints"]
        filtered_decisions = []
        
        for decision in decisions:
            # Check if decision violates resource constraints
            violates_constraints = False
            
            # Check maximum concurrent adjustments
            if len(decision.target_parameters) > constraints["max_concurrent_adjustments"]:
                self.logger.warning(f"Decision violates max_concurrent_adjustments: "
                                  f"{len(decision.target_parameters)} > {constraints['max_concurrent_adjustments']}")
                violates_constraints = True
            
            # Check resource utilization limits (simulated check)
            if "bandwidth_multiplier" in decision.target_parameters:
                multiplier = decision.target_parameters["bandwidth_multiplier"]
                if multiplier > constraints["resource_utilization_limit"] + 1.0:  # +1.0 because multiplier is relative
                    self.logger.warning(f"Decision violates resource_utilization_limit: "
                                      f"multiplier {multiplier} too high")
                    violates_constraints = True
            
            # Check stability period
            if self._violates_stability_period(decision.action_type, constraints["min_stability_period"]):
                self.logger.warning(f"Decision violates min_stability_period for {decision.action_type.value}")
                violates_constraints = True
            
            if not violates_constraints:
                filtered_decisions.append(decision)
            else:
                self.logger.info(f"Filtered out decision due to resource constraints: {decision.rationale}")
        
        # If all decisions were filtered out, return a no-action decision
        if not filtered_decisions:
            filtered_decisions = [self._create_no_action_decision(
                "All decisions filtered out due to resource constraints"
            )]
        
        return filtered_decisions
    
    def _violates_stability_period(self, action_type: ActionType, min_stability_seconds: int) -> bool:
        """Check if the action type violates the minimum stability period."""
        if not self.recent_decisions:
            return False
        
        current_time = datetime.now()
        
        # Look for recent opposite actions
        opposite_actions = {
            ActionType.INCREASE_CAPACITY: ActionType.DECREASE_CAPACITY,
            ActionType.DECREASE_CAPACITY: ActionType.INCREASE_CAPACITY,
            ActionType.NO_ACTION: None
        }
        
        opposite_action = opposite_actions.get(action_type)
        if not opposite_action:
            return False
        
        # Check if there was a recent opposite action within the stability period
        for recent in self.recent_decisions:
            if recent["action_type"] == opposite_action:
                time_diff = (current_time - recent["timestamp"]).total_seconds()
                if time_diff < min_stability_seconds:
                    return True
        
        return False
    
    def _classify_conflict_type(self, decisions: List[OptimizationDecision]) -> str:
        """Classify the type of conflict based on decision action types."""
        action_types = set(d.action_type for d in decisions)
        
        if len(action_types) == 1:
            return "same_type_conflicts"
        elif (ActionType.INCREASE_CAPACITY in action_types and 
              ActionType.DECREASE_CAPACITY in action_types and 
              len(action_types) == 2):
            return "opposite_type_conflicts"
        else:
            return "mixed_type_conflicts"
    
    def _apply_resolution_strategy(self, decisions: List[OptimizationDecision], 
                                 strategy: str, conflict_type: str) -> OptimizationDecision:
        """Apply the specified resolution strategy to resolve conflicts."""
        if strategy == "highest_priority":
            return max(decisions, key=lambda d: d.priority)
        
        elif strategy == "priority_order":
            # Use business rule priority order
            priority_order = self.business_rules["priority_order"]
            for action_type in priority_order:
                matching_decisions = [d for d in decisions if d.action_type == action_type]
                if matching_decisions:
                    return max(matching_decisions, key=lambda d: d.priority)
            
            # Fallback to highest priority if no match found
            return max(decisions, key=lambda d: d.priority)
        
        elif strategy == "weighted_priority":
            # Apply weighted priority based on action type importance and decision priority
            action_weights = {
                ActionType.INCREASE_CAPACITY: 3.0,
                ActionType.DECREASE_CAPACITY: 2.0,
                ActionType.NO_ACTION: 1.0
            }
            
            best_decision = None
            best_weighted_score = -1
            
            for decision in decisions:
                weight = action_weights.get(decision.action_type, 1.0)
                weighted_score = decision.priority * weight
                
                if weighted_score > best_weighted_score:
                    best_weighted_score = weighted_score
                    best_decision = decision
            
            return best_decision or decisions[0]
        
        elif strategy == "most_recent":
            return max(decisions, key=lambda d: d.timestamp)
        
        else:
            # Default fallback
            self.logger.warning(f"Unknown resolution strategy: {strategy}, using highest_priority")
            return max(decisions, key=lambda d: d.priority)
    
    def _create_resolved_decision(self, base_decision: OptimizationDecision, 
                                original_decisions: List[OptimizationDecision],
                                conflict_type: str, resolution_strategy: str) -> OptimizationDecision:
        """Create the final resolved decision with comprehensive conflict resolution context."""
        # Build detailed rationale
        conflict_summary = f"Resolved {conflict_type} using {resolution_strategy} strategy"
        original_rationales = [d.rationale[:50] + "..." if len(d.rationale) > 50 else d.rationale 
                             for d in original_decisions]
        
        enhanced_rationale = (
            f"{conflict_summary}: {base_decision.rationale} "
            f"(selected from {len(original_decisions)} decisions: "
            f"{'; '.join(original_rationales)})"
        )
        
        # Add conflict resolution metadata to target parameters
        enhanced_target_parameters = base_decision.target_parameters.copy()
        enhanced_target_parameters.update({
            "conflict_resolution_metadata": {
                "original_decision_count": len(original_decisions),
                "conflict_type": conflict_type,
                "resolution_strategy": resolution_strategy,
                "original_priorities": [d.priority for d in original_decisions],
                "resolution_timestamp": datetime.now().isoformat()
            }
        })
        
        return OptimizationDecision(
            action_type=base_decision.action_type,
            target_parameters=enhanced_target_parameters,
            rationale=enhanced_rationale,
            priority=base_decision.priority + 1,  # Slight boost for resolved decisions
            timestamp=datetime.now()
        )
    
    def _create_increase_capacity_decision(self, predicted_load: float, 
                                         forecast: LoadForecast) -> OptimizationDecision:
        """Create decision to increase network capacity."""
        # Calculate adjustment magnitude based on load excess
        load_excess = predicted_load - self.high_load_threshold
        adjustment_magnitude = min(
            self.capacity_adjustment_factor * (load_excess / 20.0),  # Scale by excess
            self.business_rules["max_capacity_increase"] - 1.0  # Respect maximum
        )
        
        target_parameters = {
            "bandwidth_multiplier": 1.0 + adjustment_magnitude,
            "queue_size_multiplier": 1.0 + (adjustment_magnitude * 0.5),  # Smaller queue adjustment
            "predicted_load": predicted_load,
            "load_excess": load_excess,
            "confidence_lower": forecast.confidence_interval[0],
            "confidence_upper": forecast.confidence_interval[1]
        }
        
        rationale = (f"Predicted load {predicted_load:.1f}% exceeds threshold "
                    f"{self.high_load_threshold}%, increasing capacity by "
                    f"{adjustment_magnitude*100:.1f}%")
        
        return OptimizationDecision(
            action_type=ActionType.INCREASE_CAPACITY,
            target_parameters=target_parameters,
            rationale=rationale,
            priority=self._calculate_priority(predicted_load, ActionType.INCREASE_CAPACITY),
            timestamp=datetime.now()
        )
    
    def _create_decrease_capacity_decision(self, predicted_load: float, 
                                         forecast: LoadForecast) -> OptimizationDecision:
        """Create decision to decrease network capacity for energy saving."""
        # Calculate adjustment magnitude based on load deficit
        load_deficit = self.low_load_threshold - predicted_load
        adjustment_magnitude = min(
            self.capacity_adjustment_factor * (load_deficit / 20.0),  # Scale by deficit
            1.0 - self.business_rules["min_capacity_decrease"]  # Respect minimum
        )
        
        target_parameters = {
            "bandwidth_multiplier": 1.0 - adjustment_magnitude,
            "queue_size_multiplier": 1.0 - (adjustment_magnitude * 0.3),  # Smaller queue adjustment
            "predicted_load": predicted_load,
            "load_deficit": load_deficit,
            "confidence_lower": forecast.confidence_interval[0],
            "confidence_upper": forecast.confidence_interval[1]
        }
        
        rationale = (f"Predicted load {predicted_load:.1f}% below threshold "
                    f"{self.low_load_threshold}%, decreasing capacity by "
                    f"{adjustment_magnitude*100:.1f}% for energy saving")
        
        return OptimizationDecision(
            action_type=ActionType.DECREASE_CAPACITY,
            target_parameters=target_parameters,
            rationale=rationale,
            priority=self._calculate_priority(predicted_load, ActionType.DECREASE_CAPACITY),
            timestamp=datetime.now()
        )
    
    def _create_emergency_decision(self, max_load: float, 
                                 forecast: LoadForecast) -> OptimizationDecision:
        """Create emergency capacity increase decision."""
        emergency_multiplier = min(
            self.business_rules["max_capacity_increase"],
            1.0 + (max_load - self.business_rules["emergency_threshold"]) / 100.0
        )
        
        target_parameters = {
            "bandwidth_multiplier": emergency_multiplier,
            "queue_size_multiplier": emergency_multiplier * 0.8,
            "predicted_load": max_load,
            "emergency_threshold": self.business_rules["emergency_threshold"],
            "confidence_lower": forecast.confidence_interval[0],
            "confidence_upper": forecast.confidence_interval[1]
        }
        
        rationale = (f"EMERGENCY: Predicted load {max_load:.1f}% exceeds emergency "
                    f"threshold {self.business_rules['emergency_threshold']}%, "
                    f"applying maximum capacity increase")
        
        return OptimizationDecision(
            action_type=ActionType.INCREASE_CAPACITY,
            target_parameters=target_parameters,
            rationale=rationale,
            priority=100,  # Highest priority for emergency
            timestamp=datetime.now()
        )
    
    def _create_no_action_decision(self, rationale: str) -> OptimizationDecision:
        """Create decision for no action."""
        return OptimizationDecision(
            action_type=ActionType.NO_ACTION,
            target_parameters={},
            rationale=rationale,
            priority=1,  # Lowest priority
            timestamp=datetime.now()
        )
    
    def _calculate_priority(self, predicted_load: float, action_type: ActionType) -> int:
        """Calculate decision priority based on load and action type."""
        if action_type == ActionType.INCREASE_CAPACITY:
            # Higher priority for higher loads
            excess = predicted_load - self.high_load_threshold
            return min(50 + int(excess), 99)  # Cap at 99 (emergency uses 100)
        elif action_type == ActionType.DECREASE_CAPACITY:
            # Higher priority for lower loads (more energy savings)
            deficit = self.low_load_threshold - predicted_load
            return min(20 + int(deficit), 40)  # Lower priority than increases
        else:
            return 1  # Lowest priority for no action
    
    def _generate_capacity_increase_parameters(self, target_params: Dict[str, float]) -> NetworkParameters:
        """Generate network parameters for capacity increase."""
        bandwidth_multiplier = target_params.get("bandwidth_multiplier", 1.2)
        queue_multiplier = target_params.get("queue_size_multiplier", 1.1)
        
        # Default network topology parameters (would be provided by NetworkSimulator in real implementation)
        default_bandwidth = {"link_ue1_enodeb": 50.0, "link_enodeb_core": 100.0, 
                           "link_core_server": 100.0, "link_server_ue2": 50.0}
        default_queues = {"ue1": 100, "enodeb": 500, "core_router": 1000, "server": 500, "ue2": 100}
        default_scheduling = {"enodeb": "WFQ", "core_router": "WFQ", "server": "FIFO"}
        
        # Apply capacity increases
        new_bandwidth = {
            link_id: bw * bandwidth_multiplier 
            for link_id, bw in default_bandwidth.items()
        }
        
        new_queues = {
            node_id: int(size * queue_multiplier)
            for node_id, size in default_queues.items()
        }
        
        return NetworkParameters(
            bandwidth=new_bandwidth,
            queue_size=new_queues,
            scheduling_algorithm=default_scheduling,
            update_timestamp=datetime.now()
        )
    
    def _generate_capacity_decrease_parameters(self, target_params: Dict[str, float]) -> NetworkParameters:
        """Generate network parameters for capacity decrease."""
        bandwidth_multiplier = target_params.get("bandwidth_multiplier", 0.8)
        queue_multiplier = target_params.get("queue_size_multiplier", 0.9)
        
        # Default network topology parameters
        default_bandwidth = {"link_ue1_enodeb": 50.0, "link_enodeb_core": 100.0, 
                           "link_core_server": 100.0, "link_server_ue2": 50.0}
        default_queues = {"ue1": 100, "enodeb": 500, "core_router": 1000, "server": 500, "ue2": 100}
        default_scheduling = {"enodeb": "FIFO", "core_router": "FIFO", "server": "FIFO"}  # Simpler scheduling for energy saving
        
        # Apply capacity decreases
        new_bandwidth = {
            link_id: max(bw * bandwidth_multiplier, self.parameter_ranges["bandwidth"]["min"])
            for link_id, bw in default_bandwidth.items()
        }
        
        new_queues = {
            node_id: max(int(size * queue_multiplier), self.parameter_ranges["queue_size"]["min"])
            for node_id, size in default_queues.items()
        }
        
        return NetworkParameters(
            bandwidth=new_bandwidth,
            queue_size=new_queues,
            scheduling_algorithm=default_scheduling,
            update_timestamp=datetime.now()
        )
    
    def _generate_no_change_parameters(self) -> NetworkParameters:
        """Generate network parameters with no changes."""
        # Return current default parameters
        default_bandwidth = {"link_ue1_enodeb": 50.0, "link_enodeb_core": 100.0, 
                           "link_core_server": 100.0, "link_server_ue2": 50.0}
        default_queues = {"ue1": 100, "enodeb": 500, "core_router": 1000, "server": 500, "ue2": 100}
        default_scheduling = {"enodeb": "WFQ", "core_router": "WFQ", "server": "FIFO"}
        
        return NetworkParameters(
            bandwidth=default_bandwidth,
            queue_size=default_queues,
            scheduling_algorithm=default_scheduling,
            update_timestamp=datetime.now()
        )
    
    def _validate_network_parameters(self, params: NetworkParameters) -> None:
        """Validate network parameters are within acceptable ranges."""
        # Validate bandwidth values
        for link_id, bw in params.bandwidth.items():
            min_bw = self.parameter_ranges["bandwidth"]["min"]
            max_bw = self.parameter_ranges["bandwidth"]["max"]
            if not (min_bw <= bw <= max_bw):
                raise ValueError(f"Bandwidth {bw} for {link_id} outside range [{min_bw}, {max_bw}]")
        
        # Validate queue sizes
        for node_id, size in params.queue_size.items():
            min_size = self.parameter_ranges["queue_size"]["min"]
            max_size = self.parameter_ranges["queue_size"]["max"]
            if not (min_size <= size <= max_size):
                raise ValueError(f"Queue size {size} for {node_id} outside range [{min_size}, {max_size}]")
    
    def _is_in_cooling_period(self, action_type: ActionType) -> bool:
        """Check if action type is in cooling period."""
        if not self.recent_decisions:
            return False
        
        current_time = datetime.now()
        cooling_period = self.business_rules["cooling_period_seconds"]
        
        # Check for recent major adjustments of the same type
        for decision in self.recent_decisions:
            if (decision["action_type"] == action_type and 
                decision["action_type"] != ActionType.NO_ACTION):
                time_diff = (current_time - decision["timestamp"]).total_seconds()
                if time_diff < cooling_period:
                    return True
        
        return False
    
    def override_decision(self, original_decision: OptimizationDecision, 
                         override_reason: str, override_priority: int = 200) -> OptimizationDecision:
        """
        Override a decision with manual intervention or system-level override.
        
        Implements decision override logic as required by Requirement 4.5.
        
        Args:
            original_decision: The original decision to override
            override_reason: Reason for the override
            override_priority: Priority for the override decision (default: 200 - very high)
            
        Returns:
            New OptimizationDecision with override context
        """
        # Create override decision with NO_ACTION to prevent the original action
        override_decision = OptimizationDecision(
            action_type=ActionType.NO_ACTION,
            target_parameters={
                "override_metadata": {
                    "original_action": original_decision.action_type.value,
                    "original_priority": original_decision.priority,
                    "original_rationale": original_decision.rationale,
                    "override_reason": override_reason,
                    "override_timestamp": datetime.now().isoformat(),
                    "override_authority": "optimization_agent_business_rules"
                }
            },
            rationale=f"DECISION OVERRIDE: {override_reason} "
                     f"(overrode {original_decision.action_type.value}: {original_decision.rationale})",
            priority=override_priority,
            timestamp=datetime.now()
        )
        
        # Log the override action
        self.logger.warning(f"DECISION_OVERRIDE: {override_reason} - "
                          f"overrode {original_decision.action_type.value} decision")
        
        # Add to action log with special override context
        self.log_actions(
            override_decision, 
            f"Business rule override: {override_reason}",
            execution_context={
                "override_action": True,
                "original_decision": {
                    "action_type": original_decision.action_type.value,
                    "priority": original_decision.priority,
                    "rationale": original_decision.rationale
                }
            }
        )
        
        return override_decision
    
    def evaluate_business_rule_compliance(self, decision: OptimizationDecision) -> Dict[str, Any]:
        """
        Evaluate if a decision complies with all business rules.
        
        Args:
            decision: Decision to evaluate
            
        Returns:
            Dictionary with compliance status and any violations
        """
        compliance_report = {
            "compliant": True,
            "violations": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Check capacity limits
        if "bandwidth_multiplier" in decision.target_parameters:
            multiplier = decision.target_parameters["bandwidth_multiplier"]
            
            if decision.action_type == ActionType.INCREASE_CAPACITY:
                max_increase = self.business_rules["max_capacity_increase"]
                if multiplier > max_increase:
                    compliance_report["compliant"] = False
                    compliance_report["violations"].append(
                        f"Bandwidth multiplier {multiplier} exceeds maximum {max_increase}"
                    )
            
            elif decision.action_type == ActionType.DECREASE_CAPACITY:
                min_retention = self.business_rules["min_capacity_decrease"]
                if multiplier < min_retention:
                    compliance_report["compliant"] = False
                    compliance_report["violations"].append(
                        f"Bandwidth multiplier {multiplier} below minimum {min_retention}"
                    )
        
        # Check cooling period compliance
        if self._is_in_cooling_period(decision.action_type):
            compliance_report["warnings"].append(
                f"Decision may violate cooling period for {decision.action_type.value}"
            )
        
        # Check resource constraints
        constraints = self.business_rules["resource_constraints"]
        
        if len(decision.target_parameters) > constraints["max_concurrent_adjustments"]:
            compliance_report["violations"].append(
                f"Too many concurrent adjustments: {len(decision.target_parameters)} > "
                f"{constraints['max_concurrent_adjustments']}"
            )
            compliance_report["compliant"] = False
        
        # Check for stability period violations
        if self._violates_stability_period(decision.action_type, constraints["min_stability_period"]):
            compliance_report["warnings"].append(
                f"Decision may violate stability period for {decision.action_type.value}"
            )
        
        # Add recommendations based on current system state
        if decision.priority < 50 and decision.action_type != ActionType.NO_ACTION:
            compliance_report["recommendations"].append(
                "Consider increasing decision priority for non-trivial actions"
            )
        
        if not decision.target_parameters and decision.action_type != ActionType.NO_ACTION:
            compliance_report["recommendations"].append(
                "Action decisions should include target parameters"
            )
        
        return compliance_report
    
    def get_conflict_resolution_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about conflict resolution performance.
        
        Returns:
            Dictionary with conflict resolution statistics
        """
        # Analyze action log for conflict resolution patterns
        conflict_resolutions = []
        total_decisions = 0
        
        for entry in self.action_log:
            if "conflict_resolution" in entry.get("execution_context", {}):
                conflict_resolutions.append(entry)
            
            if entry.get("log_entry_type") != "system_state_change":
                total_decisions += 1
        
        # Calculate metrics
        metrics = {
            "total_decisions": total_decisions,
            "conflict_resolutions": len(conflict_resolutions),
            "conflict_resolution_rate": len(conflict_resolutions) / max(total_decisions, 1),
            "business_rules": self.business_rules.copy(),
            "recent_conflicts": []
        }
        
        # Analyze recent conflict patterns
        for resolution in conflict_resolutions[-5:]:  # Last 5 conflicts
            conflict_info = resolution.get("execution_context", {}).get("conflict_resolution", {})
            metrics["recent_conflicts"].append({
                "timestamp": resolution.get("log_timestamp"),
                "conflict_type": conflict_info.get("conflict_type", "unknown"),
                "resolution_strategy": conflict_info.get("resolution_strategy", "unknown"),
                "original_decision_count": conflict_info.get("original_decision_count", 0),
                "final_action": resolution.get("action_type")
            })
        
        return metrics
    
    def get_action_log(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get recent action log entries.
        
        Args:
            limit: Maximum number of entries to return (None for all)
            
        Returns:
            List of action log entries
        """
        if limit is None:
            return list(self.action_log)
        else:
            return list(self.action_log)[-limit:]
    
    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get current status of the optimization agent.
        
        Returns:
            Dictionary with agent status information
        """
        current_time = datetime.now()
        
        status = {
            "thresholds": {
                "high_load": self.high_load_threshold,
                "low_load": self.low_load_threshold,
                "emergency": self.business_rules["emergency_threshold"]
            },
            "capacity_adjustment_factor": self.capacity_adjustment_factor,
            "action_log_entries": len(self.action_log),
            "recent_decisions": len(self.recent_decisions),
            "business_rules": self.business_rules.copy(),
            "parameter_ranges": self.parameter_ranges.copy(),
            "last_action": None,
            "cooling_periods_active": {}
        }
        
        # Add last action information
        if self.action_log:
            last_action = self.action_log[-1]
            # Handle both optimization actions and system state changes
            if "action_type" in last_action:
                status["last_action"] = {
                    "timestamp": last_action.get("log_timestamp", last_action.get("timestamp", "N/A")),
                    "action_type": last_action["action_type"],
                    "rationale": last_action.get("original_rationale", "N/A")
                }
            elif "change_type" in last_action:
                status["last_action"] = {
                    "timestamp": last_action.get("timestamp", "N/A"),
                    "action_type": f"system_state_change:{last_action['change_type']}",
                    "rationale": last_action.get("change_details", {}).get("summary", "System state change")
                }
            else:
                status["last_action"] = {
                    "timestamp": "N/A",
                    "action_type": "unknown",
                    "rationale": "Unknown log entry type"
                }
        
        # Check cooling periods for each action type
        for action_type in ActionType:
            status["cooling_periods_active"][action_type.value] = self._is_in_cooling_period(action_type)
        
        return status
    
    def configure_thresholds(self, 
                           high_load: Optional[float] = None,
                           low_load: Optional[float] = None,
                           adjustment_factor: Optional[float] = None) -> None:
        """
        Configure load thresholds and adjustment parameters.
        
        Args:
            high_load: High load threshold percentage (80% default)
            low_load: Low load threshold percentage (30% default)
            adjustment_factor: Capacity adjustment factor (0.2 default)
        """
        if high_load is not None:
            if not (50 <= high_load <= 95):
                raise ValueError("High load threshold must be between 50% and 95%")
            self.high_load_threshold = high_load
        
        if low_load is not None:
            if not (5 <= low_load <= 50):
                raise ValueError("Low load threshold must be between 5% and 50%")
            self.low_load_threshold = low_load
        
        if adjustment_factor is not None:
            if not (0.05 <= adjustment_factor <= 1.0):
                raise ValueError("Adjustment factor must be between 0.05 and 1.0")
            self.capacity_adjustment_factor = adjustment_factor
        
        # Validate threshold relationship
        if self.low_load_threshold >= self.high_load_threshold:
            raise ValueError("Low load threshold must be less than high load threshold")
        
        self.logger.info(f"Thresholds configured: high={self.high_load_threshold}%, "
                        f"low={self.low_load_threshold}%, factor={self.capacity_adjustment_factor}")
    
    def clear_action_log(self) -> int:
        """
        Clear the action log and return number of entries cleared.
        
        Returns:
            Number of log entries that were cleared
        """
        count = len(self.action_log)
        self.action_log.clear()
        self.recent_decisions.clear()
        
        # Clear stored network parameters for delta calculation
        if hasattr(self, '_last_network_params'):
            self._last_network_params = None
        
        self.logger.info(f"Cleared {count} action log entries")
        return count
    
    def export_action_log(self, format_type: str = "json", 
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> str:
        """
        Export action log in specified format for audit and analysis.
        
        Supports comprehensive logging requirements for audit trails and decision tracking.
        
        Args:
            format_type: Export format ("json", "csv", "summary")
            start_time: Filter logs from this time (optional)
            end_time: Filter logs until this time (optional)
            
        Returns:
            Formatted log data as string
        """
        # Filter logs by time range if specified
        filtered_logs = list(self.action_log)
        if start_time or end_time:
            filtered_logs = []
            for log_entry in self.action_log:
                log_time = datetime.fromisoformat(log_entry["log_timestamp"])
                if start_time and log_time < start_time:
                    continue
                if end_time and log_time > end_time:
                    continue
                filtered_logs.append(log_entry)
        
        if format_type == "json":
            return json.dumps(filtered_logs, indent=2, default=str)
        
        elif format_type == "csv":
            if not filtered_logs:
                return "No log entries found"
            
            # Create CSV with key fields
            import csv
            from io import StringIO
            
            output = StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                "Timestamp", "Action_ID", "Action_Type", "Priority", 
                "Rationale", "Parameters_Changed", "Bandwidth_Changes", 
                "Queue_Changes", "Scheduling_Changes"
            ])
            
            # Write data rows
            for entry in filtered_logs:
                writer.writerow([
                    entry["log_timestamp"],
                    entry.get("action_id", "N/A"),
                    entry["action_type"],
                    entry["priority"],
                    entry["original_rationale"],
                    len(entry.get("parameters_changed", {}).get("capacity_changes", {})),
                    len(entry.get("applied_network_parameters", {}).get("bandwidth_changes", {})),
                    len(entry.get("applied_network_parameters", {}).get("queue_size_changes", {})),
                    len(entry.get("applied_network_parameters", {}).get("scheduling_changes", {}))
                ])
            
            return output.getvalue()
        
        elif format_type == "summary":
            if not filtered_logs:
                return "No log entries found for summary"
            
            # Generate summary statistics
            action_counts = {}
            total_parameter_changes = 0
            priority_distribution = {}
            
            for entry in filtered_logs:
                action_type = entry["action_type"]
                action_counts[action_type] = action_counts.get(action_type, 0) + 1
                
                priority = entry["priority"]
                priority_distribution[priority] = priority_distribution.get(priority, 0) + 1
                
                if "parameter_deltas" in entry:
                    total_parameter_changes += entry["parameter_deltas"]["summary"]["total_changes"]
            
            summary = f"""Action Log Summary
=================
Time Range: {start_time or 'Beginning'} to {end_time or 'End'}
Total Actions: {len(filtered_logs)}
Total Parameter Changes: {total_parameter_changes}

Action Type Distribution:
{json.dumps(action_counts, indent=2)}

Priority Distribution:
{json.dumps(priority_distribution, indent=2)}

Recent Actions (last 5):
"""
            
            # Add recent actions
            for entry in filtered_logs[-5:]:
                summary += f"- {entry['log_timestamp']}: {entry['action_type']} (Priority: {entry['priority']})\n"
            
            return summary
        
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def get_action_audit_trail(self, action_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed audit trail for specific action or recent actions.
        
        Provides comprehensive action tracking for compliance and debugging.
        
        Args:
            action_id: Specific action ID to retrieve (None for recent actions)
            
        Returns:
            Dictionary with audit trail information
        """
        if action_id:
            # Find specific action
            for entry in self.action_log:
                if entry.get("action_id") == action_id:
                    return {
                        "action_found": True,
                        "action_details": entry,
                        "related_actions": self._find_related_actions(entry)
                    }
            
            return {"action_found": False, "message": f"Action ID {action_id} not found"}
        
        else:
            # Return recent actions with audit information
            recent_actions = list(self.action_log)[-10:]  # Last 10 actions
            
            audit_trail = {
                "recent_actions_count": len(recent_actions),
                "total_actions_logged": len(self.action_log),
                "actions": []
            }
            
            for entry in recent_actions:
                audit_entry = {
                    "action_id": entry.get("action_id"),
                    "timestamp": entry["log_timestamp"],
                    "action_type": entry["action_type"],
                    "priority": entry["priority"],
                    "rationale": entry["original_rationale"],
                    "parameters_affected": len(entry.get("parameters_changed", {}).get("capacity_changes", {})),
                    "has_network_params": "applied_network_parameters" in entry,
                    "has_deltas": "parameter_deltas" in entry,
                    "context_provided": bool(entry.get("execution_context"))
                }
                audit_trail["actions"].append(audit_entry)
            
            return audit_trail
    
    def _find_related_actions(self, target_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find actions related to the target action (same time period, similar parameters, etc.).
        
        Args:
            target_entry: The action entry to find related actions for
            
        Returns:
            List of related action entries
        """
        related_actions = []
        target_time = datetime.fromisoformat(target_entry["log_timestamp"])
        
        # Look for actions within 5 minutes of target action
        time_window = 300  # 5 minutes in seconds
        
        for entry in self.action_log:
            if entry.get("action_id") == target_entry.get("action_id"):
                continue  # Skip the target action itself
            
            entry_time = datetime.fromisoformat(entry["log_timestamp"])
            time_diff = abs((entry_time - target_time).total_seconds())
            
            if time_diff <= time_window:
                # Check for parameter overlap
                target_params = set(target_entry.get("target_parameters", {}).keys())
                entry_params = set(entry.get("target_parameters", {}).keys())
                
                if target_params & entry_params:  # If there's any parameter overlap
                    related_actions.append({
                        "action_id": entry.get("action_id"),
                        "timestamp": entry["log_timestamp"],
                        "action_type": entry["action_type"],
                        "time_difference_seconds": time_diff,
                        "parameter_overlap": list(target_params & entry_params),
                        "relationship_type": "temporal_and_parameter"
                    })
                else:
                    related_actions.append({
                        "action_id": entry.get("action_id"),
                        "timestamp": entry["log_timestamp"],
                        "action_type": entry["action_type"],
                        "time_difference_seconds": time_diff,
                        "relationship_type": "temporal_only"
                    })
        
        return related_actions
    
    def log_system_state_change(self, change_type: str, change_details: Dict[str, Any],
                               context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log system state changes for comprehensive system monitoring.
        
        Implements Requirement 6.4 for comprehensive logging of system state changes.
        
        Args:
            change_type: Type of system state change
            change_details: Details about the change
            context: Additional context information
        """
        state_change_entry = {
            "timestamp": datetime.now().isoformat(),
            "change_type": change_type,
            "change_details": change_details.copy(),
            "context": context or {},
            "agent_id": "optimization_agent",
            "agent_state_snapshot": {
                "thresholds": {
                    "high_load": self.high_load_threshold,
                    "low_load": self.low_load_threshold
                },
                "recent_actions": len(self.recent_decisions),
                "total_actions_logged": len(self.action_log)
            }
        }
        
        # Log to system logger with structured format
        self.logger.info(
            f"SYSTEM_STATE_CHANGE: {change_type} - "
            f"{change_details.get('summary', 'State change occurred')}"
        )
        
        self.logger.debug(
            f"SYSTEM_STATE_DETAIL: {json.dumps(state_change_entry, indent=2, default=str)}"
        )
        
        # Store in action log with special marker
        state_change_entry["log_entry_type"] = "system_state_change"
        self.action_log.append(state_change_entry)
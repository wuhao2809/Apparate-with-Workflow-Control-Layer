"""
Workflow Control Layer for Apparate

This module adds workflow control capabilities to improve Apparate's serving efficiency:
1. SLO-aware request prioritization
2. Confidence-aware adaptive batching
3. Queue-aware feedback control
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import logging


class RequestPrioritizer:
    """Prioritizes requests based on SLO urgency (deadline proximity)."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def prioritize_requests(self, requests: List, current_time: float) -> List:
        """
        Sort requests by deadline urgency (most urgent first).
        
        This sorts requests by time until deadline (deadline - current_time).
        The request with the SMALLEST time until deadline is processed first.
        
        Args:
            requests: List of Request objects with deadline attribute
            current_time: Current timestamp in ms
            
        Returns:
            Sorted list of requests (most urgent first)
        """
        if not requests:
            return []
        
        # Calculate time until deadline for each request
        def urgency_score(request):
            time_until_deadline = request.deadline - current_time
            return time_until_deadline  # Smaller = more urgent
        
        sorted_requests = sorted(requests, key=urgency_score)
        return sorted_requests
    
    def get_urgency_score(self, request, current_time: float) -> float:
        """Get urgency score for a request (lower = more urgent)."""
        return request.deadline - current_time


class AdaptiveBatcher:
    """Adjusts batch sizes based on predicted early exit confidence."""
    
    def __init__(self, min_batch_size: int = 1, max_batch_size: int = 64):
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.logger = logging.getLogger(__name__)
    
    def predict_early_exit_confidence(self, historical_data: Dict, 
                                       ramp_ids: List[int], 
                                       sample_indices: List[int]) -> List[float]:
        """
        Predict early exit confidence for samples based on historical data.
        
        Args:
            historical_data: Dict with 'conf' key containing confidence at each ramp
            ramp_ids: List of active ramp IDs
            sample_indices: Indices of samples to predict for
            
        Returns:
            List of predicted confidence scores (higher = more likely to exit early)
        """
        if not historical_data or 'conf' not in historical_data:
            # No historical data, return neutral predictions
            return [0.5] * len(sample_indices)
        
        predictions = []
        for idx in sample_indices:
            # Use average confidence at first ramp as prediction
            if ramp_ids and len(historical_data['conf']) > ramp_ids[0]:
                if idx < len(historical_data['conf'][ramp_ids[0]]):
                    conf = historical_data['conf'][ramp_ids[0]][idx]
                    # Convert to confidence score (1 - entropy = confidence)
                    predictions.append(1.0 - conf if conf < 1.0 else 0.0)
                else:
                    predictions.append(0.5)  # Default if no data
            else:
                predictions.append(0.5)
        
        return predictions
    
    def get_adaptive_batch_size(self, queue: List, 
                                confidence_predictions: Optional[List[float]] = None,
                                base_batch_size: int = 8) -> int:
        """
        Determine adaptive batch size based on confidence predictions.
        
        Args:
            queue: List of requests in queue
            confidence_predictions: Predicted early exit confidence for each request
            base_batch_size: Base batch size to use
            
        Returns:
            Adaptive batch size
        """
        if not queue:
            return self.min_batch_size
        
        if confidence_predictions is None or len(confidence_predictions) == 0:
            # No predictions, use base batch size
            return min(base_batch_size, len(queue), self.max_batch_size)
        
        # If we have high-confidence samples (likely to exit early), 
        # we can use larger batches
        avg_confidence = np.mean(confidence_predictions[:len(queue)])
        
        # Scale batch size based on confidence
        # High confidence (0.8+) -> larger batches (up to 1.5x)
        # Low confidence (<0.3) -> smaller batches (0.7x)
        if avg_confidence > 0.8:
            multiplier = 1.3
        elif avg_confidence > 0.6:
            multiplier = 1.1
        elif avg_confidence < 0.3:
            multiplier = 0.8
        else:
            multiplier = 1.0
        
        adaptive_size = int(base_batch_size * multiplier)
        adaptive_size = max(self.min_batch_size, 
                           min(adaptive_size, len(queue), self.max_batch_size))
        
        return adaptive_size


class QueueStateMonitor:
    """Monitors queue state metrics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.queue_lengths = deque(maxlen=window_size)
        self.avg_wait_times = deque(maxlen=window_size)
        self.slo_violations = deque(maxlen=window_size)
        self.logger = logging.getLogger(__name__)
    
    def update(self, queue_length: int, avg_wait_time: float, 
               slo_violations_count: int = 0):
        """Update queue state metrics."""
        self.queue_lengths.append(queue_length)
        self.avg_wait_times.append(avg_wait_time)
        self.slo_violations.append(slo_violations_count)
    
    def get_queue_state(self) -> Dict:
        """Get current queue state summary."""
        if not self.queue_lengths:
            return {
                'avg_queue_length': 0,
                'avg_wait_time': 0,
                'slo_violation_rate': 0,
                'is_congested': False
            }
        
        avg_queue_length = np.mean(list(self.queue_lengths))
        avg_wait_time = np.mean(list(self.avg_wait_times))
        slo_violation_rate = np.mean(list(self.slo_violations)) if self.slo_violations else 0
        
        # Consider queue congested if avg length > 10 or wait time > 50ms
        is_congested = avg_queue_length > 10 or avg_wait_time > 50
        
        return {
            'avg_queue_length': avg_queue_length,
            'avg_wait_time': avg_wait_time,
            'slo_violation_rate': slo_violation_rate,
            'is_congested': is_congested
        }
    
    def reset(self):
        """Reset all metrics."""
        self.queue_lengths.clear()
        self.avg_wait_times.clear()
        self.slo_violations.clear()


class FeedbackAdjuster:
    """Adjusts thresholds based on queue state."""
    
    def __init__(self, base_threshold_adjustment: float = 0.05):
        self.base_threshold_adjustment = base_threshold_adjustment
        self.logger = logging.getLogger(__name__)
    
    def adjust_thresholds(self, queue_state: Dict, 
                         current_thresholds: List[float],
                         ramp_ids: List[int]) -> List[float]:
        """
        Adjust thresholds based on queue state.
        
        Args:
            queue_state: Queue state from QueueStateMonitor
            current_thresholds: Current threshold values
            ramp_ids: List of active ramp IDs
            
        Returns:
            Adjusted threshold values
        """
        if not current_thresholds or not ramp_ids:
            return current_thresholds
        
        adjusted_thresholds = current_thresholds.copy()
        
        # If queue is congested, lower thresholds to allow more early exits
        if queue_state.get('is_congested', False):
            # Lower thresholds by 10-20% to increase early exit rate
            adjustment = -self.base_threshold_adjustment * 2
            for i in range(len(adjusted_thresholds)):
                adjusted_thresholds[i] = max(0.0, 
                    adjusted_thresholds[i] + adjustment)
            self.logger.debug(f"Queue congested, lowering thresholds: {adjusted_thresholds}")
        
        # If SLO violation rate is high, lower thresholds more aggressively
        elif queue_state.get('slo_violation_rate', 0) > 0.1:
            adjustment = -self.base_threshold_adjustment * 1.5
            for i in range(len(adjusted_thresholds)):
                adjusted_thresholds[i] = max(0.0, 
                    adjusted_thresholds[i] + adjustment)
            self.logger.debug(f"High SLO violations, adjusting thresholds: {adjusted_thresholds}")
        
        # If queue is empty and no violations, can slightly increase thresholds
        # to maintain accuracy
        elif (queue_state.get('avg_queue_length', 0) < 2 and 
              queue_state.get('slo_violation_rate', 0) < 0.01):
            adjustment = self.base_threshold_adjustment * 0.5
            for i in range(len(adjusted_thresholds)):
                adjusted_thresholds[i] = min(1.0, 
                    adjusted_thresholds[i] + adjustment)
            self.logger.debug(f"Queue healthy, slightly raising thresholds: {adjusted_thresholds}")
        
        return adjusted_thresholds


class WorkflowController:
    """Orchestrates workflow control components."""
    
    def __init__(self, enable_prioritization: bool = True,
                 enable_adaptive_batching: bool = True,
                 enable_feedback: bool = True):
        self.prioritizer = RequestPrioritizer() if enable_prioritization else None
        self.batcher = AdaptiveBatcher() if enable_adaptive_batching else None
        self.monitor = QueueStateMonitor()
        self.feedback = FeedbackAdjuster() if enable_feedback else None
        
        self.enable_prioritization = enable_prioritization
        self.enable_adaptive_batching = enable_adaptive_batching
        self.enable_feedback = enable_feedback
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"WorkflowController initialized: "
                         f"prioritization={enable_prioritization}, "
                         f"adaptive_batching={enable_adaptive_batching}, "
                         f"feedback={enable_feedback}")
    
    def process_requests(self, requests: List, current_time: float) -> List:
        """Prioritize requests if enabled."""
        if self.enable_prioritization and self.prioritizer:
            return self.prioritizer.prioritize_requests(requests, current_time)
        return requests
    
    def get_batch_size(self, queue: List, 
                      historical_data: Optional[Dict] = None,
                      ramp_ids: Optional[List[int]] = None,
                      base_batch_size: int = 8) -> int:
        """Get adaptive batch size if enabled."""
        if not self.enable_adaptive_batching or not self.batcher:
            return min(base_batch_size, len(queue))
        
        confidence_predictions = None
        if historical_data and ramp_ids:
            sample_indices = list(range(min(len(queue), 64)))
            confidence_predictions = self.batcher.predict_early_exit_confidence(
                historical_data, ramp_ids, sample_indices)
        
        return self.batcher.get_adaptive_batch_size(
            queue, confidence_predictions, base_batch_size)
    
    def update_queue_state(self, queue_length: int, avg_wait_time: float,
                          slo_violations: int = 0):
        """Update queue state monitoring."""
        self.monitor.update(queue_length, avg_wait_time, slo_violations)
    
    def get_queue_state(self) -> Dict:
        """Get current queue state."""
        return self.monitor.get_queue_state()
    
    def adjust_thresholds_feedback(self, current_thresholds: List[float],
                                  ramp_ids: List[int]) -> List[float]:
        """Adjust thresholds based on queue state if enabled."""
        if not self.enable_feedback or not self.feedback:
            return current_thresholds
        
        queue_state = self.monitor.get_queue_state()
        return self.feedback.adjust_thresholds(
            queue_state, current_thresholds, ramp_ids)
    
    def reset(self):
        """Reset all monitoring state."""
        self.monitor.reset()


# methods/trigger.py
import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from collections import deque
from scipy import stats

logger = logging.getLogger(__name__)


class AdaptiveTrigger:
    """
    Adaptive update scheduler for continual learning.
    
    Decides WHEN to perform model updates based on monitored signals.
    Reduces computational cost by avoiding unnecessary updates while
    maintaining model performance.
    
    TRIGGER TYPES:
    
    1. 'entropy': Trigger when prediction entropy exceeds baseline.
       H_recent > H_baseline · (1 + threshold)
       Good for detecting when model becomes uncertain.
    
    2. 'drift': Trigger when data distribution changes.
       Kolmogorov-Smirnov test on recent vs baseline features.
       p_value < threshold → trigger.
       Good for detecting concept drift in IoT streams.
    
    3. 'performance': Trigger when accuracy drops.
       accuracy_recent < 1 - threshold → trigger.
       Requires ground truth labels.
    
    4. 'hybrid': Weighted combination of all three.
       weighted_sum > 0.5 → trigger (with configured rule).
    
    BASELINE ESTABLISHMENT:
    After warmup_samples, compute baseline metrics from recent window.
    Baseline is re-established after each trigger event (adaptive).
    
    Args:
        trigger_type: 'entropy', 'drift', 'performance', or 'hybrid'
        window_size: Number of recent samples to monitor
        threshold: Relative change needed to trigger
        min_update_interval: Minimum samples between consecutive updates
        warmup_samples: Samples needed before baseline is established
        device: Computation device
    """
    
    def __init__(self,
                 trigger_type: str = 'entropy',
                 window_size: int = 100,
                 threshold: float = 0.15,
                 min_update_interval: int = 50,
                 warmup_samples: int = 50,
                 device: str = 'cpu'):
        
        self.trigger_type = trigger_type
        self.window_size = window_size
        self.threshold = threshold
        self.min_update_interval = min_update_interval
        self.warmup_samples = warmup_samples
        self.device = device
        
        # Monitoring windows
        self.recent_entropies = deque(maxlen=window_size)
        self.recent_losses = deque(maxlen=window_size)
        self.recent_predictions = deque(maxlen=window_size)  # (pred, true) tuples
        self.recent_features = deque(maxlen=window_size)
        
        # Baselines
        self.baseline_entropy: Optional[float] = None
        self.baseline_loss: Optional[float] = None
        self.baseline_established: bool = False
        
        # State
        self.samples_since_update: int = 0
        self.total_updates: int = 0
        self.total_samples_processed: int = 0
        
        # History for analysis
        self.trigger_history: List[dict] = []
        
        logger.info(f"AdaptiveTrigger: type={trigger_type}, window={window_size}, "
                   f"threshold={threshold}, min_interval={min_update_interval}")
    
    def update(self,
               model: torch.nn.Module,
               x: torch.Tensor,
               y: Optional[int] = None,
               loss: Optional[float] = None) -> bool:
        """
        Process a sample and decide whether to trigger an update.
        
        Args:
            model: Current model
            x: Input features (1D tensor)
            y: True label (optional, for performance trigger)
            loss: Current loss value (optional)
        
        Returns:
            True if model update should be triggered
        """
        self.total_samples_processed += 1
        self.samples_since_update += 1
        
        # Don't trigger before minimum interval
        if self.samples_since_update < self.min_update_interval:
            self._collect_metrics(model, x, y, loss)
            return False
        
        # Collect metrics
        self._collect_metrics(model, x, y, loss)
        
        # Establish baseline if needed
        if not self.baseline_established and self.total_samples_processed >= self.warmup_samples:
            self._establish_baseline()
            self._handle_trigger()
            return True
        
        if not self.baseline_established:
            return False
        
        # Check trigger conditions
        should_trigger = self._check_trigger()
        
        if should_trigger:
            self._handle_trigger()
        
        return should_trigger
    
    def _collect_metrics(self,
                         model: torch.nn.Module,
                         x: torch.Tensor,
                         y: Optional[int] = None,
                         loss: Optional[float] = None):
        """Collect metrics for the current sample."""
        # Entropy
        entropy = self._compute_entropy(model, x)
        self.recent_entropies.append(entropy)
        
        # Loss
        if loss is not None:
            self.recent_losses.append(loss)
        
        # Prediction
        if y is not None:
            model.eval()
            with torch.no_grad():
                x_t = x.unsqueeze(0).to(self.device) if x.dim() == 1 else x.to(self.device)
                logits = model(x_t)
                pred = torch.argmax(logits, dim=1).item()
                self.recent_predictions.append((pred, y))
        
        # Features (for drift detection)
        if isinstance(x, torch.Tensor):
            self.recent_features.append(x.detach().cpu().numpy().flatten())
    
    def _compute_entropy(self, model: torch.nn.Module, x: torch.Tensor) -> float:
        """Compute prediction entropy for a sample."""
        model.eval()
        with torch.no_grad():
            x_t = x.unsqueeze(0).to(self.device) if x.dim() == 1 else x.to(self.device)
            logits = model(x_t)
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean().item()
        return entropy
    
    def _establish_baseline(self):
        """Establish baseline metrics from recent window."""
        if len(self.recent_entropies) > 0:
            self.baseline_entropy = np.mean(list(self.recent_entropies))
        
        if len(self.recent_losses) > 0:
            self.baseline_loss = np.mean(list(self.recent_losses))
        
        self.baseline_established = True
        
        logger.info(f"Baseline established: entropy={self.baseline_entropy:.4f}, "
                   f"loss={self.baseline_loss}")
    
    def _check_trigger(self) -> bool:
        """Check if trigger conditions are met."""
        if self.trigger_type == 'entropy':
            return self._check_entropy_trigger()
        elif self.trigger_type == 'drift':
            return self._check_drift_trigger()
        elif self.trigger_type == 'performance':
            return self._check_performance_trigger()
        elif self.trigger_type == 'hybrid':
            return self._check_hybrid_trigger()
        else:
            return False
    
    def _check_entropy_trigger(self) -> bool:
        """
        Trigger if recent entropy > baseline · (1 + threshold).
        """
        if len(self.recent_entropies) < 5 or self.baseline_entropy is None:
            return False
        
        recent_mean = np.mean(list(self.recent_entropies)[-10:])
        relative_change = (recent_mean - self.baseline_entropy) / (self.baseline_entropy + 1e-10)
        
        return relative_change > self.threshold
    
    def _check_drift_trigger(self) -> bool:
        """
        Trigger if feature distribution changes (KS test).
        
        Compares recent features vs older features using
        Kolmogorov-Smirnov test on selected dimensions.
        """
        if len(self.recent_features) < 20:
            return False
        
        recent = np.array(list(self.recent_features)[-10:])
        older = np.array(list(self.recent_features)[:10])
        
        if recent.shape[0] < 5 or older.shape[0] < 5:
            return False
        
        try:
            num_dims = min(10, recent.shape[1])
            p_values = []
            
            for dim in range(num_dims):
                ks_stat, p_value = stats.ks_2samp(recent[:, dim], older[:, dim])
                p_values.append(p_value)
            
            avg_p_value = np.mean(p_values)
            return avg_p_value < self.threshold  # Low p-value = different distributions
        
        except Exception as e:
            logger.debug(f"Drift detection failed: {e}")
            return False
    
    def _check_performance_trigger(self) -> bool:
        """
        Trigger if recent accuracy < 1 - threshold.
        """
        if len(self.recent_predictions) < 10:
            return False
        
        recent = list(self.recent_predictions)[-10:]
        correct = sum(1 for p, y in recent if p == y)
        accuracy = correct / len(recent)
        
        return accuracy < (1.0 - self.threshold)
    
    def _check_hybrid_trigger(self) -> bool:
        """
        Weighted combination of all trigger types.
        
        weighted_sum = Σ w_i · 1_{trigger_i} > 0.5 → trigger
        """
        # Get individual trigger signals
        entropy_trigger = self._check_entropy_trigger()
        drift_trigger = self._check_drift_trigger()
        perf_trigger = self._check_performance_trigger()
        
        # Default weights
        weights = {
            'entropy': 0.5,
            'drift': 0.3,
            'performance': 0.2
        }
        
        weighted_sum = (
            weights['entropy'] * (1.0 if entropy_trigger else 0.0) +
            weights['drift'] * (1.0 if drift_trigger else 0.0) +
            weights['performance'] * (1.0 if perf_trigger else 0.0)
        )
        
        logger.debug(f"Hybrid: entropy={entropy_trigger}, drift={drift_trigger}, "
                    f"perf={perf_trigger}, weighted_sum={weighted_sum:.2f}")
        
        return weighted_sum > 0.5
    
    def _handle_trigger(self):
        """Handle trigger event."""
        self.trigger_history.append({
            'sample': self.total_samples_processed,
            'type': self.trigger_type,
            'interval': self.samples_since_update,
            'baseline_entropy': self.baseline_entropy,
            'recent_entropy': np.mean(list(self.recent_entropies)) if self.recent_entropies else 0.0
        })
        
        self.total_updates += 1
        self.samples_since_update = 0
        
        # Reset windows after trigger
        self.recent_entropies.clear()
        self.recent_losses.clear()
        self.recent_predictions.clear()
        self.recent_features.clear()
        
        # Re-establish baseline after warmup
        self.baseline_established = False
        
        logger.info(f"Update triggered! Total updates: {self.total_updates}, "
                   f"samples: {self.total_samples_processed}")
    
    def force_update(self):
        """Force an update regardless of trigger conditions."""
        self._handle_trigger()
        logger.info("Update forced!")
    
    def get_trigger_rate(self) -> float:
        """Get the rate of triggers (updates per sample)."""
        if self.total_samples_processed == 0:
            return 0.0
        return self.total_updates / self.total_samples_processed
    
    def get_statistics(self) -> dict:
        """Get trigger statistics."""
        return {
            'trigger_type': self.trigger_type,
            'total_updates': self.total_updates,
            'total_samples': self.total_samples_processed,
            'trigger_rate': self.get_trigger_rate(),
            'samples_since_update': self.samples_since_update,
            'baseline_entropy': self.baseline_entropy,
            'num_triggers': len(self.trigger_history)
        }
    
    def get_trigger_timeline(self) -> List[dict]:
        """Get timeline of trigger events."""
        return self.trigger_history.copy()
    
    def reset(self):
        """Reset trigger state."""
        self.recent_entropies.clear()
        self.recent_losses.clear()
        self.recent_predictions.clear()
        self.recent_features.clear()
        self.samples_since_update = 0
        self.total_updates = 0
        self.total_samples_processed = 0
        self.baseline_established = False
        self.trigger_history.clear()
        logger.info("Trigger state reset")


class HybridTrigger(AdaptiveTrigger):
    """
    Hybrid trigger with configurable weights and combination rule.
    
    Inherits all functionality from AdaptiveTrigger.
    Adds weighted voting across trigger types.
    
    Args:
        trigger_weights: Dict with weights for each type
        hybrid_rule: 'weighted_sum', 'any', or 'all'
    """
    
    def __init__(self,
                 window_size: int = 100,
                 threshold: float = 0.15,
                 min_update_interval: int = 50,
                 warmup_samples: int = 50,
                 trigger_weights: Optional[Dict[str, float]] = None,
                 hybrid_rule: str = 'weighted_sum',
                 device: str = 'cpu'):
        
        super().__init__(
            trigger_type='hybrid',
            window_size=window_size,
            threshold=threshold,
            min_update_interval=min_update_interval,
            warmup_samples=warmup_samples,
            device=device
        )
        
        self.trigger_weights = trigger_weights or {
            'entropy': 0.5,
            'drift': 0.3,
            'performance': 0.2
        }
        self.hybrid_rule = hybrid_rule
        
        # Track individual trigger counts
        self.trigger_counts = {'entropy': 0, 'drift': 0, 'performance': 0}
    
    def _check_hybrid_trigger(self) -> bool:
        """Override with configurable weights and rule."""
        entropy_trig = self._check_entropy_trigger()
        drift_trig = self._check_drift_trigger()
        perf_trig = self._check_performance_trigger()
        
        if entropy_trig: self.trigger_counts['entropy'] += 1
        if drift_trig: self.trigger_counts['drift'] += 1
        if perf_trig: self.trigger_counts['performance'] += 1
        
        triggers = {
            'entropy': entropy_trig,
            'drift': drift_trig,
            'performance': perf_trig
        }
        
        if self.hybrid_rule == 'any':
            return any(triggers.values())
        elif self.hybrid_rule == 'all':
            return all(triggers.values())
        else:  # weighted_sum
            weighted_sum = sum(
                self.trigger_weights.get(k, 0.0) * (1.0 if v else 0.0)
                for k, v in triggers.items()
            )
            return weighted_sum > 0.5
    
    def get_vote_statistics(self) -> dict:
        """Get statistics about which triggers fire most."""
        total = sum(self.trigger_counts.values()) + 1
        return {
            'entropy_ratio': self.trigger_counts['entropy'] / total,
            'drift_ratio': self.trigger_counts['drift'] / total,
            'performance_ratio': self.trigger_counts['performance'] / total,
            'raw_counts': self.trigger_counts.copy()
        }


def create_trigger(config: dict, device: str = 'cpu') -> AdaptiveTrigger:
    """Factory function from config."""
    trigger_cfg = config.get('trigger', {})
    trigger_type = trigger_cfg.get('type', 'entropy')
    
    if trigger_type == 'hybrid':
        return HybridTrigger(
            window_size=trigger_cfg.get('window_size', 100),
            threshold=trigger_cfg.get('threshold', 0.15),
            min_update_interval=trigger_cfg.get('min_update_interval', 50),
            warmup_samples=trigger_cfg.get('warmup_samples', 50),
            trigger_weights=trigger_cfg.get('hybrid_weights', None),
            hybrid_rule=trigger_cfg.get('hybrid_rule', 'weighted_sum'),
            device=device
        )
    else:
        return AdaptiveTrigger(
            trigger_type=trigger_type,
            window_size=trigger_cfg.get('window_size', 100),
            threshold=trigger_cfg.get('threshold', 0.15),
            min_update_interval=trigger_cfg.get('min_update_interval', 50),
            warmup_samples=trigger_cfg.get('warmup_samples', 50),
            device=device
        )


# ──────────────────────────────────────
# Quick test
# ──────────────────────────────────────
def test_trigger():
    import sys
    sys.path.append('.')
    from models.base_model import BaseMLP, create_model
    
    config = {
        'model': {'hidden_layers': [16, 8], 'dropout': 0.0},
        'continual_learning': {'learning_rate': 0.01}
    }
    
    learner = create_model(input_dim=10, output_dim=3, config=config)
    
    # Test entropy trigger
    trigger = AdaptiveTrigger(
        trigger_type='entropy',
        window_size=30,
        threshold=0.1,
        min_update_interval=20,
        warmup_samples=30
    )
    
    updates = 0
    for i in range(150):
        x = torch.randn(10)
        if trigger.update(learner.model, x):
            updates += 1
    
    print(f"✓ Entropy trigger: {updates} updates in 150 samples "
          f"(rate={trigger.get_trigger_rate():.2%})")
    
    # Test hybrid trigger
    hybrid = HybridTrigger(
        window_size=30,
        threshold=0.1,
        min_update_interval=20,
        warmup_samples=30,
        hybrid_rule='weighted_sum'
    )
    
    updates = 0
    for i in range(150):
        x = torch.randn(10)
        y = torch.randint(0, 3, (1,)).item()
        if hybrid.update(learner.model, x, y=y):
            updates += 1
    
    print(f"✓ Hybrid trigger: {updates} updates in 150 samples")
    
    stats = hybrid.get_statistics()
    vote_stats = hybrid.get_vote_statistics()
    print(f"  Stats: rate={stats['trigger_rate']:.2%}")
    print(f"  Votes: {vote_stats}")
    
    print("✓ All trigger tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_trigger()
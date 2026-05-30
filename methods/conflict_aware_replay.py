# methods/conflict_aware_replay.py
import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from methods.replay import ReplayBuffer
from methods.ewc import EWC

logger = logging.getLogger(__name__)


class ConflictAwareReplayBuffer(ReplayBuffer):
    """
    Replay buffer with conflict-aware sampling.
    
    Extends ReplayBuffer to consider conflict between replay samples
    and EWC constraints. Samples that require large updates to
    EWC-protected parameters are considered "high conflict" and
    receive a penalty in their sampling weight.
    
    FORMULATION:
        effective_weight = importance_score × conflict_penalty
        
        where:
        conflict_penalty = exp(-conflict_score / temperature)
        conflict_score = Σ_i F_i · max(0, ∇_i · (θ_i - θ_i*))
    
    This directly addresses the tension between:
    - EWC: "Don't change important parameters"
    - Replay: "Update to fix mistakes on important samples"
    
    KEY DESIGN CHOICES:
    
    1. SOFT DEPRIORITIZATION (not hard exclusion):
       High-conflict samples have lower probability but can still be
       sampled. This prevents them from being permanently ignored.
    
    2. PERIODIC FORCED INCLUSION:
       force_high_conflict_ratio ensures a fraction of each replay
       batch contains high-conflict samples, preventing them from
       being completely forgotten.
    
    3. CONFLICT RE-EVALUATION:
       Conflict scores change as model parameters evolve and EWC
       state accumulates. Periodic re-evaluation keeps scores accurate.
    
    Args:
        max_size: Maximum buffer capacity
        temperature: Sampling temperature (from ReplayBuffer)
        age_decay: Age decay factor (from ReplayBuffer)
        conflict_threshold: Conflict score above which sample is "high-conflict"
        conflict_temp: Temperature for conflict penalty
                      (higher = softer penalty, more uniform)
        force_high_conflict_ratio: Fraction of replay batch forced to be high-conflict
    """
    
    def __init__(self,
                 max_size: int = 500,
                 temperature: float = 1.0,
                 age_decay: float = 0.001,
                 conflict_threshold: float = 0.5,
                 conflict_temp: float = 1.0,
                 force_high_conflict_ratio: float = 0.1):
        
        super().__init__(
            max_size=max_size,
            temperature=temperature,
            age_decay=age_decay,
            eviction_policy='lowest_importance'
        )
        
        self.conflict_threshold = conflict_threshold
        self.conflict_temp = conflict_temp
        self.force_high_conflict_ratio = force_high_conflict_ratio
        
        # Tracking for analysis
        self.total_added: int = 0
        self.high_conflict_added: int = 0
        
        logger.info(f"ConflictAwareReplayBuffer: conflict_threshold={conflict_threshold}, "
                   f"conflict_temp={conflict_temp}, force_ratio={force_high_conflict_ratio}")
    
    def compute_conflict_score(self,
                                ewc: EWC,
                                model: torch.nn.Module,
                                x: torch.Tensor,
                                y: int,
                                device: str = 'cpu') -> float:
        """
        Compute conflict between a sample and EWC constraints.
        
        Process:
        1. Forward + backward to get gradients
        2. EWC.compute_conflict_score(gradients)
        3. Returns scalar conflict value
        
        High conflict = this sample requires updating important
        parameters in directions that violate EWC protection.
        
        Args:
            ewc: EWC instance with consolidated parameters
            model: Current model
            x: Input features (1D tensor)
            y: Label
        
        Returns:
            Conflict score (higher = more conflict)
        """
        if not ewc.consolidated_params:
            return 0.0
        
        model.train()  # Need gradients for backward
        
        x_tensor = x.unsqueeze(0).to(device) if x.dim() == 1 else x.to(device)
        y_tensor = torch.tensor([y]).to(device)
        
        model.zero_grad(set_to_none=True)
        
        logits = model(x_tensor)
        loss = F.cross_entropy(logits, y_tensor)
        loss.backward()
        
        # Collect gradients
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.detach().clone()
        
        model.zero_grad(set_to_none=True)
        
        # Compute conflict via EWC
        conflict = ewc.compute_conflict_score(gradients)
        
        return conflict
    
    def compute_batch_conflict_scores(self,
                                       ewc: EWC,
                                       model: torch.nn.Module,
                                       x_batch: torch.Tensor,
                                       y_batch: torch.Tensor,
                                       device: str = 'cpu') -> List[float]:
        """
        Compute conflict scores for a batch.
        
        Returns:
            List of conflict scores (one per sample)
        """
        scores = []
        for i in range(len(x_batch)):
            score = self.compute_conflict_score(
                ewc, model, x_batch[i], int(y_batch[i].item()), device
            )
            scores.append(score)
        return scores
    
    def _is_high_conflict(self, conflict_score: float) -> bool:
        """Check if conflict exceeds threshold."""
        return conflict_score > self.conflict_threshold
    
    def _compute_conflict_penalty(self, conflict_score: float) -> float:
        """
        Soft penalty based on conflict.
        
        penalty = exp(-conflict / temp)
        
        temp → ∞: penalty → 1 (no effect)
        temp → 0: penalty → 0 for any conflict (hard exclusion)
        """
        return np.exp(-conflict_score / self.conflict_temp)
    
    def add_sample_with_conflict(self,
                                  ewc: EWC,
                                  model: torch.nn.Module,
                                  x: torch.Tensor,
                                  y: int,
                                  task_id: int,
                                  importance_score: Optional[float] = None,
                                  device: str = 'cpu'):
        """
        Add a sample with conflict-aware metadata.
        
        Computes:
        - importance_score (if not provided)
        - conflict_score
        - conflict_penalty
        - effective_importance = importance × penalty
        
        Args:
            ewc: EWC instance
            model: Current model
            x: Input features
            y: Label
            task_id: Task identifier
            importance_score: Pre-computed importance (or None to auto-compute)
            device: Computation device
        """
        # Compute importance if not provided
        if importance_score is None:
            importance_score = self.compute_importance_score(model, x, y, device)
        
        # Compute conflict
        conflict_score = self.compute_conflict_score(ewc, model, x, y, device)
        conflict_penalty = self._compute_conflict_penalty(conflict_score)
        effective_importance = importance_score * conflict_penalty
        
        self.total_added += 1
        if self._is_high_conflict(conflict_score):
            self.high_conflict_added += 1
        
        metadata = {
            'importance_score': importance_score,
            'conflict_score': conflict_score,
            'conflict_penalty': conflict_penalty,
            'effective_importance': effective_importance,
            'is_high_conflict': self._is_high_conflict(conflict_score),
            'added_at_task': task_id,
            'times_sampled': 0
        }
        
        self.add_sample(x, y, task_id, metadata)
    
    def add_batch_with_conflict(self,
                                 ewc: EWC,
                                 model: torch.nn.Module,
                                 x_batch: torch.Tensor,
                                 y_batch: torch.Tensor,
                                 task_id: int,
                                 importance_scores: Optional[List[float]] = None,
                                 device: str = 'cpu'):
        """
        Add a batch with conflict-aware scoring.
        
        Args:
            ewc: EWC instance
            model: Current model
            x_batch: Input features [batch_size, features]
            y_batch: Labels [batch_size]
            task_id: Task identifier
            importance_scores: Pre-computed importance scores (or None)
            device: Computation device
        """
        if importance_scores is None:
            importance_scores = [None] * len(x_batch)
        
        conflict_scores = self.compute_batch_conflict_scores(
            ewc, model, x_batch, y_batch, device
        )
        
        for i in range(len(x_batch)):
            self.add_sample_with_conflict(
                ewc=ewc,
                model=model,
                x=x_batch[i],
                y=int(y_batch[i].item()),
                task_id=task_id,
                importance_score=importance_scores[i],
                device=device
            )
        
        avg_conflict = np.mean(conflict_scores)
        logger.info(f"Added {len(x_batch)} samples. "
                   f"Avg conflict={avg_conflict:.4f}, "
                   f"High conflict ratio={self.get_high_conflict_ratio():.2%}")
    
    def sample_conflict_aware(self,
                               batch_size: int,
                               force_include_high_conflict: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample with conflict-aware weighting AND optional forced inclusion.
        
        Strategy:
        1. Split buffer into low-conflict and high-conflict groups
        2. Sample (1 - force_ratio) from low-conflict (weighted by effective_importance)
        3. Sample force_ratio from high-conflict (weighted by importance_score)
        4. Combine and return
        
        This ensures:
        - Most samples are low-conflict (stable learning)
        - Some high-conflict samples are included (prevent permanent neglect)
        
        Args:
            batch_size: Number of samples
            force_include_high_conflict: If True, force some high-conflict samples
        
        Returns:
            Tuple of (x_batch, y_batch)
        """
        if not self.buffer:
            return (torch.empty(0), torch.empty(0, dtype=torch.long))
        
        actual_size = min(batch_size, len(self.buffer))
        
        # Separate samples by conflict level
        low_conflict_indices = []
        high_conflict_indices = []
        
        for i, (_, _, _, meta) in enumerate(self.buffer):
            if meta.get('is_high_conflict', False):
                high_conflict_indices.append(i)
            else:
                low_conflict_indices.append(i)
        
        # Determine how many high-conflict to include
        if force_include_high_conflict and high_conflict_indices:
            num_high = max(1, int(actual_size * self.force_high_conflict_ratio))
            num_high = min(num_high, len(high_conflict_indices))
            num_low = actual_size - num_high
            num_low = min(num_low, len(low_conflict_indices))
        else:
            num_high = 0
            num_low = actual_size
        
        selected_indices = []
        
        # Sample from low-conflict (weighted by effective_importance)
        if low_conflict_indices and num_low > 0:
            low_weights = []
            for idx in low_conflict_indices:
                _, _, _, meta = self.buffer[idx]
                eff_imp = self._get_effective_importance(meta)
                low_weights.append(max(eff_imp, 1e-8))
            
            low_weights = np.array(low_weights)
            
            if self.temperature > 0:
                low_weights = low_weights / self.temperature
                low_weights = low_weights - low_weights.max()
                low_probs = np.exp(low_weights)
                low_probs = low_probs / low_probs.sum()
            else:
                low_probs = np.zeros_like(low_weights)
                low_probs[low_weights.argmax()] = 1.0
            
            selected_low = np.random.choice(
                low_conflict_indices, size=num_low, p=low_probs, replace=False
            )
            selected_indices.extend(selected_low)
        
        # Sample from high-conflict (weighted by importance_score, ignoring conflict)
        if high_conflict_indices and num_high > 0:
            high_weights = []
            for idx in high_conflict_indices:
                _, _, _, meta = self.buffer[idx]
                imp = meta.get('importance_score', 1.0)
                high_weights.append(max(imp, 1e-8))
            
            high_weights = np.array(high_weights)
            high_probs = high_weights / high_weights.sum()
            
            selected_high = np.random.choice(
                high_conflict_indices, size=num_high, p=high_probs, replace=False
            )
            selected_indices.extend(selected_high)
        
        np.random.shuffle(selected_indices)
        
        x_batch = torch.stack([self.buffer[i][0] for i in selected_indices])
        y_batch = torch.tensor([self.buffer[i][1] for i in selected_indices], dtype=torch.long)
        
        # Update metadata
        for i in selected_indices:
            _, _, _, meta = self.buffer[i]
            meta['times_sampled'] = meta.get('times_sampled', 0) + 1
        
        return x_batch, y_batch
    

    # methods/conflict_aware_replay.py

    def _sample_impl(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Conflict-aware implementation.
        
        Encapsulates:
        - Temperature-controlled sampling
        - Conflict penalty on weights
        - Forced high-conflict inclusion
        - Age decay
        
        Called by sample() — the single policy dispatch point.
        """
        return self.sample_conflict_aware(
            batch_size,
            force_include_high_conflict=True
        )



    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        SINGLE POLICY POINT for conflict-aware sampling.
        
        Delegates to sample_conflict_aware which handles:
        - Temperature-controlled importance weighting
        - Conflict penalty
        - Forced high-conflict inclusion
        
        The main loop calls ONLY this method.
        Buffer owns the sampling policy entirely.
        """
        return self.sample_conflict_aware(
            batch_size, 
            force_include_high_conflict=True
        )
    
    def reevaluate_conflicts(self,
                              ewc: EWC,
                              model: torch.nn.Module,
                              device: str = 'cpu'):
        """
        Recompute conflict scores for all buffer samples.
        
        Should be called periodically because:
        - EWC consolidated parameters change after each task
        - Model parameters change during training
        - A low-conflict sample might become high-conflict (or vice versa)
        
        Args:
            ewc: EWC instance (updated)
            model: Current model (updated)
            device: Computation device
        """
        logger.info("Reevaluating conflict scores...")
        
        for i in range(len(self.buffer)):
            x, y, task_id, meta = self.buffer[i]
            
            new_conflict = self.compute_conflict_score(ewc, model, x, y, device)
            
            meta['conflict_score'] = new_conflict
            meta['conflict_penalty'] = self._compute_conflict_penalty(new_conflict)
            meta['is_high_conflict'] = self._is_high_conflict(new_conflict)
            meta['effective_importance'] = (
                meta.get('importance_score', 0.0) * meta['conflict_penalty']
            )
            
            self.buffer[i] = (x, y, task_id, meta)
        
        # Update stats
        self.high_conflict_added = sum(
            1 for _, _, _, m in self.buffer if m.get('is_high_conflict', False)
        )
        
        conflicts = [m.get('conflict_score', 0.0) for _, _, _, m in self.buffer]
        logger.info(f"Conflict reevaluated. "
                   f"Avg={np.mean(conflicts):.4f}, "
                   f"High={self.get_high_conflict_ratio():.2%}")
    
    def get_high_conflict_ratio(self) -> float:
        """Ratio of high-conflict samples in buffer."""
        if len(self.buffer) == 0:
            return 0.0
        return self.high_conflict_added / len(self.buffer)
    
    def get_conflict_statistics(self) -> dict:
        """Get conflict distribution statistics."""
        if not self.buffer:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        conflicts = [m.get('conflict_score', 0.0) for _, _, _, m in self.buffer]
        return {
            'mean': float(np.mean(conflicts)),
            'std': float(np.std(conflicts)),
            'min': float(np.min(conflicts)),
            'max': float(np.max(conflicts)),
            'high_conflict_ratio': self.get_high_conflict_ratio()
        }
    
    def get_statistics(self) -> dict:
        """Extended statistics including conflict info."""
        stats = super().get_statistics()
        stats.update({
            'conflict': self.get_conflict_statistics(),
            'total_added': self.total_added,
            'high_conflict_added': self.high_conflict_added
        })
        return stats


def create_conflict_aware_buffer(config: dict) -> ConflictAwareReplayBuffer:
    """Factory function from config."""
    replay_cfg = config.get('replay', {})
    conflict_cfg = config.get('conflict', {})
    
    return ConflictAwareReplayBuffer(
        max_size=replay_cfg.get('buffer_size', 500),
        temperature=replay_cfg.get('temperature', 1.0),
        age_decay=replay_cfg.get('age_decay', 0.0),
        conflict_threshold=conflict_cfg.get('threshold', 0.5),
        conflict_temp=conflict_cfg.get('temperature', 1.0),
        force_high_conflict_ratio=conflict_cfg.get('force_high_conflict_ratio', 0.1)
    )


# ──────────────────────────────────────
# Quick test
# ──────────────────────────────────────
def test_conflict_aware_replay():
    import sys
    sys.path.append('.')
    from models.base_model import BaseMLP, create_model
    from methods.ewc import EWC
    
    config = {
        'model': {'hidden_layers': [16, 8], 'dropout': 0.0},
        'continual_learning': {'learning_rate': 0.01}
    }
    
    learner = create_model(input_dim=10, output_dim=3, config=config)
    ewc = EWC(learner.model, lambda_reg=10.0)
    
    # Simulate task 0 consolidation
    X0 = torch.randn(50, 10)
    y0 = torch.randint(0, 3, (50,))
    ds0 = torch.utils.data.TensorDataset(X0, y0)
    dl0 = torch.utils.data.DataLoader(ds0, batch_size=16)
    ewc.consolidate(dl0)
    
    # Create conflict-aware buffer
    buffer = ConflictAwareReplayBuffer(
        max_size=20,
        temperature=1.0,
        age_decay=0.001,
        conflict_threshold=0.5,
        conflict_temp=1.0,
        force_high_conflict_ratio=0.1
    )
    buffer.set_num_classes(3)
    
    # Add samples from task 1 with conflict
    X1 = torch.randn(30, 10)
    y1 = torch.randint(0, 3, (30,))
    
    buffer.add_batch_with_conflict(
        ewc=ewc, model=learner.model,
        x_batch=X1, y_batch=y1, task_id=1
    )
    
    print(f"✓ Buffer: {len(buffer)} samples")
    print(f"  High conflict ratio: {buffer.get_high_conflict_ratio():.2%}")
    
    # Print some samples
    for i, (_, _, _, meta) in enumerate(buffer.buffer[:3]):
        print(f"  Sample {i}: importance={meta['importance_score']:.3f}, "
              f"conflict={meta['conflict_score']:.3f}, "
              f"high_conflict={meta['is_high_conflict']}")
    
    # Sample
    x_batch, y_batch = buffer.sample(8)
    print(f"✓ Sampled: x={x_batch.shape}")
    
    # Conflict stats
    cstats = buffer.get_conflict_statistics()
    print(f"✓ Conflict stats: mean={cstats['mean']:.4f}, "
          f"high={cstats['high_conflict_ratio']:.2%}")
    
    # Reevaluate after model change
    buffer.reevaluate_conflicts(ewc, learner.model)
    print(f"✓ After reevaluation: high_ratio={buffer.get_high_conflict_ratio():.2%}")
    
    print("✓ All conflict-aware replay tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_conflict_aware_replay()
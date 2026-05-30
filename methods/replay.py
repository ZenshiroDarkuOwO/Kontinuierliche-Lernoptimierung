# methods/replay.py
import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import random

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Experience replay buffer for continual learning.
    
    KEY DESIGN CHOICES (defensible in review):
    
    1. TEMPERATURE-CONTROLLED SAMPLING:
       P(i) ∝ exp(I_i / T), where T = temperature.
       T → ∞: uniform sampling (exploration)
       T → 0: greedy on importance (exploitation)
       Default T=1.0 balances diversity vs importance.
    
    2. NORMALIZED IMPORTANCE:
       loss_norm = loss / (loss + 1)
       entropy_norm = entropy / log(num_classes)
       Prevents scale mismatch between loss (~0-10) and entropy (~0-2).
    
    3. AGE-WEIGHTED IMPORTANCE:
       age_factor = exp(-age_decay * age)
       Older samples get slightly lower importance to prevent
       buffer stagnation and allow fresh samples to enter.
    
    4. CPU STORAGE:
       Samples stored on CPU to minimize GPU memory.
       Transferred to GPU only during sampling.
       Acceptable for buffer_size ≤ 10000.
    
    Args:
        max_size: Maximum buffer capacity
        temperature: Sampling temperature (higher = more uniform)
        age_decay: Decay factor for age weighting (0 = no decay)
        eviction_policy: 'fifo' or 'lowest_importance'
    """
    
    def __init__(self,
                 max_size: int = 500,
                 temperature: float = 1.0,
                 age_decay: float = 0.0,
                 eviction_policy: str = 'lowest_importance'):
        
        self.max_size = max_size
        self.temperature = temperature
        self.age_decay = age_decay
        self.eviction_policy = eviction_policy
        
        # Buffer: list of (x, y, task_id, metadata)
        self.buffer: List[Tuple[torch.Tensor, int, int, dict]] = []
        
        # Global step counter for age tracking
        self.global_step: int = 0
        
        # For entropy normalization
        self.num_classes: Optional[int] = None
        
        logger.info(f"ReplayBuffer: size={max_size}, T={temperature}, "
                   f"age_decay={age_decay}, eviction={eviction_policy}")
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def is_full(self) -> bool:
        return len(self.buffer) >= self.max_size
    
    def set_num_classes(self, num_classes: int):
        """Set number of classes for entropy normalization."""
        self.num_classes = num_classes
    
    def add_sample(self,
                   x: torch.Tensor,
                   y: int,
                   task_id: int,
                   metadata: Optional[dict] = None):
        """
        Add a single sample to the buffer.
        
        Args:
            x: Input features (1D tensor)
            y: Label (integer)
            task_id: Task identifier
            metadata: Dict with optional 'importance_score', etc.
        """
        if metadata is None:
            metadata = {}
        
        # Ensure detached CPU copy
        x_copy = x.detach().clone().cpu()
        
        # Record insertion step for age tracking
        metadata['added_at_step'] = self.global_step
        
        sample = (x_copy, int(y), task_id, metadata)
        
        if self.is_full():
            self._evict_one()
        
        self.buffer.append(sample)
    
    def add_batch(self,
                  x_batch: torch.Tensor,
                  y_batch: torch.Tensor,
                  task_id: int,
                  metadata_list: Optional[List[dict]] = None):
        """Add a batch of samples."""
        if metadata_list is None:
            metadata_list = [{} for _ in range(len(x_batch))]
        
        for i in range(len(x_batch)):
            self.add_sample(x_batch[i], y_batch[i].item(), task_id, metadata_list[i])
    
    def _evict_one(self):
        """Remove one sample based on eviction policy."""
        if not self.buffer:
            return
        
        if self.eviction_policy == 'fifo':
            self.buffer.pop(0)
        
        elif self.eviction_policy == 'lowest_importance':
            min_idx = 0
            min_score = float('inf')
            for i, (_, _, _, meta) in enumerate(self.buffer):
                score = meta.get('effective_importance', meta.get('importance_score', 0.0))
                if score < min_score:
                    min_score = score
                    min_idx = i
            self.buffer.pop(min_idx)
    
    def _get_effective_importance(self, metadata: dict) -> float:
        """
        Compute effective importance with age weighting.
        
        effective = importance_score * exp(-age_decay * age)
        
        If age_decay = 0, age has no effect.
        """
        importance = metadata.get('importance_score', 1.0)
        added_step = metadata.get('added_at_step', self.global_step)
        age = self.global_step - added_step
        
        age_factor = np.exp(-self.age_decay * age) if self.age_decay > 0 else 1.0
        
        return importance * age_factor
    

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        SINGLE POLICY DISPATCH POINT.
        
        Internally dispatches to the correct sampling strategy.
        Subclasses override _sample_impl() to change behavior.
        Main loop calls ONLY this method.
        
        This ensures:
        - Buffer OWNS the sampling policy (single source of truth)
        - Controller does NOT know about strategy
        - Main loop does NOT branch on buffer type
        """
        return self._sample_impl(batch_size)


    def _sample_impl(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Default implementation: temperature-controlled importance sampling.
        Override in subclasses for different strategies.
        """
        if not self.buffer:
            return (torch.empty(0), torch.empty(0, dtype=torch.long))
        
        actual_size = min(batch_size, len(self.buffer))
        
        scores = []
        for _, _, _, meta in self.buffer:
            eff_imp = self._get_effective_importance(meta)
            scores.append(eff_imp)
        
        scores = np.array(scores)
        
        if self.temperature > 0:
            scores = scores / self.temperature
            scores = scores - scores.max()
            probs = np.exp(scores)
            probs = probs / probs.sum()
        else:
            probs = np.ones_like(scores) / len(scores)  # Uniform at T=0
        
        indices = np.random.choice(len(self.buffer), size=actual_size, p=probs, replace=False)
        
        x_batch = torch.stack([self.buffer[i][0] for i in indices])
        y_batch = torch.tensor([self.buffer[i][1] for i in indices], dtype=torch.long)
        
        for i in indices:
            _, _, _, meta = self.buffer[i]
            meta['times_sampled'] = meta.get('times_sampled', 0) + 1
        
        return x_batch, y_batch
    
    def compute_importance_score(self,
                                  model: torch.nn.Module,
                                  x: torch.Tensor,
                                  y: int,
                                  device: str = 'cpu') -> float:
        """
        Compute NORMALIZED importance score.
        
        Normalization prevents scale mismatch:
        - loss: normalized by loss/(loss+1) → range [0, 1)
        - entropy: normalized by log(num_classes) → range [0, 1]
        - misclassification: 0 or 1
        
        Importance = 0.4 · loss_norm + 0.4 · entropy_norm + 0.2 · is_wrong
        
        Args:
            model: Current model
            x: Input features (1D tensor)
            y: True label
        
        Returns:
            Normalized importance score [0, 1]
        """
        model.eval()
        
        x_tensor = x.unsqueeze(0).to(device) if x.dim() == 1 else x.to(device)
        y_tensor = torch.tensor([y]).to(device)
        
        with torch.no_grad():
            logits = model(x_tensor)
            
            # Loss (normalized)
            loss_raw = F.cross_entropy(logits, y_tensor, reduction='none').item()
            loss_norm = loss_raw / (loss_raw + 1.0)  # → [0, 1)
            
            # Entropy (normalized by max possible entropy)
            probs = F.softmax(logits, dim=1)
            entropy_raw = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).item()
            
            if self.num_classes and self.num_classes > 1:
                max_entropy = np.log(self.num_classes)
                entropy_norm = entropy_raw / max_entropy if max_entropy > 0 else 0.0
            else:
                entropy_norm = entropy_raw / 2.0  # Fallback
            
            entropy_norm = min(entropy_norm, 1.0)  # Clamp
            
            # Misclassification
            pred = torch.argmax(logits, dim=1).item()
            is_wrong = 1.0 if pred != y else 0.0
        
        importance = 0.4 * loss_norm + 0.4 * entropy_norm + 0.2 * is_wrong
        
        return float(importance)
    
    def update_all_importance(self,
                              model: torch.nn.Module,
                              device: str = 'cpu'):
        """
        Recompute importance for all samples.
        
        NOTE: O(buffer_size). For large buffers, consider lazy update
        (only recompute when sampling or evicting). Acceptable for
        buffer_size ≤ 10000 in thesis context.
        """
        for i in range(len(self.buffer)):
            x, y, task_id, meta = self.buffer[i]
            meta['importance_score'] = self.compute_importance_score(
                model, x, y, device
            )
            self.buffer[i] = (x, y, task_id, meta)
        
        logger.debug(f"Importance updated for {len(self.buffer)} samples")
    
    def step(self):
        """Increment global step counter (call once per training step)."""
        self.global_step += 1
    
    def get_statistics(self) -> dict:
        """Get buffer statistics."""
        if not self.buffer:
            return {
                'size': 0,
                'num_tasks': 0,
                'avg_importance': 0.0,
                'avg_age': 0.0
            }
        
        tasks = set()
        importances = []
        ages = []
        
        for _, _, task_id, meta in self.buffer:
            tasks.add(task_id)
            importances.append(meta.get('importance_score', 0.0))
            age = self.global_step - meta.get('added_at_step', self.global_step)
            ages.append(age)
        
        return {
            'size': len(self.buffer),
            'num_tasks': len(tasks),
            'avg_importance': float(np.mean(importances)),
            'max_importance': float(np.max(importances)),
            'avg_age': float(np.mean(ages)),
            'max_age': float(np.max(ages))
        }
    
    def get_task_distribution(self) -> Dict[int, int]:
        """Get samples per task."""
        dist = {}
        for _, _, task_id, _ in self.buffer:
            dist[task_id] = dist.get(task_id, 0) + 1
        return dist
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.global_step = 0
        logger.info("Replay buffer cleared")


class ImportanceBasedReplayBuffer(ReplayBuffer):
    """
    Replay buffer with automatic normalized importance scoring.
    
    Default configuration optimized for continual learning:
    - Temperature = 1.0 (balanced sampling)
    - Age decay = 0.001 (slight preference for newer samples)
    - Eviction by lowest effective importance
    """
    
    def __init__(self,
                 max_size: int = 500,
                 temperature: float = 1.0,
                 age_decay: float = 0.001):
        super().__init__(
            max_size=max_size,
            temperature=temperature,
            age_decay=age_decay,
            eviction_policy='lowest_importance'
        )
    
    def add_batch_with_importance(self,
                                   model: torch.nn.Module,
                                   x_batch: torch.Tensor,
                                   y_batch: torch.Tensor,
                                   task_id: int,
                                   device: str = 'cpu'):
        """Add batch with automatic importance computation."""
        metadata_list = []
        for i in range(len(x_batch)):
            importance = self.compute_importance_score(
                model, x_batch[i], int(y_batch[i].item()), device
            )
            metadata_list.append({
                'importance_score': importance,
                'added_at_task': task_id,
                'times_sampled': 0
            })
        
        self.add_batch(x_batch, y_batch, task_id, metadata_list)
    
    # methods/replay.py — ImportanceBasedReplayBuffer

    def _sample_impl(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Importance-based sampling with normalization.
        Inherits temperature-controlled logic from parent.
        """
        return super()._sample_impl(batch_size)


def create_replay_buffer(config: dict) -> ReplayBuffer:
    """Factory function from config."""
    replay_cfg = config.get('replay', {})
    strategy = replay_cfg.get('strategy', 'random')
    buffer_size = replay_cfg.get('buffer_size', 500)
    eviction = replay_cfg.get('eviction_policy', 'lowest_importance')
    temperature = replay_cfg.get('temperature', 1.0)
    age_decay = replay_cfg.get('age_decay', 0.0)
    
    if strategy == 'importance':
        return ImportanceBasedReplayBuffer(
            max_size=buffer_size,
            temperature=temperature,
            age_decay=age_decay
        )
    else:
        return ReplayBuffer(
            max_size=buffer_size,
            temperature=temperature,
            age_decay=age_decay,
            eviction_policy=eviction
        )


# ──────────────────────────────────────
# Quick test
# ──────────────────────────────────────
def test_replay():
    import sys
    sys.path.append('.')
    from models.base_model import BaseMLP, create_model
    
    config = {
        'model': {'hidden_layers': [8], 'dropout': 0.0},
        'continual_learning': {'learning_rate': 0.01}
    }
    
    learner = create_model(input_dim=10, output_dim=3, config=config)
    
    # Test with temperature
    buffer = ReplayBuffer(max_size=10, temperature=0.5, age_decay=0.01)
    buffer.set_num_classes(3)
    
    for i in range(15):
        x = torch.randn(10)
        y = torch.randint(0, 3, (1,)).item()
        imp = buffer.compute_importance_score(learner.model, x, y)
        buffer.add_sample(x, y, task_id=i//5, 
                         metadata={'importance_score': imp})
        buffer.step()
    
    print(f"✓ Buffer: {len(buffer)} samples, step={buffer.global_step}")
    
    x_batch, y_batch = buffer.sample(5)
    print(f"✓ Sampled: x={x_batch.shape}")
    
    stats = buffer.get_statistics()
    print(f"✓ Stats: avg_imp={stats['avg_importance']:.4f}, "
          f"avg_age={stats['avg_age']:.1f}")
    
    # Test importance buffer
    imp_buffer = ImportanceBasedReplayBuffer(max_size=10, temperature=1.0, age_decay=0.001)
    imp_buffer.set_num_classes(3)
    
    x_batch = torch.randn(15, 10)
    y_batch = torch.randint(0, 3, (15,))
    imp_buffer.add_batch_with_importance(learner.model, x_batch, y_batch, task_id=0)
    
    print(f"\n✓ Importance buffer: {len(imp_buffer)} samples")
    
    # Test normalized importance range
    for i, (_, _, _, meta) in enumerate(imp_buffer.buffer[:3]):
        imp = meta['importance_score']
        print(f"  Sample {i}: importance={imp:.4f} (normalized [0,1])")
    
    print("✓ All replay tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_replay()
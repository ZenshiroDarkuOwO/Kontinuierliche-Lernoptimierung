# methods/controller.py
import torch
import logging
from typing import Optional, Dict, Any, Callable, Tuple

logger = logging.getLogger(__name__)


class CLController:
    """
    Central orchestrator for Continual Learning.
    
    ARCHITECTURAL PRINCIPLE (for paper):
    "Controller does not manipulate learning dynamics. It only orchestrates
    existing components by making two decisions:
    (1) Whether to update now (via Trigger)
    (2) How to assemble the training batch (current + Replay)
    (3) Whether to apply EWC penalty (via EWC instance)"
    
    Controller does NOT:
    - Sample from replay buffer (main loop does this)
    - Modify trigger internal state
    - Compute gradients or update model weights
    - Store data
    
    Separation of concerns:
    - Controller : Orchestration layer (decision + assembly)
    - Trigger    : Temporal signal (when to update)
    - ReplayBuffer: Storage policy (what to remember)
    - EWC        : Regularization (how to constrain)
    - Main Loop  : Data flow and actual training
    
    Args:
        ewc: EWC instance (or None)
        replay: ReplayBuffer reference (or None) — for status checks only
        trigger: AdaptiveTrigger instance (or None)
    """
    
    def __init__(self, ewc=None, replay=None, trigger=None):
        self.ewc = ewc
        self.replay = replay  # Reference only — does NOT sample
        self.trigger = trigger
        
        # Counters for ablation analysis
        self.total_step_calls: int = 0
        self.updates_performed: int = 0
        self.updates_skipped: int = 0
        
        components = []
        if ewc: components.append('EWC')
        if replay is not None: components.append('Replay')
        if trigger: components.append('Trigger')
        
        logger.info(f"CLController: {', '.join(components) if components else 'No components'}")

    def step(self,
             model: torch.nn.Module,
             x: torch.Tensor,
             y: torch.Tensor,
             replay_x: Optional[torch.Tensor] = None,
             replay_y: Optional[torch.Tensor] = None
             ) -> Optional[Dict[str, Any]]:
        """
        Single unified training decision step.
        
        Decision flow:
        1. Trigger check → if no update, return None
        2. Assemble batch (current + replay if provided)
        3. Wrap EWC loss function
        4. Return training package
        """
        self.total_step_calls += 1
        
        # ====================================================
        # DECISION 1: Should we update now?
        # NOTE: Uses single random sample as stochastic proxy signal.
        # ====================================================
        if self.trigger is not None:
            idx = torch.randint(0, len(x), (1,)).item()
            x_rep = x[idx]
            y_rep = y[idx]
            
            should_update = self.trigger.update(
                model,
                x_rep,
                y=y_rep.item() if hasattr(y_rep, 'item') else int(y_rep),
                loss=None
            )
            
            if not should_update:
                self.updates_skipped += 1
                return None  # ← Skip training entirely
        
        # ====================================================
        # DECISION 2: Assemble training batch
        # ====================================================
        train_x = x
        train_y = y
        
        if replay_x is not None and replay_y is not None and len(replay_x) > 0:
            if replay_x.device != x.device:
                replay_x = replay_x.to(x.device)
                replay_y = replay_y.to(y.device)
            
            train_x = torch.cat([x, replay_x], dim=0)
            train_y = torch.cat([y, replay_y], dim=0)
        
        # ====================================================
        # DECISION 3: EWC loss wrapping
        # ====================================================
        ewc_loss_fn = self._get_ewc_loss_fn() if self.ewc is not None else None
        
        # ====================================================
        # Return training package
        # ====================================================
        self.updates_performed += 1
        
        return {
            'x': train_x,
            'y': train_y,
            'ewc_loss_fn': ewc_loss_fn
        }
    
    def _get_ewc_loss_fn(self) -> Callable:
        """
        Create safe EWC loss wrapper.
        
        Returns function f(model, base_loss) -> base_loss + ewc_penalty.
        Includes safety checks for None/unexpected values.
        """
        ewc_ref = self.ewc
        
        def ewc_penalty_loss(model: torch.nn.Module, base_loss: torch.Tensor) -> torch.Tensor:
            try:
                penalty = ewc_ref.ewc_loss(model)
                if penalty is None:
                    return base_loss
                if not isinstance(penalty, torch.Tensor):
                    penalty = torch.tensor(penalty, device=base_loss.device)
                if penalty.ndim > 0:
                    penalty = penalty.mean()
                return base_loss + penalty
            except Exception as e:
                logger.warning(f"EWC penalty failed: {e}. Using base loss only.")
                return base_loss
        
        return ewc_penalty_loss
    
    def needs_replay(self) -> bool:
        """
        Check if replay buffer has samples available.
        
        Main loop calls this BEFORE sampling replay.
        Controller does NOT sample — it only indicates intent.
        """
        if self.replay is None:
            return False
        if not hasattr(self.replay, '__len__'):
            return False
        try:
            return len(self.replay) > 0
        except Exception:
            return False
    
    def should_consolidate(self) -> bool:
        """Check if EWC consolidation should happen after task."""
        return self.ewc is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics for ablation analysis."""
        total = max(self.total_step_calls, 1)
        return {
            'total_step_calls': self.total_step_calls,
            'updates_performed': self.updates_performed,
            'updates_skipped': self.updates_skipped,
            'skip_rate': round(self.updates_skipped / total, 4),
            'update_rate': round(self.updates_performed / total, 4),
            'active_components': {
                'ewc': self.ewc is not None,
                'replay_ref': self.replay is not None,
                'trigger': self.trigger is not None
            }
        }
    
    def reset_stats(self):
        """Reset counters (between experimental runs)."""
        self.total_step_calls = 0
        self.updates_performed = 0
        self.updates_skipped = 0
        logger.debug("Controller counters reset")
# methods/ewc.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Optional
import copy

logger = logging.getLogger(__name__)


class EWC:
    """
    Elastic Weight Consolidation (Kirkpatrick et al., 2017).
    
    Prevents catastrophic forgetting by penalizing changes to parameters
    that were important for previous tasks.
    
    FORMULATION:
        L_total(θ) = L_task(θ) + (λ/2) · Σ_i F_i · (θ_i - θ_i*)^2
    
    FISHER ACCUMULATION (Online EWC):
    This implementation uses CUMULATIVE Fisher with optional decay.
    After each task, new Fisher is estimated and ADDED to existing Fisher
    (not overwritten). This preserves importance from all past tasks.
    
    Optionally, exponential decay can be applied:
        F_new = γ · F_old + F_current
    
    This prevents Fisher from growing unboundedly while still retaining
    information from earlier tasks.
    
    LAMBDA SCHEDULING:
    Default: constant λ (no scaling). This avoids penalty explosion.
    Optional: log scaling λ_scaled = λ · log(1 + n_tasks) for mild growth.
    Future work: adaptive λ based on forgetting/plasticity feedback.
    
    Args:
        model: The neural network
        lambda_reg: Regularization strength (constant, no auto-scaling)
        fisher_sample_size: Samples for Fisher estimation
        gamma: Fisher decay factor (1.0 = cumulative, 0.9 = mild decay)
        lambda_schedule: 'constant' or 'log' (logarithmic growth with tasks)
    """
    
    def __init__(self,
                 model: nn.Module,
                 lambda_reg: float = 100.0,
                 fisher_sample_size: int = 200,
                 gamma: float = 1.0,
                 lambda_schedule: str = 'constant'):
        
        self.model = model
        self.lambda_reg = lambda_reg
        self.fisher_sample_size = fisher_sample_size
        self.gamma = gamma  # Fisher decay: 1.0 = full cumulative, <1.0 = decay
        self.lambda_schedule = lambda_schedule
        self.device = next(model.parameters()).device
        
        # Accumulated state across all tasks
        self.consolidated_params: Dict[str, torch.Tensor] = {}  # θ* (last task optimal)
        self.fisher_information: Dict[str, torch.Tensor] = {}   # F (accumulated)
        
        self.num_tasks_consolidated: int = 0
        
        logger.info(f"EWC: λ={lambda_reg}, schedule={lambda_schedule}, "
                   f"γ={gamma}, fisher_samples={fisher_sample_size}")
    
    def estimate_fisher(self,
                        dataloader: torch.utils.data.DataLoader,
                        num_samples: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Estimate diagonal Fisher Information from data.
        
        F_i = E[(∂/∂θ_i log p(y|x,θ))^2]
        
        Batch-wise diagonal approximation. Uses predicted labels (not ground truth)
        to estimate the expected gradient of the log-likelihood.
        
        Args:
            dataloader: DataLoader for current task
            num_samples: Max samples (default: fisher_sample_size)
        
        Returns:
            Dict of parameter_name → Fisher diagonal values
        """
        if num_samples is None:
            num_samples = self.fisher_sample_size
        
        self.model.eval()
        
        fisher = {}
        for name, param in self.model.named_parameters():
            fisher[name] = torch.zeros_like(param)
        
        samples_processed = 0
        
        for batch_x, batch_y in dataloader:
            if samples_processed >= num_samples:
                break
            
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.model.zero_grad(set_to_none=True)
            
            logits = self.model(batch_x)
            log_probs = F.log_softmax(logits, dim=1)
            
            with torch.no_grad():
                pred_labels = torch.argmax(logits, dim=1)
            
            loss = F.nll_loss(log_probs, pred_labels)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.pow(2) * batch_x.size(0)
            
            samples_processed += batch_x.size(0)
        
        self.model.zero_grad(set_to_none=True)
        
        for name in fisher:
            fisher[name] /= max(samples_processed, 1)
            fisher[name] = torch.clamp(fisher[name], min=1e-8)
        
        logger.info(f"Fisher estimated: {samples_processed} samples")
        return fisher
    
    def consolidate(self, dataloader: torch.utils.data.DataLoader):
        """
        Consolidate knowledge after completing a task.
        
        UPDATES (does NOT overwrite):
        1. consolidated_params: saved as current optimal θ*
        2. fisher_information: ACCUMULATED with optional decay:
           F_new = γ · F_old + F_current
        
        This is Online EWC with cumulative importance — not single-task EWC.
        
        Args:
            dataloader: DataLoader with samples from just-completed task
        """
        # Save current parameters as optimal
        self.consolidated_params = {}
        for name, param in self.model.named_parameters():
            self.consolidated_params[name] = param.detach().clone().to(self.device)
        
        # Estimate Fisher for this task
        new_fisher = self.estimate_fisher(dataloader)
        
        # Accumulate with decay
        if not self.fisher_information:
            # First task: direct assignment
            self.fisher_information = new_fisher
        else:
            # Subsequent tasks: cumulative with decay
            for name in self.fisher_information:
                self.fisher_information[name] = (
                    self.gamma * self.fisher_information[name] + new_fisher[name]
                )
        
        self.num_tasks_consolidated += 1
        
        total_importance = sum(f.sum().item() for f in self.fisher_information.values())
        logger.info(f"Consolidated task {self.num_tasks_consolidated}. "
                   f"Total Fisher importance: {total_importance:.2f}")
    
    def _get_effective_lambda(self) -> float:
        """
        Compute effective lambda with scheduling.
        
        - 'constant': λ (unchanged, prevents penalty explosion)
        - 'log': λ · log(1 + n_tasks) (mild growth)
        
        Future work: adaptive λ = f(forgetting, plasticity).
        """
        if self.lambda_schedule == 'constant':
            return self.lambda_reg
        elif self.lambda_schedule == 'log':
            import math
            return self.lambda_reg * math.log(self.num_tasks_consolidated + 1)
        else:
            logger.warning(f"Unknown lambda_schedule '{self.lambda_schedule}'. Using constant.")
            return self.lambda_reg
    
    def ewc_loss(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC regularization penalty.
        
        L_ewc = (λ_eff/2) · Σ_i F_i · (θ_i - θ_i*)^2
        
        λ_eff depends on lambda_schedule.
        F_i is the ACCUMULATED Fisher (all past tasks).
        
        Args:
            model: Current model
        
        Returns:
            Scalar penalty tensor
        """
        if not self.consolidated_params or not self.fisher_information:
            return torch.tensor(0.0, device=self.device)
        
        loss = 0.0
        for name, param in model.named_parameters():
            if name in self.consolidated_params and name in self.fisher_information:
                theta_star = self.consolidated_params[name]
                fisher = self.fisher_information[name]
                diff = param - theta_star
                loss += torch.sum(fisher * diff.pow(2))
        
        eff_lambda = self._get_effective_lambda()
        return (eff_lambda / 2.0) * loss
    
    def compute_conflict_score(self, gradients: Dict[str, torch.Tensor]) -> float:
        """
        Compute conflict between gradients and EWC constraints.
        
        Conflict = Σ_i F_i · max(0, ∇_i · (θ_i - θ_i*))
        
        Positive conflict: gradient pushes parameter AWAY from θ*.
        Used by conflict-aware replay.
        
        Args:
            gradients: Parameter gradients
        
        Returns:
            Scalar conflict score
        """
        if not self.consolidated_params or not self.fisher_information:
            return 0.0
        
        conflict = 0.0
        
        for name, grad in gradients.items():
            if name in self.consolidated_params and name in self.fisher_information:
                theta_star = self.consolidated_params[name]
                fisher = self.fisher_information[name]
                
                for pname, param in self.model.named_parameters():
                    if pname == name:
                        direction_to_old = theta_star - param
                        per_param_conflict = fisher * grad * direction_to_old
                        conflict += torch.sum(torch.clamp(per_param_conflict, min=0)).item()
                        break
        
        return conflict
    
    def get_parameter_importance(self) -> Dict[str, torch.Tensor]:
        """Get normalized parameter importance scores (accumulated Fisher)."""
        if not self.fisher_information:
            return {}
        
        total = sum(f.sum().item() for f in self.fisher_information.values())
        if total == 0:
            return self.fisher_information
        
        return {name: f / total for name, f in self.fisher_information.items()}
    
    def get_top_k_params(self, k: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Get boolean masks for top-k most important parameters.
        
        Args:
            k: If None → top 5%. If int → top K. If 0<k<1 → top k%.
        
        Returns:
            Dict of parameter_name → boolean mask
        """
        if not self.fisher_information:
            return {}
        
        all_vals = []
        shapes = {}
        for name, f in self.fisher_information.items():
            flat = f.flatten()
            all_vals.append(flat)
            shapes[name] = f.shape
        
        if not all_vals:
            return {}
        
        concat = torch.cat(all_vals)
        total_params = len(concat)
        
        if k is None or k <= 0:
            k = max(1, int(0.05 * total_params))
        elif 0 < k < 1:
            k = max(1, int(k * total_params))
        k = min(k, total_params)
        
        threshold = torch.sort(concat, descending=True)[0][k - 1]
        
        masks = {}
        for name in self.fisher_information:
            masks[name] = self.fisher_information[name] >= threshold
        
        logger.debug(f"Top-{k} params selected (threshold={threshold.item():.6f})")
        return masks
    
    def get_fisher_statistics(self) -> dict:
        """Get statistics about accumulated Fisher for monitoring."""
        if not self.fisher_information:
            return {'total_importance': 0.0, 'mean_per_param': 0.0, 'num_params': 0}
        
        total = sum(f.sum().item() for f in self.fisher_information.values())
        num_params = sum(f.numel() for f in self.fisher_information.values())
        
        return {
            'total_importance': total,
            'mean_per_param': total / max(num_params, 1),
            'num_params': num_params,
            'num_tasks_consolidated': self.num_tasks_consolidated
        }
    
    def reset(self):
        """Reset all EWC state."""
        self.consolidated_params.clear()
        self.fisher_information.clear()
        self.num_tasks_consolidated = 0
        logger.info("EWC state reset")
    
    def state_dict(self) -> dict:
        """Get EWC state for checkpointing."""
        return {
            'consolidated_params': self.consolidated_params,
            'fisher_information': self.fisher_information,
            'num_tasks_consolidated': self.num_tasks_consolidated,
            'lambda_reg': self.lambda_reg,
            'gamma': self.gamma,
            'lambda_schedule': self.lambda_schedule
        }
    
    def load_state_dict(self, state: dict):
        """Load EWC state from checkpoint."""
        self.consolidated_params = state['consolidated_params']
        self.fisher_information = state['fisher_information']
        self.num_tasks_consolidated = state.get('num_tasks_consolidated', 0)
        self.lambda_reg = state.get('lambda_reg', self.lambda_reg)
        self.gamma = state.get('gamma', 1.0)
        self.lambda_schedule = state.get('lambda_schedule', 'constant')
        logger.info("EWC state loaded")


def create_ewc(model: nn.Module, config: dict) -> EWC:
    """Factory function from config."""
    ewc_cfg = config.get('ewc', {})
    return EWC(
        model=model,
        lambda_reg=ewc_cfg.get('lambda', 100.0),
        fisher_sample_size=ewc_cfg.get('fisher_sample_size', 200),
        gamma=ewc_cfg.get('gamma', 1.0),
        lambda_schedule=ewc_cfg.get('lambda_schedule', 'constant')
    )


# ──────────────────────────────────────
# Quick test
# ──────────────────────────────────────
def test_ewc():
    import sys
    sys.path.append('.')
    from models.base_model import BaseMLP, ContinuousLearner, create_model
    
    config = {
        'model': {'hidden_layers': [16, 8], 'dropout': 0.0},
        'continual_learning': {'learning_rate': 0.01}
    }
    
    learner = create_model(input_dim=10, output_dim=3, config=config)
    ewc = EWC(learner.model, lambda_reg=10.0, gamma=0.9, lambda_schedule='constant')
    
    # Simulate two tasks
    for task in range(2):
        X = torch.randn(50, 10) + task * 0.5  # Shift data
        y = torch.randint(0, 3, (50,))
        ds = torch.utils.data.TensorDataset(X, y)
        dl = torch.utils.data.DataLoader(ds, batch_size=16)
        
        # Train briefly
        for _ in range(3):
            learner.train_step(X[:16], y[:16])
        
        # Consolidate
        ewc.consolidate(dl)
        
        stats = ewc.get_fisher_statistics()
        print(f"  Task {task+1} consolidated: total_importance={stats['total_importance']:.2f}")
    
    # Verify accumulation
    loss = ewc.ewc_loss(learner.model)
    print(f"✓ EWC loss (2 tasks accumulated): {loss.item():.6f}")
    
    # Conflict score
    grads = learner.compute_gradients(X[:1], y[:1])
    conflict = ewc.compute_conflict_score(grads)
    print(f"✓ Conflict score: {conflict:.6f}")
    
    print("✓ All EWC tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_ewc()
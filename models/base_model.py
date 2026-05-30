# models/base_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


class BaseMLP(nn.Module):
    """
    Configurable MLP for IoT traffic classification.
    
    Architecture:
        Input → [Linear → BatchNorm → ReLU → Dropout] × N → Linear → Output
    
    Output: raw logits (no activation, no temperature scaling).
    CrossEntropyLoss handles log-softmax internally.
    
    BATCH NORMALIZATION IN CONTINUAL LEARNING:
    BN is included because IoT traffic features are assumed to have
    relatively stable distributions across tasks (same sensor types,
    similar network patterns). For non-stationary streams, future work
    should consider freezing BN statistics after initial training
    or replacing with LayerNorm.
    
    Args:
        input_dim: Number of input features
        hidden_layers: List of hidden layer sizes
        output_dim: Number of classes
        dropout: Dropout probability
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_layers: List[int],
                 output_dim: int,
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_layers
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"BaseMLP: {input_dim} → {hidden_layers} → {output_dim}")
        logger.info(f"  Parameters: {total:,} total, {trainable:,} trainable")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ContinuousLearner:
    """
    Training wrapper (composition over inheritance).
    
    Key design:
    - Gradient clipping (max_norm=5.0) for CL stability
    - Numerically stable entropy: H = -Σ(probs · log(probs + ε))
    - Batch-wise Fisher approximation for EWC (documented)
    - inference_mode for evaluation (PyTorch 1.9+, faster than no_grad)
    - Single source of truth: dimensions from data_loader, not config
    
    Args:
        model: BaseMLP instance
        learning_rate: Adam learning rate
        device: 'auto', 'cpu', or 'cuda'
        grad_clip: Max gradient norm (None to disable)
    """
    
    def __init__(self,
                 model: nn.Module,
                 learning_rate: float = 0.001,
                 device: str = 'auto',
                 grad_clip: Optional[float] = 5.0):
        
        self.device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                      if device == 'auto' else torch.device(device))
        
        self.model = model.to(self.device)
        self.lr = learning_rate
        self.grad_clip = grad_clip
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        self.current_task: int = 0
        self.total_steps: int = 0
        
        logger.info(f"ContinuousLearner: device={self.device}, lr={learning_rate}, "
                   f"grad_clip={grad_clip}")
    
    def _reset_gradients(self):
        """Centralized gradient reset. Use set_to_none for efficiency."""
        self.model.zero_grad(set_to_none=True)
    
    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.to(self.device))
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class labels."""
        self.eval()
        with torch.inference_mode():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities (softmax)."""
        self.eval()
        with torch.inference_mode():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)
    
    def get_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction uncertainty as entropy.
        
        Numerically stable formulation:
        H = -Σ softmax(logits) · log(softmax(logits) + ε)
        
        Higher entropy = more uncertain prediction.
        Used by trigger (entropy-based) and importance-based replay.
        """
        self.eval()
        with torch.inference_mode():
            probs = self.predict_proba(x)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        return entropy
    
    def train_step(self,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   ewc_loss_fn: Optional[callable] = None) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            x: Input batch [batch_size, features]
            y: Target labels [batch_size]
            ewc_loss_fn: Optional EWC penalty.
                         Signature: f(model, base_loss) -> total_loss
        
        Returns:
            Dict with 'loss' and 'accuracy'
        """
        self.train()
        
        x = x.to(self.device)
        y = y.to(self.device)
        
        logits = self.forward(x)
        base_loss = self.criterion(logits, y)
        
        if ewc_loss_fn is not None:
            loss = ewc_loss_fn(self.model, base_loss)
        else:
            loss = base_loss
        
        self._reset_gradients()
        loss.backward()
        
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        self.optimizer.step()
        
        with torch.inference_mode():
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean()
        
        self.total_steps += 1
        
        return {'loss': loss.item(), 'accuracy': acc.item()}
    
    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        Uses inference_mode (faster than no_grad, PyTorch 1.9+).
        """
        self.eval()
        
        X = X.to(self.device)
        y = y.to(self.device)
        
        with torch.inference_mode():
            logits = self.forward(X)
            loss = self.criterion(logits, y)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean()
        
        return {'loss': loss.item(), 'accuracy': acc.item()}
    
    def compute_gradients(self,
                          x: torch.Tensor,
                          y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute batch-level gradients for EWC Fisher estimation.
        
        NOTE FOR PAPER: This uses BATCH-WISE Fisher approximation.
        Full per-sample Fisher is more accurate but computationally
        expensive. Batch-wise approximation is standard in EWC literature
        (Kirkpatrick et al., 2017; Schwarz et al., 2018).
        
        Args:
            x: Input batch
            y: Target labels
        
        Returns:
            Dict of parameter_name → gradient_tensor (detached, cloned, on correct device)
        """
        self.train()
        x, y = x.to(self.device), y.to(self.device)
        
        self._reset_gradients()
        
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        loss.backward()
        
        grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.detach().clone()
            else:
                grads[name] = torch.zeros_like(param, device=param.device)
        
        self._reset_gradients()
        return grads
    
    def get_model_size_mb(self) -> float:
        """Model size in megabytes."""
        param_bytes = sum(p.numel() * p.element_size() 
                         for p in self.model.parameters())
        return param_bytes / (1024 * 1024)
    
    def save(self, path: str):
        """Save model and optimizer state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_task': self.current_task,
            'total_steps': self.total_steps,
            'lr': self.lr
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model and optimizer state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_task = checkpoint.get('current_task', 0)
        self.total_steps = checkpoint.get('total_steps', 0)
        logger.info(f"Model loaded from {path}")


def create_model(input_dim: int,
                 output_dim: int,
                 config: dict) -> ContinuousLearner:
    """
    Factory function.
    
    SINGLE SOURCE OF TRUTH: Dimensions come from data_loader.
    Caller must provide input_dim and output_dim from
    data_loader.prepare_data(), not from config auto-detection.
    
    Args:
        input_dim: Number of features (from data_loader.input_dim)
        output_dim: Number of classes (from data_loader.num_classes)
        config: Full config dict
    
    Returns:
        Configured ContinuousLearner
    """
    model_cfg = config.get('model', {})
    cl_cfg = config.get('continual_learning', {})
    
    base_model = BaseMLP(
        input_dim=input_dim,
        hidden_layers=model_cfg.get('hidden_layers', [128, 64, 32]),
        output_dim=output_dim,
        dropout=model_cfg.get('dropout', 0.3)
    )
    
    return ContinuousLearner(
        model=base_model,
        learning_rate=cl_cfg.get('learning_rate', 0.001),
        device='auto',
        grad_clip=5.0
    )


# ──────────────────────────────────────
# Quick test
# ──────────────────────────────────────
def test_model():
    config = {
        'model': {'hidden_layers': [64, 32], 'dropout': 0.3},
        'continual_learning': {'learning_rate': 0.001}
    }
    
    learner = create_model(input_dim=100, output_dim=5, config=config)
    
    x = torch.randn(32, 100)
    y = torch.randint(0, 5, (32,))
    
    # Forward
    assert learner.forward(x).shape == (32, 5)
    print(f"✓ Forward: {x.shape} → (32, 5)")
    
    # Predict
    assert learner.predict(x).shape == (32,)
    print(f"✓ Predict")
    
    # Proba sums to 1
    probs = learner.predict_proba(x)
    assert torch.allclose(probs.sum(dim=1), torch.ones(32), atol=1e-5)
    print(f"✓ Proba sums to 1")
    
    # Entropy
    entropy = learner.get_uncertainty(x)
    assert entropy.shape == (32,)
    print(f"✓ Entropy: mean={entropy.mean().item():.4f}")
    
    # Train
    m = learner.train_step(x, y)
    print(f"✓ Train: loss={m['loss']:.4f}, acc={m['accuracy']:.4f}")
    
    # Eval
    m = learner.evaluate(x, y)
    print(f"✓ Eval: loss={m['loss']:.4f}, acc={m['accuracy']:.4f}")
    
    # Gradients
    grads = learner.compute_gradients(x, y)
    for name, g in grads.items():
        assert g.device == learner.device, f"Device mismatch: {name}"
    print(f"✓ Gradients: {len(grads)} params, all on {learner.device}")
    
    # Size
    print(f"✓ Size: {learner.get_model_size_mb():.4f} MB")
    
    print("✓ All tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_model()
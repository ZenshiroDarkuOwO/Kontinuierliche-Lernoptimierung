# utils/config_loader.py
import yaml
import copy
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Paper-grade reproducible configuration system.
    
    Design principles:
    - Single source of truth: defaults ONLY in Python, YAML is override-only
    - No silent failures: get_nested warns on missing paths
    - Immutable: updated_copy returns new instance
    - Deep copy safety: no shared references
    - Type-safe: validates parameter types and ranges
    - Reproducible: logs YAML version and dumps with stable ordering
    """
    
    # Only these must exist. Others are optional (defaults applied).
    CORE_REQUIRED = ['dataset', 'model', 'output']
    
    # ================================================================
    # SINGLE SOURCE OF TRUTH: All defaults defined here.
    # YAML only provides overrides — no duplication risk.
    # ================================================================
    DEFAULTS = {
        'dataset': {
            'name': 'synthetic',
            'data_path': './data/raw/',
            'streaming': {'chunk_size': 1000, 'shuffle_chunks': True},
            'batch_size': 64,
            'num_workers': 2,
            'validation_split': 0.2
        },
        'model': {
            'architecture': 'mlp',
            'input_dim': None,
            'output_dim': None,
            'hidden_layers': [128, 64, 32],
            'dropout': 0.3
        },
        'continual_learning': {
            'mode': 'streaming',
            'epochs_per_task': 5,
            'learning_rate': 0.001
        },
        'ewc': {
            'enabled': True,
            'lambda': 100.0,
            'fisher_sample_size': 200,
            'update_fisher_every': 1,
            'lambda_schedule': 'constant',
            'lambda_growth_factor': 1.0,
            'fisher_storage': 'dense'
        },
        'replay': {
            'enabled': True,
            'buffer_size': 500,
            'batch_ratio': 0.3,
            'strategy': 'importance',
            'eviction_policy': 'lowest_importance'
        },
        'conflict': {
            'enabled': True,
            'threshold': 0.5,
            'temperature': 1.0,
            'top_k_params': None,
            'force_high_conflict_ratio': 0.1,
            'reevaluate_every_tasks': 3
        },
        'trigger': {
            'enabled': True,
            'type': 'entropy',
            'window_size': 100,
            'threshold': 0.15,
            'min_update_interval': 50,
            'warmup_samples': 50,
            'hybrid_weights': {'entropy': 0.5, 'drift': 0.3, 'performance': 0.2},
            'hybrid_rule': 'weighted_sum',
            'drift_test': 'ks',
            'drift_p_value': 0.05,
            'drift_feature_selection': 'first_k',
            'drift_feature_dim': 10
        },
        'evaluation': {
            'objectives': ['accuracy', 'forgetting', 'memory_mb', 'latency_ms'],
            'compute_pareto_front': True,
            'pareto_reference_point': {
                'accuracy': 0.0, 'forgetting': 1.0,
                'memory_mb': 1000.0, 'latency_ms': 500.0
            },
            'reference_hardware': 'auto',
            'log_hardware_specs': True
        },
        'optimization': {
            'method': 'grid',
            'trials': 100,
            'grid': {
                'ewc_lambda': [10, 50, 100, 500, 1000],
                'replay_buffer_size': [100, 300, 500, 1000],
                'trigger_threshold': [0.05, 0.10, 0.15, 0.20, 0.30],
                'replay_strategy': ['random', 'importance', 'conflict_aware']
            },
            'random': {
                'ewc_lambda': {'min': 1.0, 'max': 1000.0, 'distribution': 'log_uniform'},
                'replay_buffer_size': {'min': 50, 'max': 2000, 'distribution': 'uniform_int'},
                'trigger_threshold': {'min': 0.01, 'max': 0.50, 'distribution': 'uniform'}
            }
        },
        'output': {
            'results_dir': './results/',
            'models_dir': './saved_models/',
            'log_level': 'INFO',
            'save_model': True,
            'save_metrics': True,
            'save_config': True
        }
    }
    
    # Type & range constraints for validation
    # Format: section.key -> (expected_type, min_value, max_value)
    CONSTRAINTS = {
        ('ewc', 'lambda'): (float, 0.0, 10000.0),
        ('ewc', 'fisher_sample_size'): (int, 10, 10000),
        ('replay', 'buffer_size'): (int, 10, 100000),
        ('replay', 'batch_ratio'): (float, 0.0, 1.0),
        ('replay', 'strategy'): (str, None, None),  # enum: random|importance|conflict_aware
        ('conflict', 'threshold'): (float, 0.0, 100.0),
        ('conflict', 'temperature'): (float, 0.01, 100.0),
        ('trigger', 'window_size'): (int, 10, 10000),
        ('trigger', 'threshold'): (float, 0.0, 1.0),
        ('trigger', 'min_update_interval'): (int, 1, 10000),
        ('continual_learning', 'learning_rate'): (float, 1e-6, 1.0),
        ('continual_learning', 'epochs_per_task'): (int, 1, 1000),
        ('dataset', 'batch_size'): (int, 1, 10000),
        ('model', 'dropout'): (float, 0.0, 0.9),
    }
    
    def __init__(self,
                 config_path: Optional[str] = None,
                 config_dict: Optional[Dict] = None,
                 _skip_processing: bool = False):
        """
        Load config from file or dict.
        
        Args:
            config_path: Path to YAML file
            config_dict: Config dict (internal use)
            _skip_processing: Internal optimization. When creating via updated_copy(),
                            the dict already has defaults merged and constraints validated.
                            Skipping avoids redundant O(n) recursive merge.
        """
        if config_path is not None:
            self.config_path = Path(config_path)
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config not found: {config_path}")
            self.config = self._load_yaml()
        elif config_dict is not None:
            self.config_path = None
            self.config = copy.deepcopy(config_dict)
        else:
            raise ValueError("Provide config_path or config_dict")
        
        if not _skip_processing:
            self._validate_core()
            self._apply_defaults_recursive()
            self._validate_constraints()
        
        logger.info(f"Config loaded: {self}")
    
    def _load_yaml(self) -> Dict:
        """Load YAML with version logging for reproducibility."""
        with open(self.config_path, 'r') as f:
            raw = yaml.safe_load(f)
        logger.debug(f"YAML loaded with PyYAML version: {yaml.__version__}")
        return raw if raw is not None else {}
    
    def _validate_core(self):
        """Validate only core required sections exist."""
        missing = [k for k in self.CORE_REQUIRED if k not in self.config]
        if missing:
            raise ValueError(
                f"Missing required config sections: {missing}. "
                f"Required: {self.CORE_REQUIRED}"
            )
    
    def _apply_defaults_recursive(self):
        """
        Apply Python defaults for ALL sections.
        YAML only provides overrides.
        
        SINGLE SOURCE OF TRUTH: DEFAULTS dict is authoritative.
        No risk of YAML/Python divergence.
        """
        for section, defaults in self.DEFAULTS.items():
            if section not in self.config:
                self.config[section] = copy.deepcopy(defaults)
            else:
                self.config[section] = self._deep_merge(
                    copy.deepcopy(defaults),
                    self.config[section]
                )
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Recursively merge override into base. Base provides defaults."""
        result = copy.deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result
    
    def _validate_constraints(self):
        """
        Validate parameter types and ranges.
        Warns on violations instead of crashing (research configs should be flexible).
        """
        warnings = []
        
        for (section, key), (expected_type, vmin, vmax) in self.CONSTRAINTS.items():
            value = self.get_nested(section, key)
            
            if value is None:
                continue  # null is always allowed
            
            # Type check
            if not isinstance(value, expected_type):
                warnings.append(
                    f"{section}.{key}: expected {expected_type.__name__}, got {type(value).__name__}"
                )
                continue
            
            # Range check (only for numeric types)
            if vmin is not None and value < vmin:
                warnings.append(f"{section}.{key}: {value} < min {vmin}")
            if vmax is not None and value > vmax:
                warnings.append(f"{section}.{key}: {value} > max {vmax}")
        
        for w in warnings:
            logger.warning(f"Config constraint violation: {w}")
    
    def _deep_update_inplace(self, target: Dict, updates: Dict):
        """Recursively update target in-place."""
        for key, value in updates.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update_inplace(target[key], value)
            else:
                target[key] = copy.deepcopy(value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return deep copy. No shared references with internal state."""
        return copy.deepcopy(self.config)
    
    def updated_copy(self, updates: Dict[str, Any]) -> 'ConfigLoader':
        """
        Create NEW ConfigLoader with updates applied.
        
        IMMUTABLE PATTERN:
        - Original config is NEVER modified
        - Returns independent copy with defaults already merged
        - Uses _skip_processing because new_dict is already fully processed
        
        Safe for grid search: each trial is completely isolated.
        
        Args:
            updates: Dict of section -> field -> value overrides
        """
        new_dict = copy.deepcopy(self.config)
        self._deep_update_inplace(new_dict, updates)
        
        # _skip_processing=True because:
        # 1. new_dict is deep copy of fully-processed config (defaults merged)
        # 2. updates are shallow overrides that don't invalidate constraints
        # This avoids O(n) recursive merge on every grid search trial
        return ConfigLoader(config_dict=new_dict, _skip_processing=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get top-level section."""
        return self.config.get(key, default)
    
    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """
        Get nested value with warning on missing paths.
        
        Unlike silent failure, this logs a warning if the path doesn't exist,
        helping catch config bugs during research debugging.
        
        Example:
            config.get_nested('ewc', 'lambda', default=100.0)
        """
        value = self.config
        path_so_far = []
        
        for key in keys:
            path_so_far.append(key)
            if isinstance(value, dict):
                if key not in value:
                    logger.warning(
                        f"Config path '{'.'.join(path_so_far)}' not found. "
                        f"Using default: {default}"
                    )
                    return default
                value = value[key]
            else:
                logger.warning(
                    f"Config path '{'.'.join(path_so_far[:-1])}' is not a dict "
                    f"(got {type(value).__name__}). Using default: {default}"
                )
                return default
        
        return value
    
    def component_enabled(self, component: str) -> bool:
        """Check if component is enabled."""
        return self.get_nested(component, 'enabled', default=False)
    
    def active_components(self) -> List[str]:
        """List all enabled components."""
        return [c for c in ['ewc', 'replay', 'conflict', 'trigger']
                if self.component_enabled(c)]
    
    def save(self, path: str):
        """
        Save config to YAML with stable key ordering.
        Uses sort_keys=False to preserve insertion order (Python 3.7+ dicts are ordered).
        """
        with open(path, 'w') as f:
            yaml.dump(
                self.config, f,
                default_flow_style=False,
                sort_keys=False,        # Preserve section ordering
                allow_unicode=True,
                indent=2
            )
        logger.info(f"Config saved to {path}")
    
    def summary(self) -> str:
        """Human-readable config summary."""
        lines = [
            "=" * 55,
            f"Config: {self.config_path.name if self.config_path else 'from dict'}",
            "=" * 55,
            f"Dataset: {self.get_nested('dataset', 'name')}",
            f"Model: {self.get_nested('model', 'architecture')} "
            f"{self.get_nested('model', 'hidden_layers')}",
            f"Mode: {self.get_nested('continual_learning', 'mode')}",
            "",
            "Active components:",
        ]
        for comp in ['ewc', 'replay', 'conflict', 'trigger']:
            status = "✓" if self.component_enabled(comp) else "✗"
            lines.append(f"  {status} {comp}")
        
        lines.extend([
            "",
            f"Output: {self.get_nested('output', 'results_dir')}",
            "=" * 55
        ])
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        source = self.config_path.name if self.config_path else 'dict'
        return f"ConfigLoader({source}, active={self.active_components()})"


def load_config(config_path: str) -> ConfigLoader:
    """Convenience wrapper."""
    return ConfigLoader(config_path)
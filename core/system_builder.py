# core/system_builder.py
import logging
import torch
from typing import Dict, Optional

from data.data_loader import ContinualDataLoader
from models.base_model import create_model, ContinuousLearner
from methods.ewc import EWC, create_ewc
from methods.replay import ReplayBuffer, create_replay_buffer
from methods.conflict_aware_replay import ConflictAwareReplayBuffer, create_conflict_aware_buffer
from methods.trigger import AdaptiveTrigger, create_trigger
from methods.controller import CLController
from utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class SystemBuilder:
    """
    Builds the complete CL system from configuration.
    
    BUILD ORDER:
    1. DataLoader → prepare data → get dimensions
    2. Model → created with dimensions from data
    3. Components → EWC, Replay, Trigger (based on config)
    4. Controller → orchestrates components
    5. System → ConflictAwareCLSystem (ready to run)
    
    Args:
        config: ConfigLoader instance
    """
    
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"SystemBuilder on {self.device}")
    
    def build(self) -> 'ConflictAwareCLSystem':
        """
        Build the complete system.
        
        Returns:
            Ready-to-run ConflictAwareCLSystem
        """
        from core.cl_system import ConflictAwareCLSystem
        
        logger.info("=" * 50)
        logger.info("Building system...")
        
        # 1. Data
        data_loader = self._build_data_loader()
        chunks, input_dim, num_classes = data_loader.prepare_data()
        
        # 2. Model
        model = self._build_model(input_dim, num_classes)
        
        # 3. Components
        ewc = self._build_ewc(model) if self.config.component_enabled('ewc') else None
        replay_buffer = self._build_replay() if self.config.component_enabled('replay') else None
        trigger = self._build_trigger() if self.config.component_enabled('trigger') else None
        
        # Set num_classes for replay buffer (needed for entropy normalization)
        if replay_buffer is not None:
            replay_buffer.set_num_classes(num_classes)
        
        # 4. Controller
        controller = CLController(ewc=ewc, replay=replay_buffer, trigger=trigger)
        
        # 5. System
        system = ConflictAwareCLSystem(
            config=self.config.to_dict(),
            data_loader=data_loader,
            model=model,
            ewc=ewc,
            replay_buffer=replay_buffer,
            trigger=trigger,
            controller=controller,
            device=self.device
        )
        
        logger.info(f"System built: {controller.get_stats()['active_components']}")
        logger.info("=" * 50)
        
        return system
    
    def _build_data_loader(self) -> ContinualDataLoader:
        """Build data loader from config."""
        return ContinualDataLoader(self.config.to_dict())
    
    def _build_model(self, input_dim: int, num_classes: int) -> ContinuousLearner:
        """
        Build model with dimensions from data.
        
        SINGLE SOURCE OF TRUTH: dimensions come from data_loader.
        """
        return create_model(
            input_dim=input_dim,
            output_dim=num_classes,
            config=self.config.to_dict()
        )
    
    def _build_ewc(self, model: ContinuousLearner) -> Optional[EWC]:
        """Build EWC component."""
        return create_ewc(model.model, self.config.to_dict())
    
    def _build_replay(self) -> Optional[ReplayBuffer]:
        """Build replay buffer based on strategy."""
        strategy = self.config.get_nested('replay', 'strategy', default='random')
        
        if strategy == 'conflict_aware' and self.config.component_enabled('conflict'):
            return create_conflict_aware_buffer(self.config.to_dict())
        elif strategy == 'importance':
            return create_replay_buffer(self.config.to_dict())
        else:
            return create_replay_buffer(self.config.to_dict())
    
    def _build_trigger(self) -> Optional[AdaptiveTrigger]:
        """Build trigger component."""
        return create_trigger(self.config.to_dict(), device=str(self.device))


def build_system(config_path: str) -> 'ConflictAwareCLSystem':
    """
    Convenience: load config and build system in one call.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Ready-to-run ConflictAwareCLSystem
    """
    config = ConfigLoader(config_path)
    builder = SystemBuilder(config)
    return builder.build()


def build_system_from_dict(config_dict: Dict) -> 'ConflictAwareCLSystem':
    """
    Build system from config dict (for grid search).
    
    Args:
        config_dict: Config dictionary (already processed)
    
    Returns:
        Ready-to-run ConflictAwareCLSystem
    """
    config = ConfigLoader(config_dict=config_dict, _from_processed_dict=True)
    builder = SystemBuilder(config)
    return builder.build()
# main.py — نسخه نهایی با هر سه اصلاح
"""
Conflict-Aware Continual Learning Framework
Main entry point — CLI and experiment runner.

Usage:
    python main.py --config config/default_config.yaml --mode single
    python main.py --config config/default_config.yaml --mode grid_search
"""

import logging
import argparse
import json
import time
import random
from pathlib import Path

import torch
import numpy as np

from utils.config_loader import ConfigLoader
from core.system_builder import build_system, build_system_from_dict


def set_seed(seed: int, deterministic: bool = True):
    """
    Set all random seeds for full reproducibility.
    
    Sets Python, NumPy, PyTorch seeds.
    Enables deterministic algorithms and disables CuDNN benchmark
    for GPU reproducibility at the cost of potential slowdown.
    
    Args:
        seed: Random seed
        deterministic: If True, enforce deterministic algorithms (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # For full determinism (may impact performance)
            # torch.use_deterministic_algorithms(True)  # Uncomment if needed
    
    # Set environment variable for Python hash seed
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def compute_cl_score(accuracy: float, forgetting: float, 
                     alpha: float = 0.3) -> float:
    """
    Compute Continual Learning score balancing accuracy and stability.
    
    CL_score = accuracy - α · forgetting
    
    This penalizes models that achieve high accuracy at the cost
    of catastrophic forgetting. α controls the trade-off:
    - α = 0: Pure accuracy maximization
    - α = 1: Equal weight to accuracy and forgetting
    
    Args:
        accuracy: Mean accuracy across all tasks
        forgetting: Mean forgetting across all tasks
        alpha: Forgetting penalty weight (default 0.3)
    
    Returns:
        CL score (higher is better)
    """
    return accuracy - alpha * forgetting


def run_single_experiment(config_path: str) -> dict:
    """
    Run a single experiment with given configuration.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Dict with experiment results
    """
    logger = logging.getLogger(__name__)
    
    # Set base seed for full reproducibility
    BASE_SEED = 42
    set_seed(BASE_SEED, deterministic=True)
    logger.info(f"Random seed set: {BASE_SEED} (deterministic mode)")
    
    logger.info("=" * 60)
    logger.info("CONFLICT-AWARE CONTINUAL LEARNING FRAMEWORK")
    logger.info("=" * 60)
    
    # Load config
    config = ConfigLoader(config_path)
    logger.info(config.summary())
    
    # Build and run system
    logger.info("Building system...")
    system = build_system(config_path)
    
    logger.info("Starting experiment...")
    start_time = time.time()
    
    results = system.run()
    
    total_time = time.time() - start_time
    
    # Extract metrics for CL score
    if system.metrics_history['accuracy']:
        final_accuracy = system.metrics_history['accuracy'][-1]
        avg_forgetting = np.mean(system.metrics_history['forgetting']) if system.metrics_history['forgetting'] else 0.0
        cl_score = compute_cl_score(final_accuracy, avg_forgetting)
    else:
        final_accuracy = results.get('avg_accuracy', 0.0)
        avg_forgetting = 0.0
        cl_score = final_accuracy
    
    # Save results
    output_dir = config.get_nested('output', 'results_dir', default='./results/')
    system.save_results(output_dir)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT COMPLETE")
    logger.info(f"Final accuracy: {final_accuracy:.4f}")
    logger.info(f"Average forgetting: {avg_forgetting:.4f}")
    logger.info(f"CL Score (α=0.3): {cl_score:.4f}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"{'='*60}")
    
    return {
        'avg_accuracy': final_accuracy,
        'avg_forgetting': avg_forgetting,
        'cl_score': cl_score,
        'total_time': total_time
    }


def run_grid_search(config_path: str) -> list:
    """
    Run grid search over hyperparameter space.
    
    Each trial:
    - Gets unique seed for independent reproducibility
    - Is evaluated with CL_score (accuracy - α·forgetting)
    - Has error classification for debugging
    
    Uses ConfigLoader.updated_copy() for immutable config copies.
    
    Args:
        config_path: Path to base YAML config
    
    Returns:
        List of result dicts
    """
    logger = logging.getLogger(__name__)
    
    base_config = ConfigLoader(config_path)
    BASE_SEED = 42
    ALPHA = 0.3  # Forgetting penalty weight
    
    # Get search space from config
    opt_cfg = base_config.get('optimization', {})
    grid_cfg = opt_cfg.get('grid', {})
    
    lambda_vals = grid_cfg.get('ewc_lambda', [10, 100, 500])
    buffer_vals = grid_cfg.get('replay_buffer_size', [100, 500])
    threshold_vals = grid_cfg.get('trigger_threshold', [0.05, 0.15, 0.3])
    strategy_vals = grid_cfg.get('replay_strategy', ['random', 'importance'])
    
    total_combinations = len(lambda_vals) * len(buffer_vals) * len(threshold_vals) * len(strategy_vals)
    
    # Count valid combinations
    valid_count = total_combinations
    if 'conflict_aware' in strategy_vals and not base_config.component_enabled('conflict'):
        valid_count -= len(lambda_vals) * len(buffer_vals) * len(threshold_vals)
    
    logger.info("=" * 60)
    logger.info(f"GRID SEARCH: {valid_count} valid / {total_combinations} total configurations")
    logger.info("=" * 60)
    logger.info(f"Search space:")
    logger.info(f"  λ ∈ {lambda_vals}")
    logger.info(f"  buffer ∈ {buffer_vals}")
    logger.info(f"  τ ∈ {threshold_vals}")
    logger.info(f"  strategy ∈ {strategy_vals}")
    logger.info(f"Base seed: {BASE_SEED}")
    logger.info(f"CL Score: accuracy - {ALPHA} · forgetting")
    
    all_results = []
    trial_num = 0
    
    for lam in lambda_vals:
        for buf_size in buffer_vals:
            for thresh in threshold_vals:
                for strategy in strategy_vals:
                    trial_num += 1
                    
                    # Skip invalid combinations
                    if strategy == 'conflict_aware' and not base_config.component_enabled('conflict'):
                        logger.debug(f"Skipping trial {trial_num}: "
                                   f"conflict_aware requires conflict.enabled=true")
                        continue
                    
                    # Set unique seed for this trial (full deterministic)
                    trial_seed = BASE_SEED + trial_num
                    set_seed(trial_seed, deterministic=True)
                    
                    logger.info(f"\n{'─'*50}")
                    logger.info(f"Trial {trial_num} (seed={trial_seed}): "
                               f"λ={lam}, buf={buf_size}, τ={thresh}, "
                               f"strategy={strategy}")
                    logger.info(f"{'─'*50}")
                    
                    # Create independent config copy (immutable pattern)
                    cfg = base_config.updated_copy({
                        'ewc': {'lambda': lam},
                        'replay': {'buffer_size': buf_size, 'strategy': strategy},
                        'trigger': {'threshold': thresh}
                    })
                    
                    try:
                        # Build and run
                        system = build_system_from_dict(cfg.to_dict())
                        results = system.run()
                        
                        # Extract metrics for CL score
                        if system.metrics_history['accuracy']:
                            final_accuracy = system.metrics_history['accuracy'][-1]
                            avg_forgetting = np.mean(system.metrics_history['forgetting']) if system.metrics_history['forgetting'] else 0.0
                            cl_score = compute_cl_score(final_accuracy, avg_forgetting, ALPHA)
                            final_memory = system.metrics_history['memory_mb'][-1] if system.metrics_history['memory_mb'] else 0.0
                            avg_latency = np.mean(system.metrics_history['latency_ms']) if system.metrics_history['latency_ms'] else 0.0
                        else:
                            final_accuracy = results.get('avg_accuracy', 0.0)
                            avg_forgetting = 0.0
                            cl_score = final_accuracy
                            final_memory = 0.0
                            avg_latency = 0.0
                        
                        # Save per-trial results
                        trial_dir = Path(cfg.get_nested('output', 'results_dir', 
                                                        default='./results/'))
                        trial_dir = trial_dir / f"trial_{trial_num:03d}"
                        system.save_results(str(trial_dir))
                        
                        # Store summary with CL score
                        trial_result = {
                            'trial': trial_num,
                            'seed': trial_seed,
                            'lambda': lam,
                            'buffer_size': buf_size,
                            'trigger_threshold': thresh,
                            'replay_strategy': strategy,
                            'final_accuracy': final_accuracy,
                            'avg_forgetting': avg_forgetting,
                            'cl_score': cl_score,
                            'memory_mb': final_memory,
                            'avg_latency_ms': avg_latency,
                            'status': 'completed'
                        }
                        all_results.append(trial_result)
                        
                        logger.info(f"Trial {trial_num}: acc={final_accuracy:.4f}, "
                                   f"forget={avg_forgetting:.4f}, "
                                   f"CL_score={cl_score:.4f}")
                    
                    except MemoryError as e:
                        logger.error(f"Trial {trial_num} — Out of memory: {e}")
                        all_results.append({
                            'trial': trial_num, 'seed': trial_seed,
                            'lambda': lam, 'buffer_size': buf_size,
                            'trigger_threshold': thresh, 'replay_strategy': strategy,
                            'final_accuracy': 0.0, 'avg_forgetting': 0.0, 'cl_score': 0.0,
                            'status': 'failed', 'error_type': 'memory', 'error': str(e)
                        })
                    
                    except RuntimeError as e:
                        logger.error(f"Trial {trial_num} — Runtime error: {e}")
                        all_results.append({
                            'trial': trial_num, 'seed': trial_seed,
                            'lambda': lam, 'buffer_size': buf_size,
                            'trigger_threshold': thresh, 'replay_strategy': strategy,
                            'final_accuracy': 0.0, 'avg_forgetting': 0.0, 'cl_score': 0.0,
                            'status': 'failed', 'error_type': 'runtime', 'error': str(e)
                        })
                    
                    except Exception as e:
                        logger.error(f"Trial {trial_num} — Unknown error: {e}", exc_info=True)
                        all_results.append({
                            'trial': trial_num, 'seed': trial_seed,
                            'lambda': lam, 'buffer_size': buf_size,
                            'trigger_threshold': thresh, 'replay_strategy': strategy,
                            'final_accuracy': 0.0, 'avg_forgetting': 0.0, 'cl_score': 0.0,
                            'status': 'failed', 'error_type': 'unknown', 'error': str(e)
                        })
    
    # Save grid search summary
    output_dir = Path(base_config.get_nested('output', 'results_dir', default='./results/'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'grid_search_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Analyze results
    completed = [r for r in all_results if r['status'] == 'completed']
    failed = [r for r in all_results if r['status'] == 'failed']
    
    logger.info(f"\n{'='*60}")
    logger.info(f"GRID SEARCH COMPLETE")
    logger.info(f"Completed: {len(completed)}/{len(all_results)}")
    logger.info(f"Failed: {len(failed)}/{len(all_results)}")
    
    if failed:
        error_types = {}
        for r in failed:
            etype = r.get('error_type', 'unknown')
            error_types[etype] = error_types.get(etype, 0) + 1
        logger.info(f"Error breakdown: {error_types}")
    
    if completed:
        # Best by CL score (accuracy - α·forgetting)
        best_cl = max(completed, key=lambda r: r['cl_score'])
        logger.info(f"\n── BEST BY CL SCORE (α={ALPHA}) ──")
        logger.info(f"  Trial {best_cl['trial']} (seed={best_cl['seed']})")
        logger.info(f"  λ={best_cl['lambda']}, buf={best_cl['buffer_size']}, "
                   f"τ={best_cl['trigger_threshold']}, {best_cl['replay_strategy']}")
        logger.info(f"  Accuracy: {best_cl['final_accuracy']:.4f}")
        logger.info(f"  Forgetting: {best_cl['avg_forgetting']:.4f}")
        logger.info(f"  CL Score: {best_cl['cl_score']:.4f}")
        logger.info(f"  Memory: {best_cl.get('memory_mb', 0):.2f} MB")
        
        # Best by accuracy only (for comparison)
        best_acc = max(completed, key=lambda r: r['final_accuracy'])
        logger.info(f"\n── BEST BY ACCURACY ONLY ──")
        logger.info(f"  Trial {best_acc['trial']} (seed={best_acc['seed']})")
        logger.info(f"  λ={best_acc['lambda']}, buf={best_acc['buffer_size']}, "
                   f"τ={best_acc['trigger_threshold']}, {best_acc['replay_strategy']}")
        logger.info(f"  Accuracy: {best_acc['final_accuracy']:.4f}")
        logger.info(f"  Forgetting: {best_acc['avg_forgetting']:.4f}")
        logger.info(f"  CL Score: {best_acc['cl_score']:.4f}")
        
        # Best by memory efficiency
        best_mem = min(completed, key=lambda r: r.get('memory_mb', float('inf')))
        logger.info(f"\n── BEST BY MEMORY EFFICIENCY ──")
        logger.info(f"  Trial {best_mem['trial']}: {best_mem.get('memory_mb', 0):.2f} MB, "
                   f"acc={best_mem['final_accuracy']:.4f}")
        
        # Top 5 by CL score
        top5 = sorted(completed, key=lambda r: r['cl_score'], reverse=True)[:5]
        logger.info(f"\n── TOP 5 BY CL SCORE ──")
        for i, r in enumerate(top5, 1):
            logger.info(f"  {i}. Trial {r['trial']}: λ={r['lambda']}, buf={r['buffer_size']}, "
                       f"τ={r['trigger_threshold']}, {r['replay_strategy']}")
            logger.info(f"     acc={r['final_accuracy']:.4f}, forget={r['avg_forgetting']:.4f}, "
                       f"CL={r['cl_score']:.4f}")
    
    logger.info(f"{'='*60}")
    
    return all_results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Conflict-Aware Continual Learning Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --config config/default_config.yaml --mode single
  python main.py --config config/default_config.yaml --mode grid_search
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/default_config.yaml',
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=str,
        default='single',
        choices=['single', 'grid_search'],
        help='Experiment mode: single run or grid search'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Base random seed for reproducibility'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.3,
        help='Forgetting penalty weight for CL score (0 = accuracy only)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Check config exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {args.config}")
        logger.info("Tip: Create config/default_config.yaml first")
        return 1
    
    try:
        if args.mode == 'single':
            run_single_experiment(args.config)
        elif args.mode == 'grid_search':
            run_grid_search(args.config)
        
        return 0
    
    except KeyboardInterrupt:
        logger.info("\nExperiment interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
# core/cl_system.py
import torch
import numpy as np
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from data.data_loader import ContinualDataLoader
from models.base_model import ContinuousLearner
from methods.ewc import EWC
from methods.replay import ReplayBuffer
from methods.conflict_aware_replay import ConflictAwareReplayBuffer
from methods.trigger import AdaptiveTrigger
from methods.controller import CLController

logger = logging.getLogger(__name__)


class ConflictAwareCLSystem:
    """Complete Continual Learning System."""
    
    def __init__(self, config, data_loader, model, ewc, replay_buffer, trigger, controller, device):
        self.config = config
        self.data_loader = data_loader
        self.learner = model
        self.model = model.model
        self.ewc = ewc
        self.replay_buffer = replay_buffer
        self.trigger = trigger
        self.controller = controller
        self.device = device
        
        self.chunks, self.input_dim, self.num_classes = data_loader.prepare_data()
        self.task_accuracies: Dict[int, float] = {}
        self.current_task: int = 0
        
        self.metrics_history = {
            'task_id': [], 'accuracy': [], 'forgetting': [],
            'memory_mb': [], 'latency_ms': [], 'controller_skip_rate': []
        }
        
        logger.info(f"ConflictAwareCLSystem ready: {len(self.chunks)} tasks, device={device}")
    
    def run(self) -> Dict[str, float]:
        mode = self.config.get('continual_learning', {}).get('mode', 'streaming')
        if mode == 'streaming' and self.config.get('trigger', {}).get('enabled', False):
            logger.info("Running in STREAMING mode with trigger")
            return self._run_streaming()
        else:
            logger.info("Running in TASK-BASED mode")
            return self._run_task_based()
    
    def _run_task_based(self) -> Dict[str, float]:
        total_start = time.time()
        for task_id, (X_chunk, y_chunk) in enumerate(self.chunks):
            self.current_task = task_id
            logger.info(f"\n{'='*55}\nTASK {task_id + 1}/{len(self.chunks)}\n{'='*55}")
            
            dataloader = self.data_loader.get_task_dataloader(X_chunk, y_chunk, task_id)
            task_metrics = self._train_on_task(task_id, dataloader)
            eval_results = self._evaluate_all_tasks()
            forgetting = self._compute_forgetting(task_id)
            memory_mb = self._measure_memory()
            
            self.metrics_history['task_id'].append(task_id)
            self.metrics_history['accuracy'].append(eval_results['avg_accuracy'])
            self.metrics_history['forgetting'].append(forgetting)
            self.metrics_history['memory_mb'].append(memory_mb)
            self.metrics_history['latency_ms'].append(task_metrics.get('avg_step_time_ms', 0.0))
            
            if self.controller and self.controller.should_consolidate():
                self.ewc.consolidate(dataloader)
                logger.info("EWC consolidated")
            if self.controller and self.controller.needs_replay():
                self._update_replay_buffer(X_chunk, y_chunk, task_id)
            
            logger.info(f"Task {task_id}: acc={eval_results['avg_accuracy']:.4f}, "
                       f"forgetting={forgetting:.4f}, memory={memory_mb:.2f}MB")
        
        final_results = self._evaluate_all_tasks()
        logger.info(f"\n{'='*55}\nTRAINING COMPLETE\nFinal: {final_results['avg_accuracy']:.4f}\n{'='*55}")
        return final_results
    
    def _run_streaming(self) -> Dict[str, float]:
        all_X = np.vstack([c[0] for c in self.chunks])
        all_y = np.concatenate([c[1] for c in self.chunks])
        indices = np.random.permutation(len(all_X))
        all_X, all_y = all_X[indices], all_y[indices]
        
        logger.info(f"Streaming {len(all_X)} samples with trigger")
        updates_done = 0
        correct_preds = 0
        total_preds = 0
        start_time = time.time()
        
        for i in tqdm(range(len(all_X)), desc="Streaming"):
            x = torch.FloatTensor(all_X[i]).unsqueeze(0)
            y = torch.LongTensor([all_y[i]])
            
            # Online accuracy: predict BEFORE training on this sample
            if i > 0 and i % 100 == 0:
                with torch.no_grad():
                    logits = self.model(x.to(self.device))
                    pred = torch.argmax(logits, dim=1).item()
                    if pred == y.item():
                        correct_preds += 1
                    total_preds += 1
            
            replay_x, replay_y = None, None
            if self.controller and self.controller.needs_replay():
                replay_batch_size = min(32, len(self.replay_buffer))
                if replay_batch_size > 0:
                    replay_x, replay_y = self.replay_buffer.sample(replay_batch_size)
                    if len(replay_x) > 0:
                        replay_x = replay_x.to(self.device)
                        replay_y = replay_y.to(self.device)
            
            result = self.controller.step(model=self.model, x=x, y=y,
                                         replay_x=replay_x, replay_y=replay_y)
            
            if result is not None:
                self.learner.train_step(result['x'], result['y'], result['ewc_loss_fn'])
                updates_done += 1
                
                if self.replay_buffer:
                    imp = self.replay_buffer.compute_importance_score(
                        self.model, x[0], y[0].item(), str(self.device))
                    task_id = i // self.config.get('dataset', {}).get('streaming', {}).get('chunk_size', 500)
                    self.replay_buffer.add_sample(x[0], y[0].item(), task_id,
                        {'importance_score': imp})
                    self.replay_buffer.step()
        
        total_time = time.time() - start_time
        stats = self.controller.get_stats() if self.controller else {}
        
        # Online streaming accuracy
        streaming_acc = correct_preds / max(total_preds, 1)
        
        # Also evaluate on task eval sets
        final_results = self._evaluate_all_tasks()
        
        logger.info(f"\n{'='*55}")
        logger.info(f"STREAMING COMPLETE")
        logger.info(f"Samples: {len(all_X)}, Updates: {updates_done}")
        logger.info(f"Skip rate: {stats.get('skip_rate', 0):.1%}")
        logger.info(f"Online accuracy (stream): {streaming_acc:.4f}")
        logger.info(f"Eval accuracy (tasks): {final_results['avg_accuracy']:.4f}")
        logger.info(f"Time: {total_time:.2f}s")
        logger.info(f"{'='*55}")
        
        # Return streaming accuracy as primary metric
        return {'avg_accuracy': streaming_acc}
    
    def _train_on_task(self, task_id, dataloader):
        epochs = self.config.get('continual_learning', {}).get('epochs_per_task', 5)
        batch_times = []
        for epoch in range(epochs):
            epoch_loss, epoch_acc, n_batches = 0.0, 0.0, 0
            for batch_x, batch_y in dataloader:
                step_start = time.time()
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                replay_x, replay_y = None, None
                if self.controller and self.controller.needs_replay():
                    rb_size = int(len(batch_x) * self.config.get('replay', {}).get('batch_ratio', 0.3))
                    replay_x, replay_y = self.replay_buffer.sample(rb_size)
                    if len(replay_x) > 0:
                        replay_x, replay_y = replay_x.to(self.device), replay_y.to(self.device)
                
                result = self.controller.step(model=self.model, x=batch_x, y=batch_y,
                                             replay_x=replay_x, replay_y=replay_y)
                if result is None:
                    continue
                
                metrics = self.learner.train_step(result['x'], result['y'], result['ewc_loss_fn'])
                epoch_loss += metrics['loss']
                epoch_acc += metrics['accuracy']
                n_batches += 1
                if self.replay_buffer:
                    self.replay_buffer.step()
                batch_times.append((time.time() - step_start) * 1000)
        
        return {'avg_loss': epoch_loss / max(n_batches, 1),
                'avg_accuracy': epoch_acc / max(n_batches, 1),
                'avg_step_time_ms': np.mean(batch_times) if batch_times else 0.0}
    
    def _evaluate_all_tasks(self):
        all_eval = self.data_loader.get_all_eval_sets()
        if not all_eval:
            return {'avg_accuracy': 0.0}
        accs = {}
        for tid, Xe, ye in all_eval:
            m = self.learner.evaluate(torch.FloatTensor(Xe), torch.LongTensor(ye))
            accs[tid] = m['accuracy']
        return {'avg_accuracy': np.mean(list(accs.values())), 'per_task_accuracy': accs}
    
    def _compute_forgetting(self, task_id):
        if task_id == 0:
            return 0.0
        Xe, ye = self.data_loader.get_task_eval_set(task_id)
        acc = self.learner.evaluate(torch.FloatTensor(Xe), torch.LongTensor(ye))['accuracy']
        self.task_accuracies[task_id] = acc
        scores = []
        for tid in range(task_id):
            if tid in self.task_accuracies:
                Xe, ye = self.data_loader.get_task_eval_set(tid)
                old_acc = self.learner.evaluate(torch.FloatTensor(Xe), torch.LongTensor(ye))['accuracy']
                scores.append(max(0.0, self.task_accuracies[tid] - old_acc))
        return np.mean(scores) if scores else 0.0
    
    def _measure_memory(self):
        mem = sum(p.numel() * p.element_size() for p in self.model.parameters())
        if self.ewc:
            for f in self.ewc.fisher_information.values():
                mem += f.numel() * f.element_size()
            for p in self.ewc.consolidated_params.values():
                mem += p.numel() * p.element_size()
        if self.replay_buffer:
            for x, _, _, _ in self.replay_buffer.buffer:
                mem += x.numel() * x.element_size()
        return mem / (1024 * 1024)
    
    def _update_replay_buffer(self, X_chunk, y_chunk, task_id):
        buf_size = self.config.get('replay', {}).get('buffer_size', 500)
        n = min(buf_size // max(len(self.chunks), 1), len(X_chunk))
        idx = np.random.choice(len(X_chunk), n, replace=False)
        Xa = torch.FloatTensor(X_chunk[idx])
        ya = torch.LongTensor(y_chunk[idx])
        
        if isinstance(self.replay_buffer, ConflictAwareReplayBuffer):
            imps = [self.replay_buffer.compute_importance_score(self.model, Xa[i], int(ya[i].item()), str(self.device)) for i in range(len(Xa))]
            self.replay_buffer.add_batch_with_conflict(self.ewc, self.model, Xa, ya, task_id, imps, str(self.device))
        else:
            meta = [{'importance_score': self.replay_buffer.compute_importance_score(self.model, Xa[i], int(ya[i].item()), str(self.device)), 'added_at_task': task_id} for i in range(len(Xa))]
            self.replay_buffer.add_batch(Xa, ya, task_id, meta)
        logger.info(f"Replay buffer: +{n} samples (total: {len(self.replay_buffer)})")
    
    def save_results(self, output_dir):
        op = Path(output_dir)
        op.mkdir(parents=True, exist_ok=True)
        with open(op / 'metrics.json', 'w') as f: json.dump(self.metrics_history, f, indent=2)
        if self.controller:
            with open(op / 'controller_stats.json', 'w') as f: json.dump(self.controller.get_stats(), f, indent=2)
        if self.replay_buffer:
            with open(op / 'replay_stats.json', 'w') as f: json.dump(self.replay_buffer.get_statistics(), f, indent=2)
        if self.ewc:
            with open(op / 'ewc_stats.json', 'w') as f: json.dump(self.ewc.get_fisher_statistics(), f, indent=2)
        if self.config.get('output', {}).get('save_model', True):
            self.learner.save(str(op / 'final_model.pt'))
        logger.info(f"Results saved to {output_dir}")
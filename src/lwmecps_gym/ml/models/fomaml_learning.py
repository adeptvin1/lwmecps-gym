import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import wandb
from pathlib import Path
import random
import copy

from lwmecps_gym.core.wandb_config import log_metrics
from lwmecps_gym.core.models import ModelType
from .maml_learning import MAMLAgent, TaskDistribution, MAMLActorCritic, MAMLMetricsCollector

logger = logging.getLogger(__name__)

class FOMAMLAgent(MAMLAgent):
    """
    First-Order Model-Agnostic Meta-Learning (FOMAML) implementation.
    
    FOMAML is a computationally efficient approximation of MAML that only uses
    first-order gradients, avoiding the expensive computation of second-order
    derivatives required in full MAML.
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_size: int = 256,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        adaptation_steps: int = 5,
        meta_batch_size: int = 4,
        device: str = "cpu",
        max_replicas: int = 10,
        deployments: List[str] = None,
        task_distribution: Optional[TaskDistribution] = None,
        use_implicit_gradients: bool = True
    ):
        super().__init__(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_size=hidden_size,
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            adaptation_steps=adaptation_steps,
            meta_batch_size=meta_batch_size,
            device=device,
            max_replicas=max_replicas,
            deployments=deployments,
            task_distribution=task_distribution
        )
        
        self.use_implicit_gradients = use_implicit_gradients
        
        # FOMAML-specific metrics
        self.metrics_collector = FOMAMLMetricsCollector()
        
        logger.info(f"FOMAML Agent initialized with obs_dim={obs_dim}, act_dim={act_dim}")
        logger.info(f"Using implicit gradients: {use_implicit_gradients}")
        
    def _compute_meta_gradient(
        self, 
        meta_batch: List[Dict[str, Any]], 
        env_factory
    ) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
        """
        Compute meta-gradient using FOMAML algorithm.
        
        FOMAML approximates the meta-gradient by using only first-order gradients,
        making it computationally more efficient than full MAML.
        
        Returns:
            Tuple of (meta_loss, adaptation_results)
        """
        meta_losses = []
        adaptation_results = []
        
        # Store original meta-model parameters
        original_params = {name: param.clone() for name, param in self.meta_model.named_parameters()}
        
        for task in meta_batch:
            # Create environment for this task
            env = env_factory(task)
            
            # Explicitly start the workload
            if hasattr(env, 'start_workload'):
                env.start_workload()

            # Clone meta-model for task-specific adaptation
            adapted_model = copy.deepcopy(self.meta_model)
            
            # Inner loop: adapt to the specific task
            task_losses = []
            adapted_params = {}
            
            for adaptation_step in range(self.adaptation_steps):
                # Sample data from task
                task_data = self._sample_task_data(env, adapted_model)
                
                if not task_data:
                    logger.info("Experiment group completed. Stopping meta-gradient computation.")
                    return None, []

                # Compute task loss
                task_loss = self._compute_task_loss(task_data, adapted_model)
                
                # Update adapted model using first-order gradient
                task_loss.backward()
                
                # Manual parameter update (FOMAML approach)
                with torch.no_grad():
                    for name, param in adapted_model.named_parameters():
                        if param.grad is not None:
                            param.data = param.data - self.inner_lr * param.grad
                            adapted_params[name] = param.data.clone()
                
                # Clear gradients
                adapted_model.zero_grad()
                
                task_losses.append(task_loss.item())
            
            # Evaluate adapted model on task
            adaptation_result = self._evaluate_adapted_model(env, adapted_model, task)
            adaptation_results.append(adaptation_result)
            
            # Compute meta-loss using adapted parameters
            meta_loss = self._compute_fomaml_meta_loss(env, adapted_model, task, adapted_params)
            meta_losses.append(meta_loss)
        
        # Average meta-loss across tasks
        total_meta_loss = torch.stack(meta_losses).mean()
        
        return total_meta_loss, adaptation_results
        
    def _compute_fomaml_meta_loss(
        self, 
        env, 
        adapted_model, 
        task, 
        adapted_params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute meta-loss for FOMAML.
        
        In FOMAML, we compute the meta-loss using the adapted parameters
        and then compute gradients with respect to the original meta-model parameters.
        """
        # Temporarily update adapted model with final adapted parameters
        for name, param in adapted_model.named_parameters():
            if name in adapted_params:
                param.data = adapted_params[name]
        
        # Sample evaluation data
        eval_data = self._sample_task_data(env, adapted_model, num_samples=5)
        
        # Compute loss on evaluation data
        meta_loss = self._compute_task_loss(eval_data, adapted_model)
        
        return meta_loss
        
    def meta_train(
        self,
        env_factory,
        meta_episodes: int = 100,
        wandb_run_id: str = None,
        training_service=None,
        task_id: str = None,
        loop=None,
        db_connection=None
    ) -> Dict[str, List[float]]:
        """
        Meta-training loop for FOMAML
        
        Args:
            env_factory: Function that creates environments for tasks
            meta_episodes: Number of meta-episodes
            wandb_run_id: Weights & Biases run ID
            training_service: Training service instance
            task_id: Task ID for progress tracking
            loop: Event loop for async operations
            db_connection: Database connection
            
        Returns:
            Dictionary of meta-training metrics
        """
        logger.info(f"Starting FOMAML meta-training for {meta_episodes} episodes")
        
        meta_losses = []
        adaptation_accuracies = []
        task_performances = []
        gradient_norms = []
        adaptation_speeds = []
        
        for meta_episode in range(meta_episodes):
            logger.info(f"FOMAML Meta-episode {meta_episode + 1}/{meta_episodes}")
            
            # Sample meta-batch of tasks
            meta_batch = self._sample_meta_batch()
            
            # Compute meta-gradient using FOMAML
            meta_loss, adaptation_results = self._compute_meta_gradient(meta_batch, env_factory)
            
            if meta_loss is None:
                logger.info("Meta-gradient computation stopped. Finishing training.")
                break

            # Compute gradient norm for monitoring
            total_norm = 0
            for param in self.meta_model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            gradient_norms.append(total_norm)
            
            # Update meta-model
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.meta_model.parameters(), max_norm=1.0)
            
            self.meta_optimizer.step()
            
            # Store metrics
            meta_losses.append(meta_loss.item())
            adaptation_accuracies.append(np.mean([r["accuracy"] for r in adaptation_results]))
            task_performances.append(np.mean([r["performance"] for r in adaptation_results]))
            
            # Calculate adaptation speed (how quickly models adapt)
            adaptation_speed = self._calculate_adaptation_speed(adaptation_results)
            adaptation_speeds.append(adaptation_speed)
            
            # Log to wandb
            if wandb_run_id:
                log_metrics({
                    "fomaml/meta_loss": meta_loss.item(),
                    "fomaml/adaptation_accuracy": np.mean([r["accuracy"] for r in adaptation_results]),
                    "fomaml/task_performance": np.mean([r["performance"] for r in adaptation_results]),
                    "fomaml/gradient_norm": total_norm,
                    "fomaml/adaptation_speed": adaptation_speed,
                    "fomaml/meta_episode": meta_episode
                }, step=meta_episode)
            
            # Update progress in database
            if training_service and task_id and loop and db_connection:
                progress = (meta_episode / meta_episodes) * 100
                metrics = {
                    "meta_loss": meta_loss.item(),
                    "adaptation_accuracy": np.mean([r["accuracy"] for r in adaptation_results]),
                    "task_performance": np.mean([r["performance"] for r in adaptation_results]),
                    "gradient_norm": total_norm,
                    "adaptation_speed": adaptation_speed
                }
                
                future = asyncio.run_coroutine_threadsafe(
                    training_service.save_training_result(task_id, meta_episode, metrics, db_connection),
                    loop
                )
                future.result()
            
            # Log progress
            if meta_episode % 10 == 0:
                logger.info(f"FOMAML Meta-episode {meta_episode}: Loss={meta_loss.item():.4f}, "
                           f"Adaptation Accuracy={np.mean([r['accuracy'] for r in adaptation_results]):.3f}, "
                           f"Gradient Norm={total_norm:.4f}")
        
        # Store final metrics
        self.metrics_collector.update({
            "meta_losses": meta_losses,
            "adaptation_accuracies": adaptation_accuracies,
            "task_performances": task_performances,
            "gradient_norms": gradient_norms,
            "adaptation_speeds": adaptation_speeds
        })
        
        logger.info("FOMAML meta-training completed")
        return {
            "meta_losses": meta_losses,
            "adaptation_accuracies": adaptation_accuracies,
            "task_performances": task_performances,
            "gradient_norms": gradient_norms,
            "adaptation_speeds": adaptation_speeds
        }
        
    def _calculate_adaptation_speed(self, adaptation_results: List[Dict[str, float]]) -> float:
        """Calculate how quickly models adapt to new tasks"""
        # Adaptation speed is measured as the improvement in performance
        # over the adaptation steps
        if len(adaptation_results) < 2:
            return 0.0
            
        performances = [r["performance"] for r in adaptation_results]
        if len(performances) < 2:
            return 0.0
            
        # Calculate the rate of improvement
        improvement_rate = (max(performances) - min(performances)) / len(performances)
        return improvement_rate
        
    def adapt_to_new_task(
        self, 
        env, 
        adaptation_steps: Optional[int] = None,
        inner_lr: Optional[float] = None
    ) -> nn.Module:
        """
        Adapt the meta-model to a new task using FOMAML approach
        
        Args:
            env: Environment for the new task
            adaptation_steps: Number of adaptation steps (uses default if None)
            inner_lr: Learning rate for adaptation (uses default if None)
            
        Returns:
            Adapted model for the new task
        """
        adaptation_steps = adaptation_steps or self.adaptation_steps
        inner_lr = inner_lr or self.inner_lr
        
        # Clone meta-model
        adapted_model = copy.deepcopy(self.meta_model)
        
        logger.info(f"FOMAML adapting to new task with {adaptation_steps} steps, LR={inner_lr}")
        
        # FOMAML adaptation loop
        adaptation_losses = []
        
        for step in range(adaptation_steps):
            # Sample data from new task
            task_data = self._sample_task_data(env, adapted_model, num_samples=10)
            
            if not task_data:
                logger.info("Experiment group completed. Stopping adaptation.")
                break

            # Compute loss
            loss = self._compute_task_loss(task_data, adapted_model)
            adaptation_losses.append(loss.item())
            
            # Manual gradient update (FOMAML style)
            loss.backward()
            
            with torch.no_grad():
                for param in adapted_model.parameters():
                    if param.grad is not None:
                        param.data = param.data - inner_lr * param.grad
            
            # Clear gradients
            adapted_model.zero_grad()
            
            logger.debug(f"FOMAML adaptation step {step + 1}: Loss={loss.item():.4f}")
        
        final_loss = adaptation_losses[-1] if adaptation_losses else 0.0
        logger.info(f"FOMAML task adaptation completed. Final loss: {final_loss:.4f}")
        return adapted_model
        
    def fast_adapt(
        self, 
        env, 
        num_samples: int = 5,
        inner_lr: Optional[float] = None
    ) -> nn.Module:
        """
        Fast adaptation using minimal samples (FOMAML advantage)
        
        Args:
            env: Environment for the new task
            num_samples: Number of samples to use for adaptation
            inner_lr: Learning rate for adaptation
            
        Returns:
            Quickly adapted model
        """
        inner_lr = inner_lr or self.inner_lr
        
        # Clone meta-model
        adapted_model = copy.deepcopy(self.meta_model)
        
        logger.info(f"FOMAML fast adaptation with {num_samples} samples")
        
        # Single adaptation step with minimal data
        task_data = self._sample_task_data(env, adapted_model, num_samples=num_samples)
        
        if not task_data:
            logger.info("Experiment group completed. Stopping fast adaptation.")
            return adapted_model

        loss = self._compute_task_loss(task_data, adapted_model)
        
        # Single gradient step
        loss.backward()
        
        with torch.no_grad():
            for param in adapted_model.parameters():
                if param.grad is not None:
                    param.data = param.data - inner_lr * param.grad
        
        adapted_model.zero_grad()
        
        logger.info(f"FOMAML fast adaptation completed. Loss: {loss.item():.4f}")
        return adapted_model
        
    def save_model(self, path: str):
        """Save the FOMAML meta-model"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'meta_model_state_dict': self.meta_model.state_dict(),
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'hidden_size': self.hidden_size,
            'inner_lr': self.inner_lr,
            'outer_lr': self.outer_lr,
            'adaptation_steps': self.adaptation_steps,
            'meta_batch_size': self.meta_batch_size,
            'deployments': self.deployments,
            'max_replicas': self.max_replicas,
            'use_implicit_gradients': self.use_implicit_gradients,
            'model_type': 'fomaml'
        }, path)
        
        logger.info(f"FOMAML model saved to {path}")
        
    def load_model(self, path: str):
        """Load the FOMAML meta-model"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.meta_model.load_state_dict(checkpoint['meta_model_state_dict'])
        self.inner_lr = checkpoint.get('inner_lr', self.inner_lr)
        self.outer_lr = checkpoint.get('outer_lr', self.outer_lr)
        self.adaptation_steps = checkpoint.get('adaptation_steps', self.adaptation_steps)
        self.meta_batch_size = checkpoint.get('meta_batch_size', self.meta_batch_size)
        self.deployments = checkpoint.get('deployments', self.deployments)
        self.max_replicas = checkpoint.get('max_replicas', self.max_replicas)
        self.use_implicit_gradients = checkpoint.get('use_implicit_gradients', self.use_implicit_gradients)
        
        logger.info(f"FOMAML model loaded from {path}")

class FOMAMLMetricsCollector(MAMLMetricsCollector):
    """Collects and validates FOMAML specific metrics"""
    
    def __init__(self):
        super().__init__()
        self.metric_validators.update({
            "gradient_norms": lambda x: x >= 0,
            "adaptation_speeds": lambda x: x >= 0,
            "fast_adaptation_accuracy": lambda x: 0 <= x <= 1,
            "computational_efficiency": lambda x: x >= 0
        })
    
    def update(self, metrics_dict: Dict[str, Any]):
        """Update metrics with new values"""
        super().update(metrics_dict)
        
        # Add FOMAML-specific metrics
        if "gradient_norms" in metrics_dict:
            if "gradient_norms" not in self.metrics:
                self.metrics["gradient_norms"] = []
            self.metrics["gradient_norms"].extend(metrics_dict["gradient_norms"])
            
        if "adaptation_speeds" in metrics_dict:
            if "adaptation_speeds" not in self.metrics:
                self.metrics["adaptation_speeds"] = []
            self.metrics["adaptation_speeds"].extend(metrics_dict["adaptation_speeds"])

class ImplicitFOMAMLAgent(FOMAMLAgent):
    """
    Implicit FOMAML implementation that uses implicit differentiation
    for more accurate gradient computation while maintaining efficiency.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_implicit_gradients = True
        
    def _compute_meta_gradient(
        self, 
        meta_batch: List[Dict[str, Any]], 
        env_factory
    ) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
        """
        Compute meta-gradient using implicit FOMAML.
        
        Uses implicit differentiation to compute more accurate gradients
        while avoiding second-order derivatives.
        """
        meta_losses = []
        adaptation_results = []
        
        for task in meta_batch:
            # Create environment for this task
            env = env_factory(task)
            
            # Explicitly start the workload
            if hasattr(env, 'start_workload'):
                env.start_workload()
            elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'start_workload'):
                env.unwrapped.start_workload()

            # Clone meta-model for task-specific adaptation
            adapted_model = copy.deepcopy(self.meta_model)
            
            # Inner loop: adapt to the specific task
            task_losses = []
            
            for adaptation_step in range(self.adaptation_steps):
                # Sample data from task
                task_data = self._sample_task_data(env, adapted_model)
                
                if not task_data:
                    logger.info("Experiment group completed. Stopping meta-gradient computation.")
                    return None, []

                # Compute task loss
                task_loss = self._compute_task_loss(task_data, adapted_model)
                
                # Implicit gradient update
                self._implicit_gradient_update(adapted_model, task_loss)
                
                task_losses.append(task_loss.item())
            
            # Evaluate adapted model on task
            adaptation_result = self._evaluate_adapted_model(env, adapted_model, task)
            adaptation_results.append(adaptation_result)
            
            # Compute meta-loss using implicit gradients
            meta_loss = self._compute_implicit_meta_loss(env, adapted_model, task)
            meta_losses.append(meta_loss)
        
        # Average meta-loss across tasks
        total_meta_loss = torch.stack(meta_losses).mean()
        
        return total_meta_loss, adaptation_results
        
    def _implicit_gradient_update(self, model, loss):
        """Perform implicit gradient update"""
        # Compute gradients
        loss.backward()
        
        # Implicit update using momentum-like approach
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    # Use momentum for more stable updates
                    if not hasattr(param, 'momentum'):
                        param.momentum = torch.zeros_like(param.data)
                    
                    param.momentum = 0.9 * param.momentum + self.inner_lr * param.grad
                    param.data = param.data - param.momentum
        
        # Clear gradients
        model.zero_grad()
        
    def _compute_implicit_meta_loss(self, env, adapted_model, task) -> torch.Tensor:
        """Compute meta-loss using implicit differentiation"""
        # Sample evaluation data
        eval_data = self._sample_task_data(env, adapted_model, num_samples=5)
        
        # Compute loss on evaluation data
        meta_loss = self._compute_task_loss(eval_data, adapted_model)
        
        return meta_loss

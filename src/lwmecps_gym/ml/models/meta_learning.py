"""
Meta-learning implementations for reinforcement learning algorithms.

This module provides MAML (Model-Agnostic Meta-Learning) and FOMAML (First-Order MAML)
implementations that can be applied to any existing RL algorithm in the project.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import logging
from abc import ABC, abstractmethod
import copy
import gymnasium as gym

logger = logging.getLogger(__name__)


class BaseMetaLearning(ABC):
    """
    Abstract base class for meta-learning algorithms.
    
    Meta-learning algorithms learn to quickly adapt to new tasks by learning
    good initial parameters that can be fine-tuned with few gradient steps.
    """
    
    def __init__(
        self,
        base_algorithm: Any,
        meta_lr: float = 0.01,
        inner_lr: float = 0.01,
        num_inner_steps: int = 1,
        device: str = "cpu"
    ):
        """
        Initialize meta-learning algorithm.
        
        Args:
            base_algorithm: The base RL algorithm to apply meta-learning to
            meta_lr: Learning rate for meta-updates
            inner_lr: Learning rate for inner loop updates
            num_inner_steps: Number of gradient steps in inner loop
            device: Device for computation
        """
        self.base_algorithm = base_algorithm
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        self.device = device
        
        # Store original parameters
        self.meta_params = self._get_parameters()
        
        # Meta-optimizer
        self.meta_optimizer = optim.Adam(self.meta_params, lr=meta_lr)
        
        # Track meta-learning metrics
        self.meta_losses = []
        self.adaptation_metrics = []
    
    def _get_parameters(self) -> List[torch.Tensor]:
        """Get all trainable parameters from the base algorithm."""
        params = []
        if hasattr(self.base_algorithm, 'model'):
            params.extend(list(self.base_algorithm.model.parameters()))
        elif hasattr(self.base_algorithm, 'actor'):
            params.extend(list(self.base_algorithm.actor.parameters()))
            if hasattr(self.base_algorithm, 'critic1'):
                params.extend(list(self.base_algorithm.critic1.parameters()))
            if hasattr(self.base_algorithm, 'critic2'):
                params.extend(list(self.base_algorithm.critic2.parameters()))
        return params
    
    def _set_parameters(self, params: List[torch.Tensor]):
        """Set parameters in the base algorithm."""
        param_idx = 0
        if hasattr(self.base_algorithm, 'model'):
            for param in self.base_algorithm.model.parameters():
                param.data = params[param_idx].data
                param_idx += 1
        elif hasattr(self.base_algorithm, 'actor'):
            for param in self.base_algorithm.actor.parameters():
                param.data = params[param_idx].data
                param_idx += 1
            if hasattr(self.base_algorithm, 'critic1'):
                for param in self.base_algorithm.critic1.parameters():
                    param.data = params[param_idx].data
                    param_idx += 1
            if hasattr(self.base_algorithm, 'critic2'):
                for param in self.base_algorithm.critic2.parameters():
                    param.data = params[param_idx].data
                    param_idx += 1
    
    @abstractmethod
    def meta_update(self, tasks: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Perform meta-update on a batch of tasks.
        
        Args:
            tasks: List of task data for meta-learning
            
        Returns:
            Dictionary of meta-learning metrics
        """
        pass
    
    def adapt_to_task(self, task_data: Dict[str, Any], num_steps: int = None) -> Dict[str, float]:
        """
        Adapt the model to a specific task using inner loop updates.
        
        Args:
            task_data: Task-specific data
            num_steps: Number of adaptation steps (uses default if None)
            
        Returns:
            Dictionary of adaptation metrics
        """
        if num_steps is None:
            num_steps = self.num_inner_steps
            
        # Create a copy of current parameters for adaptation
        adapted_params = [param.clone() for param in self.meta_params]
        
        # Perform inner loop updates
        adaptation_losses = []
        for step in range(num_steps):
            loss = self._inner_loop_step(task_data, adapted_params)
            adaptation_losses.append(loss)
            
            # Update adapted parameters
            self._update_parameters(adapted_params, loss)
        
        return {
            "adaptation_loss": np.mean(adaptation_losses),
            "final_loss": adaptation_losses[-1] if adaptation_losses else 0.0,
            "adaptation_steps": num_steps
        }
    
    def _inner_loop_step(self, task_data: Dict[str, Any], params: List[torch.Tensor]) -> float:
        """
        Perform one step of inner loop adaptation.
        
        Args:
            task_data: Task-specific data
            params: Current parameters to update
            
        Returns:
            Loss value for this step
        """
        # This is a placeholder - should be implemented by specific algorithms
        # The actual implementation depends on the base algorithm type
        return 0.0
    
    def _update_parameters(self, params: List[torch.Tensor], loss: float):
        """
        Update parameters using gradient descent.
        
        Args:
            params: Parameters to update
            loss: Loss value for gradient computation
        """
        # Compute gradients
        grads = torch.autograd.grad(loss, params, create_graph=True)
        
        # Update parameters
        for param, grad in zip(params, grads):
            param.data = param.data - self.inner_lr * grad.data
    
    def save_meta_model(self, path: str):
        """Save meta-learned parameters."""
        torch.save({
            'meta_params': self.meta_params,
            'meta_lr': self.meta_lr,
            'inner_lr': self.inner_lr,
            'num_inner_steps': self.num_inner_steps,
            'base_algorithm_state': self.base_algorithm.state_dict() if hasattr(self.base_algorithm, 'state_dict') else None
        }, path)
        logger.info(f"Meta-model saved to {path}")
    
    def load_meta_model(self, path: str):
        """Load meta-learned parameters."""
        checkpoint = torch.load(path, map_location=self.device)
        self.meta_params = checkpoint['meta_params']
        self.meta_lr = checkpoint.get('meta_lr', self.meta_lr)
        self.inner_lr = checkpoint.get('inner_lr', self.inner_lr)
        self.num_inner_steps = checkpoint.get('num_inner_steps', self.num_inner_steps)
        
        # Restore base algorithm state if available
        if checkpoint.get('base_algorithm_state') and hasattr(self.base_algorithm, 'load_state_dict'):
            self.base_algorithm.load_state_dict(checkpoint['base_algorithm_state'])
        
        logger.info(f"Meta-model loaded from {path}")


class MAML(BaseMetaLearning):
    """
    Model-Agnostic Meta-Learning (MAML) implementation.
    
    MAML learns a good initialization that can be quickly adapted to new tasks
    using a few gradient steps. The meta-objective is to minimize the loss
    after adaptation to new tasks.
    """
    
    def __init__(self, base_algorithm: Any, **kwargs):
        super().__init__(base_algorithm, **kwargs)
        self.name = "MAML"
    
    def meta_update(self, tasks: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Perform MAML meta-update.
        
        MAML computes second-order gradients by differentiating through
        the inner loop adaptation process.
        """
        meta_losses = []
        
        for task in tasks:
            # Create adapted parameters for this task
            adapted_params = [param.clone() for param in self.meta_params]
            
            # Inner loop: adapt to the task
            for _ in range(self.num_inner_steps):
                loss = self._inner_loop_step(task, adapted_params)
                self._update_parameters(adapted_params, loss)
            
            # Compute loss on adapted parameters
            adapted_loss = self._compute_task_loss(task, adapted_params)
            meta_losses.append(adapted_loss)
        
        # Meta-update: update meta-parameters using gradients from adapted losses
        meta_loss = torch.stack(meta_losses).mean()
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        # Update meta parameters
        self.meta_params = self._get_parameters()
        
        # Store metrics
        self.meta_losses.append(meta_loss.item())
        
        return {
            "meta_loss": meta_loss.item(),
            "avg_task_loss": np.mean([loss.item() for loss in meta_losses]),
            "num_tasks": len(tasks)
        }
    
    def _compute_task_loss(self, task_data: Dict[str, Any], params: List[torch.Tensor]) -> torch.Tensor:
        """Compute loss for a specific task with given parameters."""
        # Temporarily set parameters
        original_params = self._get_parameters()
        self._set_parameters(params)
        
        # Compute loss (this should be implemented based on the base algorithm)
        loss = self._compute_loss_for_task(task_data)
        
        # Restore original parameters
        self._set_parameters(original_params)
        
        return loss
    
    def _compute_loss_for_task(self, task_data: Dict[str, Any]) -> torch.Tensor:
        """Compute loss for a specific task - to be implemented by subclasses."""
        # This is a placeholder - should be implemented based on the base algorithm
        return torch.tensor(0.0, requires_grad=True)


class FOMAML(BaseMetaLearning):
    """
    First-Order MAML (FOMAML) implementation.
    
    FOMAML is a simplified version of MAML that ignores second-order derivatives,
    making it computationally more efficient while often achieving similar performance.
    """
    
    def __init__(self, base_algorithm: Any, **kwargs):
        super().__init__(base_algorithm, **kwargs)
        self.name = "FOMAML"
    
    def meta_update(self, tasks: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Perform FOMAML meta-update.
        
        FOMAML approximates MAML by only using first-order gradients,
        making it computationally more efficient.
        """
        meta_losses = []
        adapted_params_list = []
        
        # First, adapt to each task and collect adapted parameters
        for task in tasks:
            adapted_params = [param.clone() for param in self.meta_params]
            
            # Inner loop: adapt to the task
            for _ in range(self.num_inner_steps):
                loss = self._inner_loop_step(task, adapted_params)
                self._update_parameters(adapted_params, loss)
            
            adapted_params_list.append(adapted_params)
            
            # Compute loss on adapted parameters
            adapted_loss = self._compute_task_loss(task, adapted_params)
            meta_losses.append(adapted_loss)
        
        # FOMAML meta-update: use gradients from adapted parameters
        # (ignoring second-order derivatives)
        meta_loss = torch.stack(meta_losses).mean()
        
        # Compute gradients from adapted parameters
        adapted_grads = []
        for adapted_params in adapted_params_list:
            grads = torch.autograd.grad(meta_loss, adapted_params, retain_graph=True)
            adapted_grads.append(grads)
        
        # Update meta-parameters using first-order approximation
        self.meta_optimizer.zero_grad()
        for i, param in enumerate(self.meta_params):
            # Average gradients across tasks
            avg_grad = torch.stack([grads[i] for grads in adapted_grads]).mean()
            param.grad = avg_grad
        
        self.meta_optimizer.step()
        
        # Update meta parameters
        self.meta_params = self._get_parameters()
        
        # Store metrics
        self.meta_losses.append(meta_loss.item())
        
        return {
            "meta_loss": meta_loss.item(),
            "avg_task_loss": np.mean([loss.item() for loss in meta_losses]),
            "num_tasks": len(tasks)
        }
    
    def _compute_task_loss(self, task_data: Dict[str, Any], params: List[torch.Tensor]) -> torch.Tensor:
        """Compute loss for a specific task with given parameters."""
        # Temporarily set parameters
        original_params = self._get_parameters()
        self._set_parameters(params)
        
        # Compute loss (this should be implemented based on the base algorithm)
        loss = self._compute_loss_for_task(task_data)
        
        # Restore original parameters
        self._set_parameters(original_params)
        
        return loss
    
    def _compute_loss_for_task(self, task_data: Dict[str, Any]) -> torch.Tensor:
        """Compute loss for a specific task - to be implemented by subclasses."""
        # This is a placeholder - should be implemented based on the base algorithm
        return torch.tensor(0.0, requires_grad=True)


class MetaLearningWrapper:
    """
    Wrapper class that applies meta-learning to any base RL algorithm.
    
    This class provides a unified interface for applying MAML or FOMAML
    to any existing RL algorithm in the project.
    """
    
    def __init__(
        self,
        base_algorithm: Any,
        meta_method: str = "maml",
        meta_lr: float = 0.01,
        inner_lr: float = 0.01,
        num_inner_steps: int = 1,
        device: str = "cpu"
    ):
        """
        Initialize meta-learning wrapper.
        
        Args:
            base_algorithm: The base RL algorithm to apply meta-learning to
            meta_method: Meta-learning method ("maml" or "fomaml")
            meta_lr: Learning rate for meta-updates
            inner_lr: Learning rate for inner loop updates
            num_inner_steps: Number of gradient steps in inner loop
            device: Device for computation
        """
        self.base_algorithm = base_algorithm
        self.meta_method = meta_method.lower()
        
        # Initialize meta-learning algorithm
        if self.meta_method == "maml":
            self.meta_learner = MAML(
                base_algorithm, 
                meta_lr=meta_lr, 
                inner_lr=inner_lr, 
                num_inner_steps=num_inner_steps, 
                device=device
            )
        elif self.meta_method == "fomaml":
            self.meta_learner = FOMAML(
                base_algorithm, 
                meta_lr=meta_lr, 
                inner_lr=inner_lr, 
                num_inner_steps=num_inner_steps, 
                device=device
            )
        else:
            raise ValueError(f"Unknown meta-learning method: {meta_method}")
        
        # Track training metrics
        self.training_metrics = {
            "meta_losses": [],
            "adaptation_metrics": [],
            "episode_rewards": [],
            "episode_lengths": []
        }
    
    def train_meta(self, tasks: List[Dict[str, Any]], num_meta_epochs: int = 100) -> Dict[str, List[float]]:
        """
        Train the meta-learning algorithm on a set of tasks.
        
        Args:
            tasks: List of task data for meta-learning
            num_meta_epochs: Number of meta-learning epochs
            
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Starting {self.meta_method.upper()} meta-learning training for {num_meta_epochs} epochs")
        
        for epoch in range(num_meta_epochs):
            # Shuffle tasks for each epoch
            np.random.shuffle(tasks)
            
            # Perform meta-update
            meta_metrics = self.meta_learner.meta_update(tasks)
            
            # Store metrics
            self.training_metrics["meta_losses"].append(meta_metrics["meta_loss"])
            
            if epoch % 10 == 0:
                logger.info(
                    f"Meta-epoch {epoch}/{num_meta_epochs}, "
                    f"Meta-loss: {meta_metrics['meta_loss']:.4f}, "
                    f"Avg task loss: {meta_metrics['avg_task_loss']:.4f}"
                )
        
        logger.info(f"Meta-learning training completed")
        return self.training_metrics
    
    def adapt_to_new_task(self, task_data: Dict[str, Any], num_adaptation_steps: int = None) -> Dict[str, float]:
        """
        Adapt the model to a new task using meta-learned parameters.
        
        Args:
            task_data: New task data
            num_adaptation_steps: Number of adaptation steps
            
        Returns:
            Dictionary of adaptation metrics
        """
        return self.meta_learner.adapt_to_task(task_data, num_adaptation_steps)
    
    def select_action(self, state: Union[np.ndarray, Dict], **kwargs):
        """Select action using the base algorithm."""
        return self.base_algorithm.select_action(state, **kwargs)
    
    def save_model(self, path: str):
        """Save the meta-learned model."""
        self.meta_learner.save_meta_model(path)
    
    def load_model(self, path: str):
        """Load the meta-learned model."""
        self.meta_learner.load_meta_model(path)
    
    def get_metrics(self) -> Dict[str, List[float]]:
        """Get training metrics."""
        return self.training_metrics.copy()

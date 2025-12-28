import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import wandb
from abc import ABC, abstractmethod
from pathlib import Path
import random
import copy

from lwmecps_gym.core.wandb_config import log_metrics
from lwmecps_gym.core.models import ModelType

logger = logging.getLogger(__name__)

class TaskDistribution:
    """Represents a distribution of tasks for meta-learning"""
    
    def __init__(
        self,
        task_types: List[str],
        task_params: Dict[str, Any],
        num_tasks: int = 10
    ):
        self.task_types = task_types
        self.task_params = task_params
        self.num_tasks = num_tasks
        
    def sample_task(self) -> Dict[str, Any]:
        """Sample a random task from the distribution"""
        task_type = random.choice(self.task_types)
        task_config = copy.deepcopy(self.task_params[task_type])
        
        # Add random variations to create different tasks
        if task_type == "kubernetes_scaling":
            task_config["max_pods"] = random.randint(5, 50)
            task_config["cpu_requirement"] = random.uniform(0.1, 2.0)
            task_config["memory_requirement"] = random.randint(100, 2000)
            
        elif task_type == "load_balancing":
            task_config["traffic_pattern"] = random.choice(["uniform", "burst", "gradual"])
            task_config["latency_threshold"] = random.uniform(0.1, 1.0)
            
        elif task_type == "resource_optimization":
            task_config["optimization_target"] = random.choice(["cpu", "memory", "latency"])
            task_config["constraint_weight"] = random.uniform(0.1, 1.0)
            
        return {
            "type": task_type,
            "config": task_config,
            "task_id": f"{task_type}_{random.randint(1000, 9999)}"
        }

class MAMLAgent:
    """
    Model-Agnostic Meta-Learning (MAML) implementation for RL agents.
    
    MAML learns to quickly adapt to new tasks by learning good initial parameters
    that can be fine-tuned with just a few gradient steps.
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
        task_distribution: Optional[TaskDistribution] = None
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.adaptation_steps = adaptation_steps
        self.meta_batch_size = meta_batch_size
        self.device = device
        self.max_replicas = max_replicas
        self.deployments = deployments or []
        
        # Create base model (meta-model)
        self.meta_model = self._create_meta_model()
        self.meta_optimizer = optim.Adam(self.meta_model.parameters(), lr=outer_lr)
        
        # Task distribution for sampling tasks
        self.task_distribution = task_distribution or self._create_default_task_distribution()
        
        # Metrics tracking
        self.metrics_collector = MAMLMetricsCollector()
        
        logger.info(f"MAML Agent initialized with obs_dim={obs_dim}, act_dim={act_dim}")
        logger.info(f"Inner LR: {inner_lr}, Outer LR: {outer_lr}")
        logger.info(f"Adaptation steps: {adaptation_steps}, Meta batch size: {meta_batch_size}")
        
    def _create_meta_model(self):
        """Create the meta-model architecture"""
        return MAMLActorCritic(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_size=self.hidden_size,
            max_replicas=self.max_replicas
        )
        
    def _create_default_task_distribution(self) -> TaskDistribution:
        """Create default task distribution for Kubernetes scenarios"""
        return TaskDistribution(
            task_types=["kubernetes_scaling", "load_balancing", "resource_optimization"],
            task_params={
                "kubernetes_scaling": {
                    "max_pods": 20,
                    "cpu_requirement": 1.0,
                    "memory_requirement": 1000,
                    "scaling_strategy": "horizontal"
                },
                "load_balancing": {
                    "traffic_pattern": "uniform",
                    "latency_threshold": 0.5,
                    "balancing_algorithm": "round_robin"
                },
                "resource_optimization": {
                    "optimization_target": "cpu",
                    "constraint_weight": 0.5,
                    "optimization_method": "gradient_based"
                }
            },
            num_tasks=10
        )
        
    def meta_train(
        self,
        env_factory,  # Function that creates environments for different tasks
        meta_episodes: int = 100,
        wandb_run_id: str = None,
        training_service=None,
        task_id: str = None,
        loop=None,
        db_connection=None
    ) -> Dict[str, List[float]]:
        """
        Meta-training loop for MAML
        
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
        logger.info(f"Starting MAML meta-training for {meta_episodes} episodes")
        
        meta_losses = []
        adaptation_accuracies = []
        task_performances = []
        
        for meta_episode in range(meta_episodes):
            logger.info(f"Meta-episode {meta_episode + 1}/{meta_episodes}")
            
            # Sample meta-batch of tasks
            meta_batch = self._sample_meta_batch()
            
            # Compute meta-gradient
            meta_loss, adaptation_results = self._compute_meta_gradient(meta_batch, env_factory)
            
            # Update meta-model
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()
            
            # Store metrics
            meta_losses.append(meta_loss.item())
            adaptation_accuracies.append(np.mean([r["accuracy"] for r in adaptation_results]))
            task_performances.append(np.mean([r["performance"] for r in adaptation_results]))
            
            # Log to wandb
            if wandb_run_id:
                log_metrics({
                    "maml/meta_loss": meta_loss.item(),
                    "maml/adaptation_accuracy": np.mean([r["accuracy"] for r in adaptation_results]),
                    "maml/task_performance": np.mean([r["performance"] for r in adaptation_results]),
                    "maml/meta_episode": meta_episode
                }, step=meta_episode)
            
            # Update progress in database
            if training_service and task_id and loop and db_connection:
                progress = (meta_episode / meta_episodes) * 100
                metrics = {
                    "meta_loss": meta_loss.item(),
                    "adaptation_accuracy": np.mean([r["accuracy"] for r in adaptation_results]),
                    "task_performance": np.mean([r["performance"] for r in adaptation_results])
                }
                
                future = asyncio.run_coroutine_threadsafe(
                    training_service.save_training_result(task_id, meta_episode, metrics, db_connection),
                    loop
                )
                future.result()
            
            # Log progress
            if meta_episode % 10 == 0:
                logger.info(f"Meta-episode {meta_episode}: Loss={meta_loss.item():.4f}, "
                           f"Adaptation Accuracy={np.mean([r['accuracy'] for r in adaptation_results]):.3f}")
        
        # Store final metrics
        self.metrics_collector.update({
            "meta_losses": meta_losses,
            "adaptation_accuracies": adaptation_accuracies,
            "task_performances": task_performances
        })
        
        logger.info("MAML meta-training completed")
        return {
            "meta_losses": meta_losses,
            "adaptation_accuracies": adaptation_accuracies,
            "task_performances": task_performances
        }
        
    def _sample_meta_batch(self) -> List[Dict[str, Any]]:
        """Sample a batch of tasks for meta-learning"""
        return [self.task_distribution.sample_task() for _ in range(self.meta_batch_size)]
        
    def _compute_meta_gradient(
        self, 
        meta_batch: List[Dict[str, Any]], 
        env_factory
    ) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
        """
        Compute meta-gradient using the MAML algorithm
        
        Returns:
            Tuple of (meta_loss, adaptation_results)
        """
        meta_losses = []
        adaptation_results = []
        
        for task in meta_batch:
            # Create environment for this task
            env = env_factory(task)
            
            # Clone meta-model for task-specific adaptation
            adapted_model = copy.deepcopy(self.meta_model)
            task_optimizer = optim.Adam(adapted_model.parameters(), lr=self.inner_lr)
            
            # Inner loop: adapt to the specific task
            task_losses = []
            for adaptation_step in range(self.adaptation_steps):
                # Sample data from task
                task_data = self._sample_task_data(env, adapted_model)
                
                # Compute task loss
                task_loss = self._compute_task_loss(task_data, adapted_model)
                
                # Update adapted model
                task_optimizer.zero_grad()
                task_loss.backward()
                task_optimizer.step()
                
                task_losses.append(task_loss.item())
            
            # Evaluate adapted model on task
            adaptation_result = self._evaluate_adapted_model(env, adapted_model, task)
            adaptation_results.append(adaptation_result)
            
            # Compute meta-loss (loss of adapted model on task)
            meta_loss = self._compute_meta_loss(env, adapted_model, task)
            meta_losses.append(meta_loss)
        
        # Average meta-loss across tasks
        total_meta_loss = torch.stack(meta_losses).mean()
        
        return total_meta_loss, adaptation_results
        
    def _sample_task_data(self, env, model, num_samples: int = 10) -> List[Dict[str, Any]]:
        """Sample data from environment using current model"""
        task_data = []
        
        for _ in range(num_samples):
            obs, info = env.reset()
            
            # Check if group is completed after reset
            if info.get("group_completed", False):
                logger.warning(f"Experiment group completed before task. Terminating training early.")
                break  # Exit training loop early
            done = False
            
            while not done:
                # Get action from model
                action = self._get_model_action(model, obs)
                
                # Take step
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Check if group is completed during task
                if info.get("group_completed", False):
                    logger.warning(f"Experiment group completed during task. Terminating training early.")
                    break  # Exit task loop early
                
                # Store experience
                task_data.append({
                    "obs": obs,
                    "action": action,
                    "reward": reward,
                    "next_obs": next_obs,
                    "done": done,
                    "info": info
                })
                
                obs = next_obs
        
        return task_data
        
    def _get_model_action(self, model, obs):
        """Get action from model"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        with torch.no_grad():
            if hasattr(model, 'select_action'):
                return model.select_action(obs_tensor)
            else:
                logits, _ = model(obs_tensor)
                if hasattr(model, 'act_dim'):
                    logits = logits.view(-1, model.act_dim, self.max_replicas + 1)
                    dist = torch.distributions.Categorical(logits=logits)
                    return dist.sample()
                else:
                    return torch.argmax(logits, dim=-1)
                    
    def _compute_task_loss(self, task_data: List[Dict[str, Any]], model) -> torch.Tensor:
        """Compute loss for a specific task"""
        losses = []
        
        for data in task_data:
            obs_tensor = torch.FloatTensor(data["obs"]).unsqueeze(0)
            action_tensor = torch.LongTensor(data["action"]).unsqueeze(0)
            reward_tensor = torch.FloatTensor([data["reward"]])
            
            # Forward pass
            logits, value = model(obs_tensor)
            
            # Compute policy loss (simplified)
            if hasattr(model, 'act_dim'):
                logits = logits.view(-1, model.act_dim, self.max_replicas + 1)
                dist = torch.distributions.Categorical(logits=logits)
                log_prob = dist.log_prob(action_tensor)
                policy_loss = -log_prob.mean()
            else:
                policy_loss = nn.CrossEntropyLoss()(logits, action_tensor.squeeze())
            
            # Value loss (simplified)
            value_loss = nn.MSELoss()(value.squeeze(), reward_tensor)
            
            # Total loss
            total_loss = policy_loss + 0.5 * value_loss
            losses.append(total_loss)
        
        return torch.stack(losses).mean()
        
    def _compute_meta_loss(self, env, adapted_model, task) -> torch.Tensor:
        """Compute meta-loss for the adapted model"""
        # Sample evaluation data
        eval_data = self._sample_task_data(env, adapted_model, num_samples=5)
        
        # Compute loss on evaluation data
        return self._compute_task_loss(eval_data, adapted_model)
        
    def _evaluate_adapted_model(self, env, adapted_model, task) -> Dict[str, float]:
        """Evaluate adapted model performance on task"""
        total_reward = 0
        num_episodes = 5
        
        for _ in range(num_episodes):
            obs, info = env.reset()
            
            # Check if group is completed after reset
            if info.get("group_completed", False):
                logger.warning(f"Experiment group completed before task. Terminating training early.")
                break  # Exit training loop early
            episode_reward = 0
            done = False
            
            while not done:
                action = self._get_model_action(adapted_model, obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Check if group is completed during task
                if info.get("group_completed", False):
                    logger.warning(f"Experiment group completed during task. Terminating training early.")
                    break  # Exit task loop early
                
                episode_reward += reward
                obs = next_obs
            
            total_reward += episode_reward
        
        avg_reward = total_reward / num_episodes
        
        return {
            "accuracy": min(1.0, max(0.0, (avg_reward + 100) / 200)),  # Normalize to [0, 1]
            "performance": avg_reward
        }
        
    def adapt_to_new_task(
        self, 
        env, 
        adaptation_steps: Optional[int] = None,
        inner_lr: Optional[float] = None
    ) -> nn.Module:
        """
        Adapt the meta-model to a new task
        
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
        optimizer = optim.Adam(adapted_model.parameters(), lr=inner_lr)
        
        logger.info(f"Adapting to new task with {adaptation_steps} steps, LR={inner_lr}")
        
        # Adaptation loop
        for step in range(adaptation_steps):
            # Sample data from new task
            task_data = self._sample_task_data(env, adapted_model, num_samples=10)
            
            # Compute loss and update
            loss = self._compute_task_loss(task_data, adapted_model)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            logger.debug(f"Adaptation step {step + 1}: Loss={loss.item():.4f}")
        
        logger.info("Task adaptation completed")
        return adapted_model
        
    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Select action using meta-model"""
        return self._get_model_action(self.meta_model, obs.numpy())
        
    def save_model(self, path: str):
        """Save the meta-model"""
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
            'model_type': 'maml'
        }, path)
        
        logger.info(f"MAML model saved to {path}")
        
    def load_model(self, path: str):
        """Load the meta-model"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.meta_model.load_state_dict(checkpoint['meta_model_state_dict'])
        self.inner_lr = checkpoint.get('inner_lr', self.inner_lr)
        self.outer_lr = checkpoint.get('outer_lr', self.outer_lr)
        self.adaptation_steps = checkpoint.get('adaptation_steps', self.adaptation_steps)
        self.meta_batch_size = checkpoint.get('meta_batch_size', self.meta_batch_size)
        self.deployments = checkpoint.get('deployments', self.deployments)
        self.max_replicas = checkpoint.get('max_replicas', self.max_replicas)
        
        logger.info(f"MAML model loaded from {path}")

class MAMLActorCritic(nn.Module):
    """Actor-Critic network for MAML"""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 256, max_replicas: int = 10):
        super().__init__()
        self.act_dim = act_dim
        self.max_replicas = max_replicas
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor head
        self.actor = nn.Linear(hidden_size, act_dim * (max_replicas + 1))
        
        # Critic head
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """Forward pass through the network"""
        features = self.feature_extractor(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value
        
    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Select action using the actor"""
        logits, _ = self(obs)
        logits = logits.view(-1, self.act_dim, self.max_replicas + 1)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample()

class MAMLMetricsCollector:
    """Collects and validates MAML specific metrics"""
    
    def __init__(self):
        self.metrics = {}
        self.metric_validators = {
            "meta_losses": lambda x: x >= 0,
            "adaptation_accuracies": lambda x: 0 <= x <= 1,
            "task_performances": lambda x: True,  # Can be any value
            "adaptation_speed": lambda x: x >= 0,
            "generalization_error": lambda x: x >= 0
        }
    
    def update(self, metrics_dict: Dict[str, Any]):
        """Update metrics with new values"""
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = []
            
            if isinstance(value, list):
                self.metrics[key].extend(value)
            else:
                self.metrics[key].append(value)
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Calculate average values for all metrics"""
        return {key: np.mean(values) for key, values in self.metrics.items() if values}
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get the most recent values for all metrics"""
        return {key: values[-1] for key, values in self.metrics.items() if values}
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()

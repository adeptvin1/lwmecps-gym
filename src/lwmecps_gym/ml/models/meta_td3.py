"""
Meta-learning implementation for TD3 algorithm.

This module provides MAML and FOMAML implementations specifically designed
for TD3 (Twin Delayed Deep Deterministic Policy Gradient) algorithm.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import gymnasium as gym

from .meta_learning import MAML, FOMAML, MetaLearningWrapper
from .td3_learning import TD3, Actor, Critic

logger = logging.getLogger(__name__)


class MetaTD3(MetaLearningWrapper):
    """
    Meta-learning wrapper for TD3 algorithm.
    
    This class applies MAML or FOMAML to TD3, enabling the algorithm
    to quickly adapt to new tasks using meta-learned parameters.
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        meta_method: str = "maml",
        hidden_size: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_delay: int = 2,
        noise_clip: float = 0.5,
        noise: float = 0.2,
        batch_size: int = 256,
        device: str = "cpu",
        deployments: List[str] = None,
        max_replicas: int = 10,
        # Meta-learning parameters
        meta_lr: float = 0.01,
        inner_lr: float = 0.01,
        num_inner_steps: int = 1
    ):
        """
        Initialize MetaTD3.
        
        Args:
            obs_dim: Observation dimension
            act_dim: Action dimension
            meta_method: Meta-learning method ("maml" or "fomaml")
            hidden_size: Hidden layer size
            lr: Learning rate for TD3
            gamma: Discount factor
            tau: Soft update coefficient
            policy_delay: Policy update delay
            noise_clip: Noise clipping parameter
            noise: Action noise
            batch_size: Batch size for training
            device: Device for computation
            deployments: List of deployments
            max_replicas: Maximum number of replicas per deployment
            meta_lr: Learning rate for meta-updates
            inner_lr: Learning rate for inner loop updates
            num_inner_steps: Number of gradient steps in inner loop
        """
        # Initialize base TD3 algorithm
        base_td3 = TD3(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_size=hidden_size,
            lr=lr,
            gamma=gamma,
            tau=tau,
            policy_delay=policy_delay,
            noise_clip=noise_clip,
            noise=noise,
            batch_size=batch_size,
            device=device,
            deployments=deployments,
            max_replicas=max_replicas
        )
        
        # Initialize meta-learning wrapper
        super().__init__(
            base_algorithm=base_td3,
            meta_method=meta_method,
            meta_lr=meta_lr,
            inner_lr=inner_lr,
            num_inner_steps=num_inner_steps,
            device=device
        )
        
        # Store TD3-specific parameters
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.noise_clip = noise_clip
        self.noise = noise
        self.batch_size = batch_size
        self.deployments = deployments or []
        self.max_replicas = max_replicas
        
        # Initialize specialized meta-learners
        if meta_method.lower() == "maml":
            self.meta_learner = MAMLTD3(
                base_td3, meta_lr, inner_lr, num_inner_steps, device
            )
        elif meta_method.lower() == "fomaml":
            self.meta_learner = FOMAMLTD3(
                base_td3, meta_lr, inner_lr, num_inner_steps, device
            )
        else:
            raise ValueError(f"Unknown meta-learning method: {meta_method}")
    
    def train_meta_on_tasks(self, tasks: List[Dict[str, Any]], num_meta_epochs: int = 100) -> Dict[str, List[float]]:
        """
        Train meta-learning on a set of tasks.
        
        Args:
            tasks: List of task data, each containing:
                - 'env': Environment instance
                - 'episodes': Number of episodes to train on
                - 'task_id': Unique task identifier
            num_meta_epochs: Number of meta-learning epochs
            
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Starting {self.meta_method.upper()} meta-learning training for TD3")
        
        for epoch in range(num_meta_epochs):
            epoch_metrics = []
            
            for task in tasks:
                # Train on this task using meta-learned parameters
                task_metrics = self._train_on_task(task)
                epoch_metrics.append(task_metrics)
            
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
        
        logger.info("Meta-learning training completed")
        return self.training_metrics
    
    def _train_on_task(self, task: Dict[str, Any]) -> Dict[str, float]:
        """
        Train on a specific task using current meta-parameters.
        
        Args:
            task: Task data containing environment and training parameters
            
        Returns:
            Dictionary of task-specific metrics
        """
        env = task['env']
        episodes = task.get('episodes', 100)
        
        # Collect data using current parameters
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0
            ep_length = 0
            
            # Collect one episode
            while not done:
                action = self.base_algorithm.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Store transition in replay buffer
                self.base_algorithm.replay_buffer.append((obs, action, reward, next_obs, done))
                if len(self.base_algorithm.replay_buffer) > self.base_algorithm.max_buffer_size:
                    self.base_algorithm.replay_buffer.pop(0)
                
                obs = next_obs
                ep_reward += reward
                ep_length += 1
                
                if done:
                    break
            
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
        
        return {
            "avg_reward": np.mean(episode_rewards),
            "avg_length": np.mean(episode_lengths),
            "episodes": episodes
        }
    
    def adapt_to_new_task(self, env: gym.Env, num_adaptation_episodes: int = 10) -> Dict[str, float]:
        """
        Adapt to a new task using meta-learned parameters.
        
        Args:
            env: New environment/task
            num_adaptation_episodes: Number of episodes for adaptation
            
        Returns:
            Dictionary of adaptation metrics
        """
        task_data = {
            'env': env,
            'episodes': num_adaptation_episodes,
            'task_id': f"adaptation_{np.random.randint(10000)}"
        }
        
        # Perform adaptation
        adaptation_metrics = self.meta_learner.adapt_to_task(task_data)
        
        # Test adapted performance
        test_metrics = self._test_adapted_performance(env, num_test_episodes=5)
        
        return {
            **adaptation_metrics,
            **test_metrics
        }
    
    def _test_adapted_performance(self, env: gym.Env, num_test_episodes: int = 5) -> Dict[str, float]:
        """
        Test the performance of adapted parameters.
        
        Args:
            env: Environment to test on
            num_test_episodes: Number of test episodes
            
        Returns:
            Dictionary of test metrics
        """
        test_rewards = []
        test_lengths = []
        
        for episode in range(num_test_episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0
            ep_length = 0
            
            while not done:
                action = self.base_algorithm.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                obs = next_obs
                ep_reward += reward
                ep_length += 1
                
                if done:
                    break
            
            test_rewards.append(ep_reward)
            test_lengths.append(ep_length)
        
        return {
            "test_avg_reward": np.mean(test_rewards),
            "test_avg_length": np.mean(test_lengths),
            "test_std_reward": np.std(test_rewards),
            "test_episodes": num_test_episodes
        }


class MAMLTD3(MAML):
    """
    MAML implementation specifically for TD3.
    
    This class implements the inner loop adaptation and loss computation
    specific to TD3 algorithm.
    """
    
    def _inner_loop_step(self, task_data: Dict[str, Any], params: List[torch.Tensor]) -> torch.Tensor:
        """
        Perform one step of inner loop adaptation for TD3.
        
        Args:
            task_data: Task-specific data
            params: Current parameters to update
            
        Returns:
            Loss value for this step
        """
        env = task_data['env']
        
        # Temporarily set parameters
        original_params = self._get_parameters()
        self._set_parameters(params)
        
        # Collect data and compute TD3 loss
        loss = self._compute_td3_loss(env)
        
        # Restore original parameters
        self._set_parameters(original_params)
        
        return loss
    
    def _compute_td3_loss(self, env: gym.Env) -> torch.Tensor:
        """
        Compute TD3 loss from environment data.
        
        Args:
            env: Environment to collect data from
            
        Returns:
            TD3 loss value
        """
        # Collect a batch of data
        if len(self.base_algorithm.replay_buffer) < self.base_algorithm.batch_size:
            return torch.tensor(0.0, requires_grad=True)
        
        # Sample from replay buffer
        indices = np.random.choice(len(self.base_algorithm.replay_buffer), self.base_algorithm.batch_size, replace=False)
        
        # Convert observations to numpy arrays first, then to tensor
        obs_batch = np.array([self.base_algorithm._flatten_observation(self.base_algorithm.replay_buffer[i][0]) for i in indices])
        obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        
        # Convert actions to numpy array first
        act_batch = np.array([self.base_algorithm.replay_buffer[i][1] for i in indices])
        act_batch = torch.FloatTensor(act_batch).to(self.device)
        
        # Convert rewards and done flags to numpy arrays first
        rew_batch = np.array([float(self.base_algorithm.replay_buffer[i][2]) for i in indices])
        rew_batch = torch.FloatTensor(rew_batch).unsqueeze(1).to(self.device)
        
        next_obs_batch = np.array([self.base_algorithm._flatten_observation(self.base_algorithm.replay_buffer[i][3]) for i in indices])
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)
        
        done_batch = np.array([float(self.base_algorithm.replay_buffer[i][4]) for i in indices])
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)
        
        # Update critics
        with torch.no_grad():
            next_actions = self.base_algorithm.actor_target(next_obs_batch)
            target_q1 = self.base_algorithm.critic1_target(next_obs_batch, next_actions.float())
            target_q2 = self.base_algorithm.critic2_target(next_obs_batch, next_actions.float())
            target_q = torch.min(target_q1, target_q2)
            target_q = rew_batch + (1 - done_batch) * self.base_algorithm.gamma * target_q
        
        current_q1 = self.base_algorithm.critic1(obs_batch, act_batch)
        current_q2 = self.base_algorithm.critic2(obs_batch, act_batch)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        # Policy delay: Update actor less frequently
        actor_loss = 0.0
        if self.base_algorithm.update_count % self.base_algorithm.policy_delay == 0:
            actor_actions = self.base_algorithm.actor(obs_batch)
            actor_loss = -self.base_algorithm.critic1(obs_batch, actor_actions.float()).mean()
        
        total_loss = actor_loss + critic1_loss + critic2_loss
        
        return total_loss
    
    def _compute_loss_for_task(self, task_data: Dict[str, Any]) -> torch.Tensor:
        """
        Compute loss for a specific task.
        
        Args:
            task_data: Task data
            
        Returns:
            Loss value
        """
        env = task_data['env']
        return self._compute_td3_loss(env)


class FOMAMLTD3(FOMAML):
    """
    FOMAML implementation specifically for TD3.
    
    This class inherits from FOMAML and implements TD3-specific
    inner loop adaptation and loss computation.
    """
    
    def _inner_loop_step(self, task_data: Dict[str, Any], params: List[torch.Tensor]) -> torch.Tensor:
        """
        Perform one step of inner loop adaptation for TD3.
        
        Args:
            task_data: Task-specific data
            params: Current parameters to update
            
        Returns:
            Loss value for this step
        """
        env = task_data['env']
        
        # Temporarily set parameters
        original_params = self._get_parameters()
        self._set_parameters(params)
        
        # Collect data and compute TD3 loss
        loss = self._compute_td3_loss(env)
        
        # Restore original parameters
        self._set_parameters(original_params)
        
        return loss
    
    def _compute_td3_loss(self, env: gym.Env) -> torch.Tensor:
        """
        Compute TD3 loss from environment data.
        
        Args:
            env: Environment to collect data from
            
        Returns:
            TD3 loss value
        """
        # Collect a batch of data
        if len(self.base_algorithm.replay_buffer) < self.base_algorithm.batch_size:
            return torch.tensor(0.0, requires_grad=True)
        
        # Sample from replay buffer
        indices = np.random.choice(len(self.base_algorithm.replay_buffer), self.base_algorithm.batch_size, replace=False)
        
        # Convert observations to numpy arrays first, then to tensor
        obs_batch = np.array([self.base_algorithm._flatten_observation(self.base_algorithm.replay_buffer[i][0]) for i in indices])
        obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        
        # Convert actions to numpy array first
        act_batch = np.array([self.base_algorithm.replay_buffer[i][1] for i in indices])
        act_batch = torch.FloatTensor(act_batch).to(self.device)
        
        # Convert rewards and done flags to numpy arrays first
        rew_batch = np.array([float(self.base_algorithm.replay_buffer[i][2]) for i in indices])
        rew_batch = torch.FloatTensor(rew_batch).unsqueeze(1).to(self.device)
        
        next_obs_batch = np.array([self.base_algorithm._flatten_observation(self.base_algorithm.replay_buffer[i][3]) for i in indices])
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)
        
        done_batch = np.array([float(self.base_algorithm.replay_buffer[i][4]) for i in indices])
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)
        
        # Update critics
        with torch.no_grad():
            next_actions = self.base_algorithm.actor_target(next_obs_batch)
            target_q1 = self.base_algorithm.critic1_target(next_obs_batch, next_actions.float())
            target_q2 = self.base_algorithm.critic2_target(next_obs_batch, next_actions.float())
            target_q = torch.min(target_q1, target_q2)
            target_q = rew_batch + (1 - done_batch) * self.base_algorithm.gamma * target_q
        
        current_q1 = self.base_algorithm.critic1(obs_batch, act_batch)
        current_q2 = self.base_algorithm.critic2(obs_batch, act_batch)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        # Policy delay: Update actor less frequently
        actor_loss = 0.0
        if self.base_algorithm.update_count % self.base_algorithm.policy_delay == 0:
            actor_actions = self.base_algorithm.actor(obs_batch)
            actor_loss = -self.base_algorithm.critic1(obs_batch, actor_actions.float()).mean()
        
        total_loss = actor_loss + critic1_loss + critic2_loss
        
        return total_loss
    
    def _compute_loss_for_task(self, task_data: Dict[str, Any]) -> torch.Tensor:
        """
        Compute loss for a specific task.
        
        Args:
            task_data: Task data
            
        Returns:
            Loss value
        """
        env = task_data['env']
        return self._compute_td3_loss(env)

"""
Meta-learning implementation for PPO algorithm.

This module provides MAML and FOMAML implementations specifically designed
for PPO (Proximal Policy Optimization) algorithm.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import gymnasium as gym

from .meta_learning import MAML, FOMAML, MetaLearningWrapper
from .ppo_learning import PPO, ActorCritic, RolloutBuffer

logger = logging.getLogger(__name__)


class MetaPPO(MetaLearningWrapper):
    """
    Meta-learning wrapper for PPO algorithm.
    
    This class applies MAML or FOMAML to PPO, enabling the algorithm
    to quickly adapt to new tasks using meta-learned parameters.
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        meta_method: str = "maml",
        hidden_size: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        device: str = "cpu",
        deployments: List[str] = None,
        max_replicas: int = 10,
        # Meta-learning parameters
        meta_lr: float = 0.01,
        inner_lr: float = 0.01,
        num_inner_steps: int = 1
    ):
        """
        Initialize MetaPPO.
        
        Args:
            obs_dim: Observation dimension
            act_dim: Action dimension
            meta_method: Meta-learning method ("maml" or "fomaml")
            hidden_size: Hidden layer size
            lr: Learning rate for PPO
            gamma: Discount factor
            lam: Lambda parameter for GAE
            clip_eps: Clipping parameter for PPO
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            n_steps: Number of steps to collect before update
            batch_size: Batch size for update
            n_epochs: Number of update epochs
            device: Device for computation
            deployments: List of deployments
            max_replicas: Maximum number of replicas per deployment
            meta_lr: Learning rate for meta-updates
            inner_lr: Learning rate for inner loop updates
            num_inner_steps: Number of gradient steps in inner loop
        """
        # Initialize base PPO algorithm
        base_ppo = PPO(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_size=hidden_size,
            lr=lr,
            gamma=gamma,
            lam=lam,
            clip_eps=clip_eps,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            device=device,
            deployments=deployments,
            max_replicas=max_replicas
        )
        
        # Initialize meta-learning wrapper
        super().__init__(
            base_algorithm=base_ppo,
            meta_method=meta_method,
            meta_lr=meta_lr,
            inner_lr=inner_lr,
            num_inner_steps=num_inner_steps,
            device=device
        )
        
        # Store PPO-specific parameters
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.lr = lr
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.deployments = deployments or []
        self.max_replicas = max_replicas
        
        # Initialize specialized meta-learners
        if meta_method.lower() == "maml":
            self.meta_learner = MAMLPPO(
                base_ppo, meta_lr, inner_lr, num_inner_steps, device
            )
        elif meta_method.lower() == "fomaml":
            self.meta_learner = FOMAMLPPO(
                base_ppo, meta_lr, inner_lr, num_inner_steps, device
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
        logger.info(f"Starting {self.meta_method.upper()} meta-learning training for PPO")
        
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
        
        # Collect trajectories using current parameters
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0
            ep_length = 0
            
            # Collect one episode
            while not done:
                action, log_prob, value = self.base_algorithm.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
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
                action, log_prob, value = self.base_algorithm.select_action(obs)
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


class MAMLPPO(MAML):
    """
    MAML implementation specifically for PPO.
    
    This class implements the inner loop adaptation and loss computation
    specific to PPO algorithm.
    """
    
    def _inner_loop_step(self, task_data: Dict[str, Any], params: List[torch.Tensor]) -> torch.Tensor:
        """
        Perform one step of inner loop adaptation for PPO.
        
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
        
        # Collect trajectories
        trajectories = self._collect_trajectories(env)
        
        # Compute PPO loss
        loss = self._compute_ppo_loss(trajectories)
        
        # Restore original parameters
        self._set_parameters(original_params)
        
        return loss
    
    def _collect_trajectories(self, env: gym.Env) -> Dict[str, torch.Tensor]:
        """
        Collect trajectories from environment.
        
        Args:
            env: Environment to collect from
            
        Returns:
            Dictionary of trajectory data
        """
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, log_prob, value = self.base_algorithm.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            states.append(self.base_algorithm._flatten_observation(obs))
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            dones.append(done)
            
            obs = next_obs
            
            if done:
                break
        
        return {
            'states': torch.FloatTensor(np.array(states)).to(self.device),
            'actions': torch.LongTensor(np.array(actions)).to(self.device),
            'rewards': torch.FloatTensor(np.array(rewards)).to(self.device),
            'values': torch.FloatTensor(np.array(values)).to(self.device),
            'log_probs': torch.FloatTensor(np.array(log_probs)).to(self.device),
            'dones': torch.FloatTensor(np.array(dones)).to(self.device)
        }
    
    def _compute_ppo_loss(self, trajectories: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute PPO loss from trajectories.
        
        Args:
            trajectories: Dictionary of trajectory data
            
        Returns:
            PPO loss value
        """
        states = trajectories['states']
        actions = trajectories['actions']
        rewards = trajectories['rewards']
        old_values = trajectories['values']
        old_log_probs = trajectories['log_probs']
        dones = trajectories['dones']
        
        # Compute advantages and returns
        advantages, returns = self._compute_advantages_returns(rewards, old_values, dones)
        
        # Get current policy and value estimates
        logits, values = self.base_algorithm.model(states)
        logits = logits.view(-1, self.base_algorithm.act_dim, self.base_algorithm.max_replicas + 1)
        
        # Compute new log probabilities
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        
        # Compute entropy
        entropy = dist.entropy().mean()
        
        # Compute PPO loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.base_algorithm.clip_eps, 1 + self.base_algorithm.clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        critic_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()
        
        # Total loss
        total_loss = actor_loss + self.base_algorithm.vf_coef * critic_loss - self.base_algorithm.ent_coef * entropy
        
        return total_loss
    
    def _compute_advantages_returns(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages and returns using GAE.
        
        Args:
            rewards: Reward tensor
            values: Value estimates tensor
            dones: Done flags tensor
            
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.base_algorithm.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.base_algorithm.gamma * self.base_algorithm.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def _compute_loss_for_task(self, task_data: Dict[str, Any]) -> torch.Tensor:
        """
        Compute loss for a specific task.
        
        Args:
            task_data: Task data
            
        Returns:
            Loss value
        """
        env = task_data['env']
        trajectories = self._collect_trajectories(env)
        return self._compute_ppo_loss(trajectories)


class FOMAMLPPO(FOMAML):
    """
    FOMAML implementation specifically for PPO.
    
    This class inherits from FOMAML and implements PPO-specific
    inner loop adaptation and loss computation.
    """
    
    def _inner_loop_step(self, task_data: Dict[str, Any], params: List[torch.Tensor]) -> torch.Tensor:
        """
        Perform one step of inner loop adaptation for PPO.
        
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
        
        # Collect trajectories
        trajectories = self._collect_trajectories(env)
        
        # Compute PPO loss
        loss = self._compute_ppo_loss(trajectories)
        
        # Restore original parameters
        self._set_parameters(original_params)
        
        return loss
    
    def _collect_trajectories(self, env: gym.Env) -> Dict[str, torch.Tensor]:
        """
        Collect trajectories from environment.
        
        Args:
            env: Environment to collect from
            
        Returns:
            Dictionary of trajectory data
        """
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, log_prob, value = self.base_algorithm.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            states.append(self.base_algorithm._flatten_observation(obs))
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            dones.append(done)
            
            obs = next_obs
            
            if done:
                break
        
        return {
            'states': torch.FloatTensor(np.array(states)).to(self.device),
            'actions': torch.LongTensor(np.array(actions)).to(self.device),
            'rewards': torch.FloatTensor(np.array(rewards)).to(self.device),
            'values': torch.FloatTensor(np.array(values)).to(self.device),
            'log_probs': torch.FloatTensor(np.array(log_probs)).to(self.device),
            'dones': torch.FloatTensor(np.array(dones)).to(self.device)
        }
    
    def _compute_ppo_loss(self, trajectories: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute PPO loss from trajectories.
        
        Args:
            trajectories: Dictionary of trajectory data
            
        Returns:
            PPO loss value
        """
        states = trajectories['states']
        actions = trajectories['actions']
        rewards = trajectories['rewards']
        old_values = trajectories['values']
        old_log_probs = trajectories['log_probs']
        dones = trajectories['dones']
        
        # Compute advantages and returns
        advantages, returns = self._compute_advantages_returns(rewards, old_values, dones)
        
        # Get current policy and value estimates
        logits, values = self.base_algorithm.model(states)
        logits = logits.view(-1, self.base_algorithm.act_dim, self.base_algorithm.max_replicas + 1)
        
        # Compute new log probabilities
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        
        # Compute entropy
        entropy = dist.entropy().mean()
        
        # Compute PPO loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.base_algorithm.clip_eps, 1 + self.base_algorithm.clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        critic_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()
        
        # Total loss
        total_loss = actor_loss + self.base_algorithm.vf_coef * critic_loss - self.base_algorithm.ent_coef * entropy
        
        return total_loss
    
    def _compute_advantages_returns(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages and returns using GAE.
        
        Args:
            rewards: Reward tensor
            values: Value estimates tensor
            dones: Done flags tensor
            
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.base_algorithm.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.base_algorithm.gamma * self.base_algorithm.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def _compute_loss_for_task(self, task_data: Dict[str, Any]) -> torch.Tensor:
        """
        Compute loss for a specific task.
        
        Args:
            task_data: Task data
            
        Returns:
            Loss value
        """
        env = task_data['env']
        trajectories = self._collect_trajectories(env)
        return self._compute_ppo_loss(trajectories)

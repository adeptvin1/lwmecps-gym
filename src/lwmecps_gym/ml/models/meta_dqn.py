"""
Meta-learning implementation for DQN algorithm.

This module provides MAML and FOMAML implementations specifically designed
for DQN (Deep Q-Network) algorithm.
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
from .dq_learning import DQNAgent, DQN, ReplayBuffer

logger = logging.getLogger(__name__)


class MetaDQN(MetaLearningWrapper):
    """
    Meta-learning wrapper for DQN algorithm.
    
    This class applies MAML or FOMAML to DQN, enabling the algorithm
    to quickly adapt to new tasks using meta-learned parameters.
    """
    
    def __init__(
        self,
        env: gym.Env,
        meta_method: str = "maml",
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
        memory_size: int = 10000,
        batch_size: int = 32,
        max_replicas: int = 10,
        device: str = "cpu",
        # Meta-learning parameters
        meta_lr: float = 0.01,
        inner_lr: float = 0.01,
        num_inner_steps: int = 1
    ):
        """
        Initialize MetaDQN.
        
        Args:
            env: Environment instance
            meta_method: Meta-learning method ("maml" or "fomaml")
            learning_rate: Learning rate for DQN
            discount_factor: Discount factor
            epsilon: Exploration rate
            memory_size: Replay buffer size
            batch_size: Batch size for training
            max_replicas: Maximum number of replicas per deployment
            device: Device for computation
            meta_lr: Learning rate for meta-updates
            inner_lr: Learning rate for inner loop updates
            num_inner_steps: Number of gradient steps in inner loop
        """
        # Initialize base DQN algorithm
        replay_buffer = ReplayBuffer(memory_size)
        base_dqn = DQNAgent(
            env=env,
            replay_buffer=replay_buffer,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            memory_size=memory_size,
            batch_size=batch_size
        )
        
        # Initialize meta-learning wrapper
        super().__init__(
            base_algorithm=base_dqn,
            meta_method=meta_method,
            meta_lr=meta_lr,
            inner_lr=inner_lr,
            num_inner_steps=num_inner_steps,
            device=device
        )
        
        # Store DQN-specific parameters
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.max_replicas = max_replicas
        
        # Initialize specialized meta-learners
        if meta_method.lower() == "maml":
            self.meta_learner = MAMLDQN(
                base_dqn, meta_lr, inner_lr, num_inner_steps, device
            )
        elif meta_method.lower() == "fomaml":
            self.meta_learner = FOMAMLDQN(
                base_dqn, meta_lr, inner_lr, num_inner_steps, device
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
        logger.info(f"Starting {self.meta_method.upper()} meta-learning training for DQN")
        
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
            state, _ = env.reset()
            done = False
            ep_reward = 0
            ep_length = 0
            
            # Collect one episode
            while not done:
                action = self.base_algorithm.act(state, self.base_algorithm.epsilon)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Store transition in replay buffer
                self.base_algorithm.replay_buffer.push(state, action, reward, next_state, done)
                
                state = next_state
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
            state, _ = env.reset()
            done = False
            ep_reward = 0
            ep_length = 0
            
            while not done:
                action = self.base_algorithm.act(state, 0.0)  # No exploration during testing
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                state = next_state
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


class MAMLDQN(MAML):
    """
    MAML implementation specifically for DQN.
    
    This class implements the inner loop adaptation and loss computation
    specific to DQN algorithm.
    """
    
    def _inner_loop_step(self, task_data: Dict[str, Any], params: List[torch.Tensor]) -> torch.Tensor:
        """
        Perform one step of inner loop adaptation for DQN.
        
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
        
        # Collect data and compute DQN loss
        loss = self._compute_dqn_loss(env)
        
        # Restore original parameters
        self._set_parameters(original_params)
        
        return loss
    
    def _compute_dqn_loss(self, env: gym.Env) -> torch.Tensor:
        """
        Compute DQN loss from environment data.
        
        Args:
            env: Environment to collect data from
            
        Returns:
            DQN loss value
        """
        # Collect a batch of data
        if self.base_algorithm.replay_buffer.size() < self.base_algorithm.batch_size:
            return torch.tensor(0.0, requires_grad=True)
        
        # Sample from replay buffer
        state, action, reward, next_state, done = self.base_algorithm.replay_buffer.sample(self.base_algorithm.batch_size)
        
        # Flatten observations
        state = torch.FloatTensor([self.base_algorithm._flatten_observation(s) for s in state]).to(self.device)
        next_state = torch.FloatTensor([self.base_algorithm._flatten_observation(s) for s in next_state]).to(self.device)
        
        # Handle array actions (for MultiDiscrete action space)
        if isinstance(action[0], (list, tuple, np.ndarray)):
            # For MultiDiscrete action space, use the average action value for training
            action = torch.LongTensor([int(np.mean(a)) if len(a) > 0 else 0 for a in action]).to(self.device)
        else:
            action = torch.LongTensor(action).to(self.device)
            
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        q_values = self.base_algorithm.model(state)
        next_q_values = self.base_algorithm.target_model(next_state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.base_algorithm.discount_factor * next_q_value * (1 - done)

        loss = (q_value - expected_q_value.detach()).pow(2).mean()
        
        return loss
    
    def _compute_loss_for_task(self, task_data: Dict[str, Any]) -> torch.Tensor:
        """
        Compute loss for a specific task.
        
        Args:
            task_data: Task data
            
        Returns:
            Loss value
        """
        env = task_data['env']
        return self._compute_dqn_loss(env)


class FOMAMLDQN(FOMAML):
    """
    FOMAML implementation specifically for DQN.
    
    This class inherits from FOMAML and implements DQN-specific
    inner loop adaptation and loss computation.
    """
    
    def _inner_loop_step(self, task_data: Dict[str, Any], params: List[torch.Tensor]) -> torch.Tensor:
        """
        Perform one step of inner loop adaptation for DQN.
        
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
        
        # Collect data and compute DQN loss
        loss = self._compute_dqn_loss(env)
        
        # Restore original parameters
        self._set_parameters(original_params)
        
        return loss
    
    def _compute_dqn_loss(self, env: gym.Env) -> torch.Tensor:
        """
        Compute DQN loss from environment data.
        
        Args:
            env: Environment to collect data from
            
        Returns:
            DQN loss value
        """
        # Collect a batch of data
        if self.base_algorithm.replay_buffer.size() < self.base_algorithm.batch_size:
            return torch.tensor(0.0, requires_grad=True)
        
        # Sample from replay buffer
        state, action, reward, next_state, done = self.base_algorithm.replay_buffer.sample(self.base_algorithm.batch_size)
        
        # Flatten observations
        state = torch.FloatTensor([self.base_algorithm._flatten_observation(s) for s in state]).to(self.device)
        next_state = torch.FloatTensor([self.base_algorithm._flatten_observation(s) for s in next_state]).to(self.device)
        
        # Handle array actions (for MultiDiscrete action space)
        if isinstance(action[0], (list, tuple, np.ndarray)):
            # For MultiDiscrete action space, use the average action value for training
            action = torch.LongTensor([int(np.mean(a)) if len(a) > 0 else 0 for a in action]).to(self.device)
        else:
            action = torch.LongTensor(action).to(self.device)
            
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        q_values = self.base_algorithm.model(state)
        next_q_values = self.base_algorithm.target_model(next_state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.base_algorithm.discount_factor * next_q_value * (1 - done)

        loss = (q_value - expected_q_value.detach()).pow(2).mean()
        
        return loss
    
    def _compute_loss_for_task(self, task_data: Dict[str, Any]) -> torch.Tensor:
        """
        Compute loss for a specific task.
        
        Args:
            task_data: Task data
            
        Returns:
            Loss value
        """
        env = task_data['env']
        return self._compute_dqn_loss(env)

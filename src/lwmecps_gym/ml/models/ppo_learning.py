import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gymnasium import spaces
import logging
from typing import List, Dict, Union, Tuple, Optional
import wandb
import asyncio

from lwmecps_gym.envs import LWMECPSEnv
from lwmecps_gym.core.wandb_config import log_metrics

logger = logging.getLogger(__name__)

class ActorCritic(nn.Module):
    """
    Neural network architecture for PPO, consisting of two parts:
    1. Actor (policy) - determines action distribution
    2. Critic (value function) - estimates expected reward
    
    Args:
        obs_dim (int): Observation vector dimension
        act_dim (int): Action space dimension
        hidden_size (int): Hidden layer size
        max_replicas (int): Maximum number of replicas per deployment
    """
    def __init__(self, obs_dim, act_dim, hidden_size=64, max_replicas=10):
        super().__init__()
        self.act_dim = act_dim
        self.max_replicas = max_replicas

        # Validate dimensions
        if act_dim <= 0:
            raise ValueError(f"act_dim must be positive, got {act_dim}")
        if max_replicas <= 0:
            raise ValueError(f"max_replicas must be positive, got {max_replicas}")

        # Actor (Policy)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, act_dim * (max_replicas + 1))  # Output logits for each action
        )

        # Critic (Value function)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input state tensor of shape [batch_size, obs_dim]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (action logits of shape [batch_size, act_dim * (max_replicas + 1)], 
                                             value estimate of shape [batch_size, 1])
        """
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input tensor, got shape {x.shape}")
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

    def get_action_and_value(self, x):
        """
        Get action, its log probability, and value estimate.
        
        Args:
            x (torch.Tensor): Input state tensor of shape [batch_size, obs_dim]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.distributions.Distribution, torch.Tensor]:
                (action of shape [batch_size, act_dim],
                 log_prob of shape [batch_size],
                 distribution,
                 value estimate of shape [batch_size, 1])
        """
        logits, value = self(x)
        # Reshape logits to [batch_size, num_deployments, num_actions]
        logits = logits.view(-1, self.act_dim, self.max_replicas + 1)
        # Create categorical distribution for each deployment
        dist = torch.distributions.Categorical(logits=logits)
        # Sample actions
        action = dist.sample()
        # Calculate log probability
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, dist, value


class RolloutBuffer:
    """
    Buffer for storing experience trajectories.
    Used to collect data before policy update.
    
    Args:
        n_steps (int): Number of steps to collect before update
        obs_dim (int): Observation vector dimension
        act_dim (int): Action space dimension
    """
    def __init__(self, n_steps, obs_dim, act_dim):
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        if obs_dim <= 0:
            raise ValueError(f"obs_dim must be positive, got {obs_dim}")
        if act_dim <= 0:
            raise ValueError(f"act_dim must be positive, got {act_dim}")
            
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.reset()

    def reset(self):
        """Reset buffer for new data collection."""
        self.states = np.zeros((self.n_steps, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, self.act_dim), dtype=np.int32)
        self.rewards = np.zeros(self.n_steps, dtype=np.float32)
        self.values = np.zeros(self.n_steps, dtype=np.float32)
        self.log_probs = np.zeros(self.n_steps, dtype=np.float32)
        self.dones = np.zeros(self.n_steps, dtype=np.float32)
        self.advantages = np.zeros(self.n_steps, dtype=np.float32)
        self.returns = np.zeros(self.n_steps, dtype=np.float32)
        self.pos = 0

    def add(self, state, action, reward, value, log_prob, done):
        """
        Add one step of experience to the buffer.
        
        Args:
            state (np.ndarray): State of shape [obs_dim]
            action (np.ndarray): Selected action of shape [act_dim]
            reward (float): Received reward
            value (float): State value estimate
            log_prob (float): Action log probability
            done (bool): Episode done flag
        """
        if self.pos >= self.n_steps:
            raise ValueError("Buffer is full")
            
        # Validate shapes
        if state.shape != (self.obs_dim,):
            raise ValueError(f"Expected state shape {(self.obs_dim,)}, got {state.shape}")
        if action.shape != (self.act_dim,):
            raise ValueError(f"Expected action shape {(self.act_dim,)}, got {action.shape}")
            
        # Validate types
        if not isinstance(reward, (int, float)):
            raise TypeError(f"Expected reward to be numeric, got {type(reward)}")
        if not isinstance(value, (int, float)):
            raise TypeError(f"Expected value to be numeric, got {type(value)}")
        if not isinstance(log_prob, (int, float)):
            raise TypeError(f"Expected log_prob to be numeric, got {type(log_prob)}")
        if not isinstance(done, bool):
            raise TypeError(f"Expected done to be bool, got {type(done)}")
            
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.dones[self.pos] = done
        self.pos += 1

    def is_full(self):
        """Check if buffer is full."""
        return self.pos >= self.n_steps

    def compute_advantages(self, last_value: float, gamma: float, lam: float):
        """
        Compute advantages using GAE.
        
        Args:
            last_value (float): Value estimate of the state after the last one in the buffer
            gamma (float): Discount factor
            lam (float): Lambda parameter for GAE
        """
        gae = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values


class MetricsCollector:
    def __init__(self):
        self.metrics = {}
        self.metric_validators = {
            "accuracy": lambda x: 0 <= x <= 1,
            "mse": lambda x: x >= 0,
            "mre": lambda x: x >= 0,
            "avg_latency": lambda x: x >= 0,
            "total_reward": lambda x: True,  # Can be negative
            "value": lambda x: True,  # Can be negative
            "log_prob": lambda x: True,  # Can be negative
            "actor_loss": lambda x: x >= 0,
            "critic_loss": lambda x: x >= 0
        }
    
    def update(self, metrics_dict: Dict[str, float]):
        """Update metrics with new values."""
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = []
            if key in self.metric_validators:
                if not self.metric_validators[key](value):
                    logger.warning(f"Invalid value for metric {key}: {value}")
                    continue
            self.metrics[key].append(value)
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Calculate average values for all metrics."""
        return {key: np.mean(values) for key, values in self.metrics.items() if values}
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get the most recent values for all metrics."""
        return {key: values[-1] for key, values in self.metrics.items() if values}
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()


class PPO:
    """
    Implementation of Proximal Policy Optimization algorithm.
    
    Args:
        obs_dim (int): Observation vector dimension
        act_dim (int): Action space dimension
        hidden_size (int): Hidden layer size
        lr (float): Learning rate
        gamma (float): Discount factor
        lam (float): Lambda parameter for GAE
        clip_eps (float): Clipping parameter for PPO
        ent_coef (float): Entropy coefficient
        vf_coef (float): Value function coefficient
        n_steps (int): Number of steps to collect before update
        batch_size (int): Batch size for update
        n_epochs (int): Number of update epochs
        device (str): Device for computation (cpu/cuda)
        deployments (List[str]): List of deployments to process
        max_replicas (int): Maximum number of replicas per deployment
    """
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
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
        max_replicas: int = 10
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lr = lr
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device
        self.deployments = deployments or []
        self.max_replicas = max_replicas
        self.metrics_collector = MetricsCollector()

        # Initialize model and optimizer
        self.model = ActorCritic(obs_dim, act_dim, hidden_size, max_replicas).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.buffer = RolloutBuffer(n_steps, obs_dim, act_dim)
        
        # Persistent state for training across calls
        self.current_state = None
        self.episode_num = 0
        self.total_timesteps_so_far = 0

    def _flatten_observation(self, obs):
        """
        Flatten observation into a 1D vector.
        
        Args:
            obs: Observation from environment
            
        Returns:
            np.ndarray: Flattened observation vector
        """
        if isinstance(obs, np.ndarray):
            return obs
            
        # Sort nodes for stable order
        sorted_nodes = sorted(obs["nodes"].keys())
        
        # Create observation vector
        obs_vector = []
        for node in sorted_nodes:
            node_obs = obs["nodes"][node]
            # Add node metrics
            obs_vector.extend([
                node_obs["CPU"],
                node_obs["RAM"],
                node_obs["TX"],
                node_obs["RX"]
            ])
            
            # Add deployment metrics
            for deployment in self.deployments:
                dep_obs = node_obs["deployments"][deployment]
                obs_vector.extend([
                    dep_obs["CPU_usage"],
                    dep_obs["RAM_usage"],
                    dep_obs["TX_usage"],
                    dep_obs["RX_usage"],
                    dep_obs["Replicas"]
                ])
            
            # Add latency
            obs_vector.append(node_obs["avg_latency"])
        
        return np.array(obs_vector, dtype=np.float32)

    def select_action(self, state: Union[np.ndarray, Dict, Tuple]) -> Tuple[np.ndarray, float, float]:
        """
        Select action based on current state.
        
        Args:
            state: Current state from environment
            
        Returns:
            Tuple[np.ndarray, float, float]: (action, log_prob, value)
        """
        with torch.no_grad():
            state = self._flatten_observation(state)
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, _, value = self.model.get_action_and_value(state)
            action = action.cpu().numpy()[0].astype(np.int32)
            value = value.cpu().numpy()[0][0].item()  # Convert tensor to float
            log_prob = log_prob.cpu().numpy()[0].item()  # Convert tensor to float
            
            # Validate action values
            if not np.all((action >= 0) & (action <= self.max_replicas)):
                logger.warning(f"Invalid action values detected: {action}. Clipping to valid range [0, {self.max_replicas}]")
                action = np.clip(action, 0, self.max_replicas)
            
            return action, log_prob, value

    def collect_trajectories(self, env) -> List[Dict[str, float]]:
        """
        Collect trajectories from environment for `n_steps`.
        
        Args:
            env: Environment for interaction
            
        Returns:
            List of dictionaries with metrics for each completed episode.
        """
        self.buffer.reset()
        
        # Reset current state if it's the first run
        if self.current_state is None:
            self.current_state, _ = env.reset()

        # Track metrics for episodes completed during this rollout
        completed_episodes_info = []
        episode_reward = 0
        episode_length = 0

        for _ in range(self.n_steps):
            action, log_prob, value = self.select_action(self.current_state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            flattened_state = self._flatten_observation(self.current_state)
            self.buffer.add(flattened_state, action, reward, value, log_prob, done)
            
            self.current_state = next_state
            episode_reward += reward
            episode_length += 1

            if done:
                self.episode_num += 1
                completed_episodes_info.append({
                    "episode_reward": episode_reward,
                    "episode_length": episode_length
                })
                episode_reward = 0
                episode_length = 0
                self.current_state, _ = env.reset()

        # Compute advantages for the collected buffer
        with torch.no_grad():
            # Get the value of the last state to bootstrap returns
            state_tensor = torch.FloatTensor(self._flatten_observation(self.current_state)).unsqueeze(0).to(self.device)
            _, last_value = self.model(state_tensor)
            last_value = last_value.cpu().item()
        
        self.buffer.compute_advantages(last_value, self.gamma, self.lam)

        return completed_episodes_info

    def calculate_metrics(self, state, action, reward, value, log_prob, info, actor_loss: Optional[float] = None, critic_loss: Optional[float] = None) -> Dict[str, float]:
        """Calculate all required metrics for the current step."""
        # Calculate accuracy (1 if reward is positive, 0 otherwise)
        accuracy = 1.0 if reward > 0 else 0.0
        
        # Calculate MSE (Mean Squared Error) between reward and value
        if isinstance(value, np.ndarray):
            value = value.mean()  # Convert array to scalar if needed
        mse = (reward - value) ** 2
        
        # Calculate MRE (Mean Relative Error)
        mre = abs(reward - value) / (abs(reward) + 1e-6)
        
        # Get latency from info and take absolute value
        avg_latency = abs(info.get("latency", 0))
        
        metrics = {
            "total_reward": reward,
            "value": value,
            "log_prob": log_prob,
            "accuracy": accuracy,
            "mse": mse,
            "mre": mre,
            "avg_latency": avg_latency,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss
        }
        
        # Filter out None values
        return {k: v for k, v in metrics.items() if v is not None}

    def update(self) -> Dict[str, float]:
        """Update policy based on collected data."""
        # Prepare data for update
        states = torch.FloatTensor(self.buffer.states).to(self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        advantages = torch.FloatTensor(self.buffer.advantages).to(self.device)
        returns = torch.FloatTensor(self.buffer.returns).to(self.device)
        
        # Normalize advantages
        if self.n_steps > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        # Update over multiple epochs
        for _ in range(self.n_epochs):
            # Create batches
            indices = np.random.permutation(self.n_steps)
            for start_idx in range(0, self.n_steps, self.batch_size):
                idx = indices[start_idx:start_idx + self.batch_size]
                
                # Get new values
                logits, values = self.model(states[idx])
                # Reshape logits to [batch_size, num_deployments, num_actions]
                logits = logits.view(-1, self.act_dim, self.max_replicas + 1)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions[idx]).sum(dim=-1)
                entropy = dist.entropy().mean()
                
                # Calculate ratio of probabilities
                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                
                # PPO loss
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[idx]
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                # Note: values are squeezed to match returns shape
                critic_loss = 0.5 * (returns[idx] - values.squeeze()).pow(2).mean()
                
                # Total loss
                loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy
                
                # Update parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
        
        # Reset buffer after update
        self.buffer.reset()
        
        num_updates = self.n_epochs * (self.n_steps // self.batch_size)
        return {
            "actor_loss": total_actor_loss / num_updates,
            "critic_loss": total_critic_loss / num_updates,
            "total_loss": (total_actor_loss + total_critic_loss) / num_updates,
            "entropy": total_entropy / num_updates
        }

    def train(self, env, total_episodes: int, wandb_run_id: Optional[str] = None, training_service=None, task_id=None, loop=None, db_connection=None):
        """
        Train the PPO agent.
        
        Args:
            env: The environment to train on
            total_episodes (int): Total number of episodes to train for
            wandb_run_id (str, optional): Weights & Biases run ID
            training_service: The TrainingService instance for callbacks
            task_id: The ID of the training task
            loop: The main event loop
            db_connection: Database connection for the thread
        """
        logger.info(f"Starting PPO training for {total_episodes} episodes.")
        
        episode_rewards = []
        episode_lengths = []
        actor_losses = []
        critic_losses = []
        
        # Unified metrics tracking
        episode_latencies = []
        episode_success_counts = []
        convergence_threshold = 50  # Target reward for convergence
        steps_to_convergence = None
        convergence_achieved = False
        total_steps = 0

        # Explicitly start the workload before the training loop
        if hasattr(env, 'start_workload'):
            env.start_workload()

        for episode in range(1, total_episodes + 1):
            obs, info = env.reset()
            
            if info.get("group_completed"):
                logger.info("Experiment group finished (detected at reset). Stopping training.")
                break

            done = False
            ep_reward = 0
            ep_len = 0
            ep_latencies = []

            while not done:
                action, log_prob, value = self.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                if info.get("group_completed"):
                    logger.info("Experiment group finished. Stopping training.")
                    done = True

                self.buffer.add(self._flatten_observation(obs), action, reward, value, log_prob, done)
                obs = next_obs
                ep_reward += reward
                ep_len += 1
                total_steps += 1
                
                # Collect task-specific metrics
                if 'latency' in info:
                    ep_latencies.append(info['latency'])

                if info.get("group_completed"):
                    break

                if done:
                    break
            
            if info.get("group_completed"):
                break

            # Update the policy
            update_metrics = self.update()

            # Store episode metrics
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_len)
            actor_losses.append(update_metrics.get("actor_loss", 0))
            critic_losses.append(update_metrics.get("critic_loss", 0))
            
            # Calculate running averages
            mean_reward = np.mean(episode_rewards[-100:])  # Last 100 episodes
            mean_length = np.mean(episode_lengths[-100:])  # Last 100 episodes
            
            # Task-specific metrics
            avg_latency = np.mean(ep_latencies) if ep_latencies else 0.0
            episode_latencies.extend(ep_latencies)
            success_rate = 1.0 if ep_reward > 0 else 0.0  # Success if positive reward
            episode_success_counts.append(success_rate)
            
            # Calculate training stability
            training_stability = np.std(episode_rewards[-100:]) if len(episode_rewards) >= 10 else 0.0
            
            # Check convergence
            if not convergence_achieved and mean_reward >= convergence_threshold:
                steps_to_convergence = total_steps
                convergence_achieved = True
            
            # Log metrics to wandb (similar to SAC and TD3)
            if wandb_run_id:
                # Extract entropy coefficient as exploration rate
                exploration_rate = self.ent_coef  # PPO exploration rate
                
                # Unified logging structure
                wandb.log({
                    # Core training metrics
                    "train/episode_reward": ep_reward,
                    "train/episode_reward_avg": mean_reward,
                    "train/actor_loss": update_metrics.get("actor_loss", 0),
                    "train/critic_loss": update_metrics.get("critic_loss", 0),
                    "train/exploration_rate": exploration_rate,  # PPO exploration rate
                    "train/training_stability": training_stability,
                    
                    # Task-specific metrics
                    "task/avg_latency": avg_latency,
                    "task/success_rate": success_rate,
                    
                    # Comparison metrics
                    "comparison/steps_to_convergence": steps_to_convergence if steps_to_convergence else 0,
                    
                    # Additional metrics
                    "train/episode_length": ep_len,
                    "train/total_loss": update_metrics.get("total_loss", 0),
                    "train/mean_length": mean_length,
                    "episode": episode
                })
            
            # Print progress
            logger.info(
                f"Episode {episode}/{total_episodes}, "
                f"Reward: {ep_reward:.2f}, "
                f"Length: {ep_len}, "
                f"Latency: {avg_latency:.2f}"
            )
            
            # Update progress in the database
            if training_service and task_id and loop and db_connection:
                progress = (episode / total_episodes) * 100
                try:
                    logger.info(f"Updating progress for task {task_id}: episode {episode}, progress {progress:.1f}%")
                    future = asyncio.run_coroutine_threadsafe(
                        training_service.update_training_progress(task_id, episode, progress, db_connection), 
                        loop
                    )
                    future.result(timeout=10)  # Wait for the update to complete
                    logger.info(f"Successfully updated progress for task {task_id}")
                except Exception as e:
                    logger.error(f"Failed to update progress for task {task_id}: {str(e)}")

        logger.info(f"Training finished after {total_episodes} episodes.")
        
        # Calculate final metrics
        total_losses = [a + c for a, c in zip(actor_losses, critic_losses)]
        mean_rewards = [np.mean(episode_rewards[max(0, i-100):i+1]) for i in range(len(episode_rewards))]
        mean_lengths = [np.mean(episode_lengths[max(0, i-100):i+1]) for i in range(len(episode_lengths))]

        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "actor_losses": actor_losses,
            "critic_losses": critic_losses,
            "total_losses": total_losses,
            "mean_rewards": mean_rewards,
            "mean_lengths": mean_lengths,
            "steps_to_convergence": steps_to_convergence,
            "final_success_rate": np.mean(episode_success_counts) if episode_success_counts else 0.0
        }

    def save_model(self, path: str):
        """
        Save model parameters to a file.
        
        Args:
            path (str): Path to save
        """
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'obs_dim': self.obs_dim,
                'act_dim': self.act_dim,
                'deployments': self.deployments,
                'max_replicas': self.max_replicas
            }, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, path: str):
        """
        Load model.
        
        Args:
            path (str): Path to saved model
        """
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.obs_dim = checkpoint['obs_dim']
            self.act_dim = checkpoint['act_dim']
            self.deployments = checkpoint['deployments']
            self.max_replicas = checkpoint['max_replicas']
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise


def main():
    # Create environment
    env = LWMECPSEnv(
        num_nodes=3,
        node_name=["node1", "node2", "node3"],
        max_hardware={
            "cpu": 8,
            "ram": 16000,
            "tx_bandwidth": 1000,
            "rx_bandwidth": 1000,
            "read_disks_bandwidth": 500,
            "write_disks_bandwidth": 500,
            "avg_latency": 300,
        },
        pod_usage={
            "cpu": 2,
            "ram": 2000,
            "tx_bandwidth": 20,
            "rx_bandwidth": 20,
            "read_disks_bandwidth": 100,
            "write_disks_bandwidth": 100,
        },
        node_info={},
        deployment_name="mec-test-app",
        namespace="default",
        deployments=[
            "lwmecps-testapp-server-bs1",
            "lwmecps-testapp-server-bs2",
            "lwmecps-testapp-server-bs3",
            "lwmecps-testapp-server-bs4"
        ],
        max_pods=50,
    )

    # Calculate observation dimension
    obs_dim = 0
    for node in env.node_name:
        # Add node metrics
        obs_dim += 4  # cpu, ram, tx_bandwidth, rx_bandwidth
        # Add deployment metrics
        obs_dim += len(env.deployments) * 5  # cpu_usage, ram_usage, tx_usage, rx_usage, replicas for each deployment
        # Add latency
        obs_dim += 1

    act_dim = env.action_space.shape[0]  # 3 actions: scale down, no change, scale up

    # Calculate max_replicas based on CPU capacity
    max_replicas = int(env.max_hardware["cpu"] / env.pod_usage["cpu"])  # Maximum replicas that fit on a node

    # Create PPO agent
    ppo_agent = PPO(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_size=64,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        device="cpu",
        deployments=env.deployments,
        max_replicas=max_replicas
    )

    # Start training for 10 iterations
    print("Starting PPO training for 10 iterations...")
    ppo_agent.train(env, total_episodes=10) #, wandb_run_id="ppo_training")

    # Test trained model
    print("\nTesting trained model...")
    state, _ = env.reset()
    done = False
    cum_reward = 0.0
    while not done:
        action, log_prob, value = ppo_agent.select_action(state)
        next_state, reward, done, _, info = env.step(action)
        cum_reward += reward
        state = next_state
    
    # env.render() # render method may not exist
    print(f"Final cumulative reward: {cum_reward:.2f}")
    print(f"Episode info: {info}")


if __name__ == "__main__":
    main()
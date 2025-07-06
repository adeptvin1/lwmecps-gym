import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union
import wandb
import logging

logger = logging.getLogger(__name__)

"""
ADAPTED TD3 FOR DISCRETE ACTIONS

This is an adapted version of TD3 (Twin Delayed Deep Deterministic Policy Gradient)
for discrete action spaces. The original TD3 is designed for continuous actions,
but this implementation has been modified to work with discrete actions by:

1. Using a discrete Actor that outputs action indices via argmax
2. Removing target policy smoothing (not applicable to discrete actions)
3. Converting discrete actions to float for critic networks
4. Maintaining policy delay for stability

Note: This is not a standard TD3 implementation. For discrete actions,
consider using DQN, Dueling DQN, or PPO which are more suitable.
"""

class MetricsCollector:
    def __init__(self):
        self.metrics = {}
        self.metric_validators = {
            "accuracy": lambda x: 0 <= x <= 1,
            "mse": lambda x: x >= 0,
            "mre": lambda x: x >= 0,
            "avg_latency": lambda x: x >= 0,
            "total_reward": lambda x: True,  # Can be negative
            "q_value": lambda x: True,  # Can be negative
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

class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 256, max_replicas: int = 10):
        super().__init__()
        self.act_dim = act_dim
        self.max_replicas = max_replicas
        
        # Validate dimensions
        if act_dim <= 0:
            raise ValueError(f"act_dim must be positive, got {act_dim}")
        if max_replicas <= 0:
            raise ValueError(f"max_replicas must be positive, got {max_replicas}")
            
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim * (max_replicas + 1))
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.net(obs)
        # Reshape to [batch_size, act_dim, max_replicas + 1]
        x = x.view(-1, self.act_dim, self.max_replicas + 1)
        # Convert to discrete actions
        return torch.argmax(x, dim=-1)

class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=1))

class TD3:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
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
        max_replicas: int = 10
    ):
        self.device = torch.device(device)
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.noise_clip = noise_clip
        self.noise = noise
        self.batch_size = batch_size
        self.deployments = deployments or ["mec-test-app"]
        self.max_replicas = max_replicas
        self.metrics_collector = MetricsCollector()
        
        # Initialize networks
        self.actor = Actor(obs_dim, act_dim, hidden_size, max_replicas).to(self.device)
        self.actor_target = Actor(obs_dim, act_dim, hidden_size, max_replicas).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic1 = Critic(obs_dim, act_dim, hidden_size).to(self.device)
        self.critic2 = Critic(obs_dim, act_dim, hidden_size).to(self.device)
        self.critic1_target = Critic(obs_dim, act_dim, hidden_size).to(self.device)
        self.critic2_target = Critic(obs_dim, act_dim, hidden_size).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)
        
        # Initialize replay buffer
        self.replay_buffer = []
        self.max_buffer_size = 1000000
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []
        self.total_losses = []
        self.mean_rewards = []
        self.mean_lengths = []
        
        # TD3 specific: update counter for policy delay
        self.update_count = 0
    
    def _flatten_observation(self, obs):
        """Flatten observation into a 1D vector."""
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
    
    def select_action(self, obs):
        obs = self._flatten_observation(obs)
        state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().data.numpy().squeeze()
        
        # Add noise for exploration
        noise = np.random.normal(0, self.noise, size=action.shape)
        action = (action + noise).round()  # Round to nearest integer for discrete action
        
        # Clip to valid action range and ensure correct type
        action = np.clip(action, 0, self.max_replicas).astype(np.int32)
        
        return action
    
    def calculate_metrics(self, obs, action, reward, next_obs, info, actor_loss: Optional[float] = None, critic_loss: Optional[float] = None) -> Dict[str, float]:
        """Calculate all required metrics for the current step."""
        with torch.no_grad():
            obs = self._flatten_observation(obs)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            
            # Get Q-values from both critics
            q1 = float(self.critic1(obs_tensor, action_tensor).item())
            q2 = float(self.critic2(obs_tensor, action_tensor).item())
            q_value = min(q1, q2)
            
            # Calculate accuracy (1 if reward is positive, 0 otherwise)
            accuracy = 1.0 if float(reward) > 0 else 0.0
            
            # Calculate MSE (Mean Squared Error)
            mse = float((reward - q_value) ** 2)
            
            # Calculate MRE (Mean Relative Error)
            mre = float(abs(reward - q_value) / (abs(reward) + 1e-6))
            
            # Get metrics from info and take absolute value
            avg_latency = abs(float(info.get("latency", 0)))
            cpu_usage = float(info.get("cpu_usage", 0))
            ram_usage = float(info.get("ram_usage", 0))
            network_usage = float(info.get("network_usage", 0))
            
            metrics = {
                "accuracy": accuracy,
                "mse": mse,
                "mre": mre,
                "avg_latency": avg_latency,
                "total_reward": float(reward),
                "q_value": q_value,
                "cpu_usage": cpu_usage,
                "ram_usage": ram_usage,
                "network_usage": network_usage,
                "actor_loss": actor_loss,
                "critic_loss": critic_loss
            }
            
            # Filter out None values
            return {k: v for k, v in metrics.items() if v is not None}

    def update(self, batch_size: int = None) -> Dict[str, float]:
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.replay_buffer) < batch_size:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "total_loss": 0.0
            }
        
        # Sample from replay buffer
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        
        # Convert observations to numpy arrays first, then to tensor
        obs_batch = np.array([self._flatten_observation(self.replay_buffer[i][0]) for i in indices])
        obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        
        # Convert actions to numpy array first
        act_batch = np.array([self.replay_buffer[i][1] for i in indices])
        act_batch = torch.FloatTensor(act_batch).to(self.device)
        
        # Convert rewards and done flags to numpy arrays first
        rew_batch = np.array([float(self.replay_buffer[i][2]) for i in indices])
        rew_batch = torch.FloatTensor(rew_batch).unsqueeze(1).to(self.device)
        
        next_obs_batch = np.array([self._flatten_observation(self.replay_buffer[i][3]) for i in indices])
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)
        
        done_batch = np.array([float(self.replay_buffer[i][4]) for i in indices])
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)
        
        # Update critics
        with torch.no_grad():
            next_actions = self.actor_target(next_obs_batch)
            target_q1 = self.critic1_target(next_obs_batch, next_actions.float())
            target_q2 = self.critic2_target(next_obs_batch, next_actions.float())
            target_q = torch.min(target_q1, target_q2)
            target_q = rew_batch + (1 - done_batch) * self.gamma * target_q
        
        current_q1 = self.critic1(obs_batch, act_batch)
        current_q2 = self.critic2(obs_batch, act_batch)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Policy Delay: Update actor less frequently
        actor_loss = 0.0
        if self.update_count % self.policy_delay == 0:
            actor_actions = self.actor(obs_batch)
            actor_loss = -self.critic1(obs_batch, actor_actions.float()).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update target networks
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.update_count += 1
        
        total_loss = actor_loss + critic1_loss + critic2_loss
        
        # Convert all tensors to Python floats
        return {
            "actor_loss": float(actor_loss.detach().cpu().numpy()) if isinstance(actor_loss, torch.Tensor) else 0.0,
            "critic_loss": float((critic1_loss + critic2_loss).detach().cpu().numpy() / 2),
            "total_loss": float(total_loss.detach().cpu().numpy()) if isinstance(total_loss, torch.Tensor) else float((critic1_loss + critic2_loss).detach().cpu().numpy())
        }
    
    def train(self, env, total_episodes: int, wandb_run_id: str = None) -> Dict[str, List[float]]:
        """
        Train the TD3 agent.
        
        Args:
            env: The environment to train in
            total_episodes: Total number of episodes to train for
            wandb_run_id: Optional Weights & Biases run ID for logging
        
        Returns:
            Dictionary with lists of metrics for each episode
        """
        # Reset metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []
        self.total_losses = []
        self.mean_rewards = []
        self.mean_lengths = []
        
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_num = 0
        
        # Unified metrics tracking
        episode_latencies = []
        episode_success_counts = []
        convergence_threshold = 50  # Target reward for convergence
        steps_to_convergence = None
        convergence_achieved = False

        for t in range(1, total_episodes * env.spec.max_episode_steps + 1):
            action = self.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            self.replay_buffer.append((obs, action, reward, next_obs, done))
            if len(self.replay_buffer) > self.max_buffer_size:
                self.replay_buffer.pop(0)

            obs = next_obs
            episode_reward += reward
            episode_length += 1
            
            # Collect task-specific metrics
            if 'latency' in info:
                episode_latencies.append(info['latency'])

            # Update networks
            update_metrics = self.update()
            
            if done:
                episode_num += 1
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Update metrics lists
                self.actor_losses.append(update_metrics["actor_loss"])
                self.critic_losses.append(update_metrics["critic_loss"])
                self.total_losses.append(update_metrics["total_loss"])
                self.mean_rewards.append(np.mean(self.episode_rewards[-100:]))
                self.mean_lengths.append(np.mean(self.episode_lengths[-100:]))

                # Task-specific metrics
                avg_latency = np.mean(episode_latencies) if episode_latencies else 0.0
                success_rate = 1.0 if episode_reward > 0 else 0.0  # Success if positive reward
                episode_success_counts.append(success_rate)
                
                # Calculate training stability
                training_stability = np.std(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 10 else 0.0
                
                # Check convergence
                if not convergence_achieved and self.mean_rewards[-1] >= convergence_threshold:
                    steps_to_convergence = t
                    convergence_achieved = True

                print(
                    f"Episode: {episode_num}/{total_episodes}, "
                    f"Reward: {episode_reward:.2f}, "
                    f"Length: {episode_length}, "
                    f"Actor Loss: {update_metrics['actor_loss']:.3f}, "
                    f"Critic Loss: {update_metrics['critic_loss']:.3f}, "
                    f"Noise: {self.noise:.3f}, "
                    f"Latency: {avg_latency:.2f}"
                )

                if wandb_run_id:
                    # Unified logging structure
                    wandb.log({
                        # Core training metrics
                        "train/episode_reward": episode_reward,
                        "train/episode_reward_avg": self.mean_rewards[-1],
                        "train/actor_loss": update_metrics["actor_loss"],
                        "train/critic_loss": update_metrics["critic_loss"],
                        "train/exploration_rate": self.noise,  # TD3 exploration rate
                        "train/training_stability": training_stability,
                        
                        # Task-specific metrics
                        "task/avg_latency": avg_latency,
                        "task/success_rate": success_rate,
                        
                        # Comparison metrics
                        "comparison/steps_to_convergence": steps_to_convergence if steps_to_convergence else 0,
                        
                        # Additional metrics
                        "train/episode_length": episode_length,
                        "train/total_loss": update_metrics["total_loss"],
                        "train/mean_length": self.mean_lengths[-1],
                        "episode": episode_num
                    })
                
                # Reset for next episode
                obs, _ = env.reset()
                episode_reward = 0
                episode_length = 0
                episode_latencies = []
                
                if episode_num >= total_episodes:
                    break

        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "actor_losses": self.actor_losses,
            "critic_losses": self.critic_losses,
            "total_losses": self.total_losses,
            "mean_rewards": self.mean_rewards,
            "mean_lengths": self.mean_lengths,
            "steps_to_convergence": steps_to_convergence,
            "final_success_rate": np.mean(episode_success_counts) if episode_success_counts else 0.0
        }
    
    def save_model(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict()
        }, path)
    
    def load_model(self, path: str):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic1_target.load_state_dict(checkpoint["critic1_target"])
        self.critic2_target.load_state_dict(checkpoint["critic2_target"]) 
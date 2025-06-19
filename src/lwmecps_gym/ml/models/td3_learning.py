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
        # Actions are discrete indices, ensure they're within valid range
        action = np.clip(action, 0, self.max_replicas)
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
            
            # Get metrics from info
            avg_latency = float(info.get("latency", 0))
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
                "network_usage": network_usage
            }
            
            if actor_loss is not None:
                metrics["actor_loss"] = float(actor_loss)
            if critic_loss is not None:
                metrics["critic_loss"] = float(critic_loss)
                
            return metrics

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
    
    def train(self, env, total_timesteps: int, wandb_run_id: str = None) -> Dict[str, List[float]]:
        """
        Train the TD3 agent.
        
        Args:
            env: The environment to train in
            total_timesteps: Total number of timesteps to train for
            wandb_run_id: Optional Weights & Biases run ID for logging
            
        Returns:
            Dictionary containing training metrics
        """
        # Initialize metrics
        metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "actor_losses": [],
            "critic_losses": [],
            "accuracies": [],
            "mses": [],
            "mres": [],
            "avg_latencies": [],
            "cpu_usages": [],
            "ram_usages": [],
            "network_usages": []
        }
        
        # Initialize episode counter
        episode_count = 0
        episode_reward = 0.0
        episode_length = 0
        episode_actor_loss = 0.0
        episode_critic_loss = 0.0
        episode_accuracy = 0.0
        episode_mse = 0.0
        episode_mre = 0.0
        episode_avg_latency = 0.0
        episode_cpu_usage = 0.0
        episode_ram_usage = 0.0
        episode_network_usage = 0.0
        
        # Pre-fill replay buffer with random actions
        print("Pre-filling replay buffer...")
        state, _ = env.reset()
        for _ in range(self.batch_size):
            action = env.action_space.sample()
            next_state, reward, done, _, info = env.step(action)
            self.replay_buffer.append((state, action, float(reward), next_state, done))
            state = next_state
            if done:
                state, _ = env.reset()
        print("Replay buffer pre-filled")
        
        # Training loop
        state, _ = env.reset()
        episode_count = 0
        
        while True:  # Changed to while True since we'll break explicitly
            if episode_count >= total_timesteps:
                print(f"Reached episode limit of {total_timesteps}. Stopping training.")
                break
                
            # Reset episode metrics
            episode_reward = 0.0
            episode_length = 0
            episode_actor_loss = 0.0
            episode_critic_loss = 0.0
            episode_accuracy = 0.0
            episode_mse = 0.0
            episode_mre = 0.0
            episode_avg_latency = 0.0
            episode_cpu_usage = 0.0
            episode_ram_usage = 0.0
            episode_network_usage = 0.0
            
            # Episode loop
            done = False
            truncated = False
            max_steps_per_episode = 10  # Added max steps per episode
            
            while not (done or truncated) and episode_length < max_steps_per_episode:  # Added episode length check
                # Select action
                action = self.select_action(state)
                
                # Execute action
                next_state, reward, done, truncated, info = env.step(action)
                
                # Store transition in replay buffer
                self.replay_buffer.append((state, action, float(reward), next_state, done))
                if len(self.replay_buffer) > self.max_buffer_size:
                    self.replay_buffer.pop(0)
                
                # Update state
                state = next_state
                
                # Update episode metrics
                episode_reward += float(reward)
                episode_length += 1
                
                # Check if we've reached the maximum steps
                if episode_length >= max_steps_per_episode:
                    print(f"Episode {episode_count} reached maximum steps ({max_steps_per_episode})")
                    break
                
                # Update resource usage metrics
                if 'avg_latency' in info:
                    episode_avg_latency += float(info['avg_latency'])
                if 'cpu_usage' in info:
                    episode_cpu_usage += float(info['cpu_usage'])
                if 'ram_usage' in info:
                    episode_ram_usage += float(info['ram_usage'])
                if 'network_usage' in info:
                    episode_network_usage += float(info['network_usage'])
                
                # Train agent if enough samples
                if len(self.replay_buffer) > self.batch_size:
                    update_metrics = self.update()
                    episode_actor_loss += float(update_metrics['actor_loss'])
                    episode_critic_loss += float(update_metrics['critic_loss'])
                    
                    step_metrics = self.calculate_metrics(state, action, reward, next_state, info)
                    episode_accuracy += float(step_metrics['accuracy'])
                    episode_mse += float(step_metrics['mse'])
                    episode_mre += float(step_metrics['mre'])
                
                # Log metrics to wandb if run_id is provided
                if wandb_run_id:
                    wandb.log({
                        "episode": episode_count,
                        "step": episode_length,
                        "reward": float(reward),
                        "actor_loss": float(episode_actor_loss) / (episode_length + 1e-8),
                        "critic_loss": float(episode_critic_loss) / (episode_length + 1e-8),
                        "accuracy": float(episode_accuracy) / (episode_length + 1e-8),
                        "mse": float(episode_mse) / (episode_length + 1e-8),
                        "mre": float(episode_mre) / (episode_length + 1e-8),
                        "avg_latency": float(episode_avg_latency) / (episode_length + 1e-8),
                        "cpu_usage": float(episode_cpu_usage) / (episode_length + 1e-8),
                        "ram_usage": float(episode_ram_usage) / (episode_length + 1e-8),
                        "network_usage": float(episode_network_usage) / (episode_length + 1e-8)
                    })
            
            # Increment episode counter
            episode_count += 1
            
            # Store episode metrics
            metrics["episode_rewards"].append(float(episode_reward))
            metrics["episode_lengths"].append(episode_length)
            metrics["actor_losses"].append(float(episode_actor_loss) / (episode_length + 1e-8))
            metrics["critic_losses"].append(float(episode_critic_loss) / (episode_length + 1e-8))
            metrics["accuracies"].append(float(episode_accuracy) / (episode_length + 1e-8))
            metrics["mses"].append(float(episode_mse) / (episode_length + 1e-8))
            metrics["mres"].append(float(episode_mre) / (episode_length + 1e-8))
            metrics["avg_latencies"].append(float(episode_avg_latency) / (episode_length + 1e-8))
            metrics["cpu_usages"].append(float(episode_cpu_usage) / (episode_length + 1e-8))
            metrics["ram_usages"].append(float(episode_ram_usage) / (episode_length + 1e-8))
            metrics["network_usages"].append(float(episode_network_usage) / (episode_length + 1e-8))
            
            # Print episode summary
            print(f"Episode: {episode_count}/{total_timesteps}, "
                  f"Episode Reward: {float(episode_reward):.2f}, "
                  f"Episode Length: {episode_length}, "
                  f"Actor Loss: {float(episode_actor_loss)/(episode_length + 1e-8):.3f}, "
                  f"Critic Loss: {float(episode_critic_loss)/(episode_length + 1e-8):.3f}, "
                  f"Accuracy: {float(episode_accuracy)/(episode_length + 1e-8):.3f}, "
                  f"MSE: {float(episode_mse)/(episode_length + 1e-8):.3f}, "
                  f"MRE: {float(episode_mre)/(episode_length + 1e-8):.3f}, "
                  f"Avg Latency: {float(episode_avg_latency)/(episode_length + 1e-8):.2f}, "
                  f"CPU Usage: {float(episode_cpu_usage)/(episode_length + 1e-8):.2f}, "
                  f"RAM Usage: {float(episode_ram_usage)/(episode_length + 1e-8):.2f}, "
                  f"Network Usage: {float(episode_network_usage)/(episode_length + 1e-8):.2f}")
            
            # Reset environment for next episode
            state, _ = env.reset()
        
        return metrics
    
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
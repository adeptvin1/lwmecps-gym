import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union
import wandb
import logging

logger = logging.getLogger(__name__)

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
        state = torch.FloatTensor(obs).to(self.device)
        action = self.actor(state).cpu().data.numpy().squeeze()
        # Масштабируем действия из [-1, 1] в [0, 4] и обрезаем до допустимого диапазона
        action = np.clip(2 * action + 2, 0, 4)
        return action
    
    def calculate_metrics(self, obs, action, reward, next_obs, info, actor_loss: Optional[float] = None, critic_loss: Optional[float] = None) -> Dict[str, float]:
        """Calculate all required metrics for the current step."""
        with torch.no_grad():
            obs = self._flatten_observation(obs)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            
            # Get Q-values from both critics
            q1 = self.critic1(obs_tensor, action_tensor).item()
            q2 = self.critic2(obs_tensor, action_tensor).item()
            q_value = min(q1, q2)
            
            # Calculate accuracy (1 if reward is positive, 0 otherwise)
            accuracy = 1.0 if reward > 0 else 0.0
            
            # Calculate MSE (Mean Squared Error)
            mse = (reward - q_value) ** 2
            
            # Calculate MRE (Mean Relative Error)
            mre = abs(reward - q_value) / (abs(reward) + 1e-6)
            
            # Get metrics from info
            avg_latency = info.get("latency", 0)
            cpu_usage = info.get("cpu_usage", 0)
            ram_usage = info.get("ram_usage", 0)
            network_usage = info.get("network_usage", 0)
            
            metrics = {
                "accuracy": accuracy,
                "mse": mse,
                "mre": mre,
                "avg_latency": avg_latency,
                "total_reward": reward,
                "q_value": q_value,
                "cpu_usage": cpu_usage,
                "ram_usage": ram_usage,
                "network_usage": network_usage
            }
            
            if actor_loss is not None:
                metrics["actor_loss"] = actor_loss
            if critic_loss is not None:
                metrics["critic_loss"] = critic_loss
                
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
        obs_batch = torch.FloatTensor([self._flatten_observation(self.replay_buffer[i][0]) for i in indices]).to(self.device)
        act_batch = torch.FloatTensor([self.replay_buffer[i][1] for i in indices]).to(self.device)
        rew_batch = torch.FloatTensor([self.replay_buffer[i][2] for i in indices]).to(self.device)
        next_obs_batch = torch.FloatTensor([self._flatten_observation(self.replay_buffer[i][3]) for i in indices]).to(self.device)
        done_batch = torch.FloatTensor([self.replay_buffer[i][4] for i in indices]).to(self.device)
        
        # Update critics
        with torch.no_grad():
            # Add noise to target actions
            noise = torch.randint(-1, 2, size=(batch_size, self.act_dim), device=self.device)
            next_actions = self.actor_target(next_obs_batch)
            next_actions = torch.clamp(next_actions + noise, 0, self.max_replicas)
            
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
        
        actor_loss = 0.0
        if len(self.replay_buffer) % self.policy_delay == 0:
            # Update actor
            actor_loss = -self.critic1(obs_batch, self.actor(obs_batch).float()).mean()
            
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
        
        total_loss = actor_loss + critic1_loss + critic2_loss
        
        # Calculate and collect metrics for each sample in the batch
        for i in range(batch_size):
            step_metrics = self.calculate_metrics(
                self.replay_buffer[indices[i]][0],
                act_batch[i].cpu().numpy(),
                rew_batch[i].item(),
                self.replay_buffer[indices[i]][3],
                {"latency": rew_batch[i].item()},
                actor_loss.item() if actor_loss != 0.0 else None,
                (critic1_loss.item() + critic2_loss.item()) / 2
            )
            self.metrics_collector.update(step_metrics)
        
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": (critic1_loss.item() + critic2_loss.item()) / 2,
            "total_loss": total_loss.item()
        }
    
    def train(self, env, total_timesteps: int, wandb_run_id: Optional[str] = None) -> Dict[str, List[float]]:
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_metrics = {}
        
        # Initialize wandb if run_id is provided
        if wandb_run_id:
            wandb.init(
                id=wandb_run_id,
                project="lwmecps-gym",
                config={
                    "algorithm": "TD3",
                    "total_timesteps": total_timesteps,
                    "hidden_size": self.actor.net[0].out_features,
                    "learning_rate": self.actor_optimizer.param_groups[0]['lr'],
                    "gamma": self.gamma,
                    "tau": self.tau,
                    "policy_delay": self.policy_delay,
                    "noise_clip": self.noise_clip,
                    "noise": self.noise,
                    "batch_size": self.batch_size,
                    "max_replicas": self.max_replicas
                }
            )
        
        # Pre-fill replay buffer with random actions
        print("Pre-filling replay buffer...")
        while len(self.replay_buffer) < self.batch_size:
            # Генерируем действия в диапазоне [0, 4]
            action = np.random.randint(0, 5, size=self.act_dim)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            self.replay_buffer.append((obs, action, reward, next_obs, done))
            obs = next_obs
            if done:
                obs, _ = env.reset()
        print(f"Replay buffer filled with {len(self.replay_buffer)} samples")
        
        obs, _ = env.reset()  # Reset environment after pre-filling
        episode_reward = 0
        episode_length = 0
        
        for t in range(total_timesteps):
            action = self.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition in replay buffer
            self.replay_buffer.append((obs, action, reward, next_obs, done))
            if len(self.replay_buffer) > self.max_buffer_size:
                self.replay_buffer.pop(0)
            
            # Update networks
            update_metrics = self.update()
            
            # Get average metrics
            avg_metrics = self.metrics_collector.get_average_metrics()
            
            # Combine all metrics
            all_metrics = {**avg_metrics, **update_metrics}
            
            # Store metrics
            for key, value in all_metrics.items():
                if key not in episode_metrics:
                    episode_metrics[key] = []
                episode_metrics[key].append(value)
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            
            # Log to wandb on every step
            if wandb_run_id:
                wandb.log({
                    "timesteps": t,
                    "episode_reward": episode_reward,
                    "episode_length": episode_length,
                    "metrics/accuracy": avg_metrics.get('accuracy', 0),
                    "metrics/mse": avg_metrics.get('mse', 0),
                    "metrics/mre": avg_metrics.get('mre', 0),
                    "metrics/avg_latency": avg_metrics.get('avg_latency', 0),
                    "metrics/q_value": avg_metrics.get('q_value', 0),
                    "metrics/cpu_usage": avg_metrics.get('cpu_usage', 0),
                    "metrics/ram_usage": avg_metrics.get('ram_usage', 0),
                    "metrics/network_usage": avg_metrics.get('network_usage', 0),
                    "losses/actor_loss": update_metrics['actor_loss'],
                    "losses/critic_loss": update_metrics['critic_loss'],
                    "losses/total_loss": update_metrics['total_loss']
                })
            
            # Print episode summary
            print(
                f"Timesteps: {t}/{total_timesteps}, "
                f"Episode Reward: {episode_reward:.2f}, "
                f"Episode Length: {episode_length}, "
                f"Actor Loss: {update_metrics['actor_loss']:.3f}, "
                f"Critic Loss: {update_metrics['critic_loss']:.3f}, "
                f"Accuracy: {avg_metrics.get('accuracy', 0):.3f}, "
                f"MSE: {avg_metrics.get('mse', 0):.3f}, "
                f"MRE: {avg_metrics.get('mre', 0):.3f}, "
                f"Avg Latency: {avg_metrics.get('avg_latency', 0):.2f}, "
                f"CPU Usage: {avg_metrics.get('cpu_usage', 0):.2f}, "
                f"RAM Usage: {avg_metrics.get('ram_usage', 0):.2f}, "
                f"Network Usage: {avg_metrics.get('network_usage', 0):.2f}"
            )
            
            if done:
                # Reset episode
                obs, _ = env.reset()
                episode_reward = 0
                episode_length = 0
                self.metrics_collector.reset()
        
        # Close wandb run
        if wandb_run_id:
            wandb.finish()
        
        return {
            "episode_rewards": episode_metrics.get("total_reward", []),
            "episode_lengths": episode_metrics.get("steps", []),
            "actor_losses": episode_metrics.get("actor_loss", []),
            "critic_losses": episode_metrics.get("critic_loss", []),
            "total_losses": episode_metrics.get("total_loss", []),
            "mean_rewards": episode_metrics.get("mean_reward", []),
            "mean_lengths": episode_metrics.get("mean_length", [])
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
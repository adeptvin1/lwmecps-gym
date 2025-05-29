import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import wandb

class MetricsCollector:
    def __init__(self):
        self.metrics = {}
    
    def update(self, metrics_dict: Dict[str, float]):
        """Update metrics with new values."""
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = []
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
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_size, act_dim)
        self.log_std = nn.Linear(hidden_size, act_dim)
        self.act_dim = act_dim
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(obs)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mu, log_std
    
    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, mu

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

class SAC:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_size: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_entropy: bool = True,
        target_entropy: float = -1.0,
        batch_size: int = 256,
        device: str = "cpu",
        deployments: List[str] = None
    ):
        self.device = torch.device(device)
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_entropy = auto_entropy
        self.target_entropy = target_entropy
        self.batch_size = batch_size
        self.deployments = deployments or ["mec-test-app"]
        self.metrics_collector = MetricsCollector()
        
        # Initialize networks
        self.actor = Actor(obs_dim, act_dim, hidden_size).to(self.device)
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
        
        # Initialize entropy tuning
        if self.auto_entropy:
            self.target_entropy = -np.prod((act_dim,)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        
        # Initialize replay buffer
        self.replay_buffer = []
        self.max_buffer_size = 1000000
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []
        self.alpha_losses = []
        self.total_losses = []
        self.mean_rewards = []
        self.mean_lengths = []
    
    def select_action(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, _, _ = self.actor.sample(obs_tensor)
            return action.cpu().numpy()[0]
    
    def calculate_metrics(self, obs, action, reward, next_obs, info, actor_loss: Optional[float] = None, critic_loss: Optional[float] = None, alpha_loss: Optional[float] = None) -> Dict[str, float]:
        """Calculate all required metrics for the current step."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            
            # Get Q-values from both critics
            q1 = self.critic1(obs_tensor, action_tensor).item()
            q2 = self.critic2(obs_tensor, action_tensor).item()
            q_value = min(q1, q2)
            
            # Get action log probability
            _, log_prob, _ = self.actor.sample(obs_tensor)
            log_prob = log_prob.item()
            
            # Calculate accuracy (1 if reward is positive, 0 otherwise)
            accuracy = 1.0 if reward > 0 else 0.0
            
            # Calculate MSE (Mean Squared Error)
            mse = (reward - q_value) ** 2
            
            # Calculate MRE (Mean Relative Error)
            mre = abs(reward - q_value) / (abs(reward) + 1e-6)
            
            # Get latency from info
            avg_latency = info.get("latency", 0)
            
            metrics = {
                "accuracy": accuracy,
                "mse": mse,
                "mre": mre,
                "avg_latency": avg_latency,
                "total_reward": reward,
                "q_value": q_value,
                "log_prob": log_prob,
                "alpha": self.alpha
            }
            
            if actor_loss is not None:
                metrics["actor_loss"] = actor_loss
            if critic_loss is not None:
                metrics["critic_loss"] = critic_loss
            if alpha_loss is not None:
                metrics["alpha_loss"] = alpha_loss
                
            return metrics

    def update(self, batch_size: int = None) -> Dict[str, float]:
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.replay_buffer) < batch_size:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "alpha_loss": 0.0,
                "total_loss": 0.0
            }
        
        # Sample from replay buffer
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        obs_batch = torch.FloatTensor([self.replay_buffer[i][0] for i in indices]).to(self.device)
        act_batch = torch.FloatTensor([self.replay_buffer[i][1] for i in indices]).to(self.device)
        rew_batch = torch.FloatTensor([self.replay_buffer[i][2] for i in indices]).to(self.device)
        next_obs_batch = torch.FloatTensor([self.replay_buffer[i][3] for i in indices]).to(self.device)
        done_batch = torch.FloatTensor([self.replay_buffer[i][4] for i in indices]).to(self.device)
        
        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_obs_batch)
            target_q1 = self.critic1_target(next_obs_batch, next_actions)
            target_q2 = self.critic2_target(next_obs_batch, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rew_batch + (1 - done_batch) * self.gamma * (target_q - self.alpha * next_log_probs)
        
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
        
        # Update actor
        actions, log_probs, _ = self.actor.sample(obs_batch)
        q1 = self.critic1(obs_batch, actions)
        q2 = self.critic2(obs_batch, actions)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_probs - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha
        alpha_loss = 0.0
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        
        # Update target networks
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        total_loss = actor_loss + critic1_loss + critic2_loss + alpha_loss
        
        # Calculate and collect metrics for each sample in the batch
        for i in range(batch_size):
            step_metrics = self.calculate_metrics(
                obs_batch[i].cpu().numpy(),
                act_batch[i].cpu().numpy(),
                rew_batch[i].item(),
                next_obs_batch[i].cpu().numpy(),
                {"latency": rew_batch[i].item()},
                actor_loss.item(),
                (critic1_loss.item() + critic2_loss.item()) / 2,
                alpha_loss.item() if self.auto_entropy else 0.0
            )
            self.metrics_collector.update(step_metrics)
        
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": (critic1_loss.item() + critic2_loss.item()) / 2,
            "alpha_loss": alpha_loss.item() if self.auto_entropy else 0.0,
            "total_loss": total_loss.item()
        }
    
    def train(self, env, total_timesteps: int, wandb_run_id: Optional[str] = None) -> Dict[str, List[float]]:
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_metrics = {}
        
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
            
            if done:
                # Log to wandb
                if wandb_run_id:
                    wandb.log({
                        "timesteps": t,
                        "episode_reward": episode_reward,
                        "episode_length": episode_length,
                        **all_metrics
                    })
                
                # Print episode summary
                print(
                    f"Timesteps: {t}/{total_timesteps}, "
                    f"Episode Reward: {episode_reward:.2f}, "
                    f"Episode Length: {episode_length}, "
                    f"Actor Loss: {update_metrics['actor_loss']:.3f}, "
                    f"Critic Loss: {update_metrics['critic_loss']:.3f}, "
                    f"Alpha Loss: {update_metrics['alpha_loss']:.3f}, "
                    f"Accuracy: {avg_metrics.get('accuracy', 0):.3f}, "
                    f"MSE: {avg_metrics.get('mse', 0):.3f}, "
                    f"MRE: {avg_metrics.get('mre', 0):.3f}, "
                    f"Avg Latency: {avg_metrics.get('avg_latency', 0):.2f}"
                )
                
                # Reset episode
                obs, _ = env.reset()
                episode_reward = 0
                episode_length = 0
                self.metrics_collector.reset()
        
        return episode_metrics
    
    def save_model(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "log_alpha": self.log_alpha if self.auto_entropy else None
        }, path)
    
    def load_model(self, path: str):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.critic1_target.load_state_dict(checkpoint["critic1_target"])
        self.critic2_target.load_state_dict(checkpoint["critic2_target"])
        if self.auto_entropy and checkpoint["log_alpha"] is not None:
            self.log_alpha = checkpoint["log_alpha"] 
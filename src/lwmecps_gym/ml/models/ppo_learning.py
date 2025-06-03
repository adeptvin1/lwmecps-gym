import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gymnasium import spaces
import logging
from typing import List, Dict, Union, Tuple, Optional
import wandb

from lwmecps_gym.envs import LWMECPSEnv

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

    def compute_advantages(self, gamma, lam):
        """
        Compute advantages using GAE.
        
        Args:
            gamma (float): Discount factor
            lam (float): Lambda parameter for GAE
        """
        gae = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_value = 0
            else:
                next_value = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            self.advantages[t] = gae
            self.returns[t] = self.advantages[t] + self.values[t]


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
            
            # Validate action values
            if not np.all((action >= 0) & (action <= self.max_replicas)):
                logger.warning(f"Invalid action values detected: {action}. Clipping to valid range [0, {self.max_replicas}]")
                action = np.clip(action, 0, self.max_replicas)
            
            return action, log_prob.cpu().numpy()[0], value.cpu().numpy()[0][0]

    def collect_trajectories(self, env) -> Tuple[float, int]:
        """
        Collect trajectories from environment.
        
        Args:
            env: Environment for interaction
            
        Returns:
            Tuple[float, int]: (episode reward, episode length)
        """
        self.buffer.reset()
        state, _ = env.reset()  # Get initial state and info
        done = False
        episode_reward = 0
        episode_length = 0

        try:
            while not self.buffer.is_full():
                action, log_prob, value = self.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                # Flatten state before adding to buffer
                flattened_state = self._flatten_observation(state)
                self.buffer.add(flattened_state, action, reward, value, log_prob, done)
                state = next_state
                episode_reward += reward
                episode_length += 1

                if done:
                    state, _ = env.reset()  # Get new initial state and info
                    episode_reward = 0
                    episode_length = 0

            self.buffer.compute_advantages(self.gamma, self.lam)
            return episode_reward, episode_length
        except Exception as e:
            logger.error(f"Error in collect_trajectories: {str(e)}")
            raise

    def calculate_metrics(self, state, action, reward, value, log_prob, info, actor_loss: Optional[float] = None, critic_loss: Optional[float] = None) -> Dict[str, float]:
        """Calculate all required metrics for the current step."""
        # Calculate accuracy (1 if reward is positive, 0 otherwise)
        accuracy = 1.0 if reward > 0 else 0.0
        
        # Calculate MSE (Mean Squared Error) between reward and value
        mse = (reward - value) ** 2
        
        # Calculate MRE (Mean Relative Error)
        mre = abs(reward - value) / (abs(reward) + 1e-6)
        
        # Get latency from info
        avg_latency = info.get("latency", 0)
        
        metrics = {
            "accuracy": accuracy,
            "mse": mse,
            "mre": mre,
            "avg_latency": avg_latency,
            "total_reward": reward,
            "value": value,
            "log_prob": log_prob
        }
        
        if actor_loss is not None:
            metrics["actor_loss"] = actor_loss
        if critic_loss is not None:
            metrics["critic_loss"] = critic_loss
            
        return metrics

    def update(self) -> Dict[str, float]:
        """Update policy based on collected data."""
        # Compute advantages
        self.buffer.compute_advantages(self.gamma, self.lam)
        
        # Prepare data for update
        states = torch.FloatTensor(self.buffer.states).to(self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        old_values = torch.FloatTensor(self.buffer.values).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        advantages = torch.FloatTensor(self.buffer.advantages).to(self.device)
        returns = torch.FloatTensor(self.buffer.returns).to(self.device)
        
        # Normalize advantages
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
                critic_loss = 0.5 * (returns[idx] - values.squeeze()).pow(2).mean()
                
                # Total loss
                loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy
                
                # Update parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Collect metrics
                step_metrics = self.calculate_metrics(
                    states[idx].cpu().numpy(),
                    actions[idx].cpu().numpy(),
                    self.buffer.rewards[idx],
                    values.squeeze().detach().cpu().numpy(),
                    new_log_probs.detach().cpu().numpy(),
                    {"latency": self.buffer.rewards[idx].mean()},
                    actor_loss.item(),
                    critic_loss.item()
                )
                self.metrics_collector.update(step_metrics)
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
        
        # Reset buffer
        self.buffer.reset()
        
        # Return average loss values
        return {
            "actor_loss": total_actor_loss / (self.n_epochs * (self.n_steps // self.batch_size)),
            "critic_loss": total_critic_loss / (self.n_epochs * (self.n_steps // self.batch_size)),
            "entropy": total_entropy / (self.n_epochs * (self.n_steps // self.batch_size))
        }

    def train(self, env, total_timesteps: int, wandb_run_id: Optional[str] = None) -> Dict[str, List[float]]:
        """Train agent."""
        timesteps_so_far = 0
        episode_metrics = {}
        
        while timesteps_so_far < total_timesteps:
            # Collect trajectories
            episode_reward, episode_length = self.collect_trajectories(env)
            timesteps_so_far += episode_length
            
            # Update policy
            update_metrics = self.update()
            
            # Get average metrics
            avg_metrics = self.metrics_collector.get_average_metrics()
            
            # Combine all metrics
            all_metrics = {**avg_metrics, **update_metrics}
            
            # Save metrics
            for key, value in all_metrics.items():
                if key not in episode_metrics:
                    episode_metrics[key] = []
                episode_metrics[key].append(value)
            
            # Log to wandb
            if wandb_run_id:
                wandb.log({
                    "timesteps": timesteps_so_far,
                    "episode_reward": episode_reward,
                    "episode_length": episode_length,
                    **all_metrics
                })
            
            # Output information
            logger.info(
                f"Timesteps: {timesteps_so_far}/{total_timesteps}, "
                f"Episode Reward: {episode_reward:.2f}, "
                f"Episode Length: {episode_length}, "
                f"Actor Loss: {update_metrics['actor_loss']:.3f}, "
                f"Critic Loss: {update_metrics['critic_loss']:.3f}, "
                f"Entropy: {update_metrics['entropy']:.3f}, "
                f"Accuracy: {avg_metrics.get('accuracy', 0):.3f}, "
                f"MSE: {avg_metrics.get('mse', 0):.3f}, "
                f"MRE: {avg_metrics.get('mre', 0):.3f}, "
                f"Avg Latency: {avg_metrics.get('avg_latency', 0):.2f}"
            )
        
        return episode_metrics

    def save_model(self, path: str):
        """
        Save model.
        
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
    ppo_agent.train(env, total_timesteps=20480, wandb_run_id="ppo_training")  # 2048 * 10 = 20480 timesteps

    # Test trained model
    print("\nTesting trained model...")
    state = env.reset()
    done = False
    cum_reward = 0.0
    while not done:
        action, _, _ = ppo_agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        cum_reward += reward
        state = next_state

    env.render()
    print(f"Final cumulative reward: {cum_reward:.2f}")
    print(f"Episode info: {info}")


if __name__ == "__main__":
    main()
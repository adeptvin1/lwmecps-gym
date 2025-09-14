import json
import os
import random
import re
from collections import deque, defaultdict
from time import time
from typing import Dict, List, Optional
import logging

import bitmath
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from gymnasium.envs.registration import register
from lwmecps_gym.envs.kubernetes_api import k8s

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hyperparameters
gamma = 0.99
learning_rate = 0.001
batch_size = 64
replay_buffer_size = 10000
target_update_freq = 5
num_episodes = 100
max_steps_per_episode = 100
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MetricsCollector:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.metric_validators = {
            "accuracy": lambda x: 0 <= x <= 1,
            "mse": lambda x: x >= 0,
            "mre": lambda x: x >= 0,
            "avg_latency": lambda x: x >= 0,
            "total_reward": lambda x: True,  # Can be negative
            "loss": lambda x: x >= 0
        }
    
    def update(self, metrics_dict: Dict[str, float]):
        """Update metrics with new values."""
        for key, value in metrics_dict.items():
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


# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, max_replicas=10):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_replicas = max_replicas
        
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        
        # For MultiDiscrete action space, output Q-values for each possible action
        # We'll use a single output for simplicity (same action for all deployments)
        self.fc3 = nn.Linear(64, max_replicas + 1)  # 0 to max_replicas actions

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save_model(self, file_name="./dqn_model.pth"):
        torch.save(self.state_dict(), file_name)
        print(f"Model saved to {file_name}")

    def load_model(self, file_name="./dqn_model.pth"):
        if os.path.exists(file_name):
            self.load_state_dict(torch.load(file_name))
            print(f"Model loaded from {file_name}")
        else:
            print(f"No model found at {file_name}")


# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(done),
        )

    def size(self):
        return len(self.buffer)


device = "cpu"


# DQN Agent
class DQNAgent:
    def __init__(self, env, replay_buffer=None, learning_rate=0.001, discount_factor=0.99, 
                 epsilon=0.1, memory_size=10000, batch_size=32):
        self.env = env
        self.replay_buffer = replay_buffer or ReplayBuffer(memory_size)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.batch_size = batch_size
        
        # Calculate observation dimension for Dict space
        obs_dim = self._calculate_obs_dim()
        
        # Calculate action dimension and max_replicas
        if hasattr(env.action_space, 'n'):
            action_dim = env.action_space.n
            max_replicas = 10  # Default
        elif hasattr(env.action_space, 'shape'):
            action_dim = env.action_space.shape[0]
            # For MultiDiscrete, get max action value from nvec
            if hasattr(env.action_space, 'nvec'):
                max_replicas = max(env.action_space.nvec) - 1  # Convert to 0-based indexing
            else:
                max_replicas = 10  # Default fallback
        else:
            action_dim = 4  # Default fallback
            max_replicas = 10
            
        self.max_replicas = max_replicas
        self.model = DQN(obs_dim, action_dim, max_replicas).to(device)
        self.target_model = DQN(obs_dim, action_dim, max_replicas).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.update_target_network()
        self.metrics_collector = MetricsCollector()
    
    def _calculate_obs_dim(self):
        """Calculate observation dimension from Dict space."""
        obs_dim = 0
        if hasattr(self.env, 'node_name') and hasattr(self.env, 'deployments'):
            for node in self.env.node_name:
                # Add node metrics: CPU, RAM, TX, RX
                obs_dim += 4
                # Add deployment metrics for each deployment
                for deployment in self.env.deployments:
                    obs_dim += 5  # CPU_usage, RAM_usage, TX_usage, RX_usage, Replicas
                # Add latency
                obs_dim += 1
        else:
            # Fallback: try to get from observation space
            try:
                if hasattr(self.env.observation_space, 'shape'):
                    obs_dim = self.env.observation_space.shape[0]
                else:
                    obs_dim = 100  # Default fallback
            except:
                obs_dim = 100
        return obs_dim
    
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
            if hasattr(self.env, 'deployments'):
                for deployment in self.env.deployments:
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

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = self._flatten_observation(state)
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.model(state)
            
            # Handle MultiDiscrete action space
            if hasattr(self.env.action_space, 'shape') and self.env.action_space.shape[0] > 1:
                # For MultiDiscrete, we need to select actions for each deployment
                num_deployments = self.env.action_space.shape[0]
                max_replicas = self.env.action_space.nvec[0] - 1  # Get max replicas from action space
                
                actions = np.zeros(num_deployments, dtype=np.int32)
                
                # Choose action for each deployment
                for i in range(num_deployments):
                    # For simplicity, use the same Q-values for all deployments
                    # In a more sophisticated approach, you could have separate networks for each deployment
                    action = torch.argmax(q_values, dim=1).item()
                    actions[i] = min(action, max_replicas)  # Ensure within bounds
                
                return actions
            else:
                # For discrete action space
                action = torch.argmax(q_values, dim=1).item()
                return action
        else:
            # Random action selection
            if hasattr(self.env.action_space, 'shape') and self.env.action_space.shape[0] > 1:
                # For MultiDiscrete, sample random actions for each deployment
                num_deployments = self.env.action_space.shape[0]
                max_replicas = self.env.action_space.nvec[0] - 1
                actions = np.zeros(num_deployments, dtype=np.int32)
                for i in range(num_deployments):
                    actions[i] = random.randint(0, max_replicas)
                return actions
            else:
                return self.env.action_space.sample()

    def calculate_metrics(self, state, action, reward, next_state, info, loss: Optional[float] = None) -> Dict[str, float]:
        """Calculate all required metrics for the current step."""
        state = self._flatten_observation(state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = self.model(state_tensor)
        
        # Handle array actions (for MultiDiscrete action space)
        if isinstance(action, (list, tuple, np.ndarray)):
            # For MultiDiscrete action space, use the average action value for Q-value calculation
            action_for_q = int(np.mean(action)) if len(action) > 0 else 0
        else:
            action_for_q = int(action)
        
        current_q = q_values[0][action_for_q].item()
        
        # Calculate accuracy (1 if reward is positive, 0 otherwise)
        accuracy = 1.0 if reward > 0 else 0.0
        
        # Calculate MSE (Mean Squared Error)
        mse = (reward - current_q) ** 2
        
        # Calculate MRE (Mean Relative Error)
        mre = abs(reward - current_q) / (abs(reward) + 1e-6)
        
        # Get latency from info and take absolute value
        avg_latency = abs(info.get("latency", 0))
        
        metrics = {
            "accuracy": accuracy,
            "mse": mse,
            "mre": mre,
            "avg_latency": avg_latency,
            "total_reward": reward
        }
        
        if loss is not None:
            metrics["loss"] = loss
            
        return metrics

    def train_step(self):
        if self.replay_buffer.size() < self.batch_size:
            return None

        state, action, reward, next_state, done = \
            self.replay_buffer.sample(self.batch_size)

        # Flatten observations
        state = torch.FloatTensor([self._flatten_observation(s) for s in state]).to(device)
        next_state = torch.FloatTensor([self._flatten_observation(s) for s in next_state]).to(device)
        
        # Handle array actions (for MultiDiscrete action space)
        if isinstance(action[0], (list, tuple, np.ndarray)):
            # For MultiDiscrete action space, use the average action value for training
            # In a more sophisticated approach, you could train separate networks for each deployment
            action = torch.LongTensor([int(np.mean(a)) if len(a) > 0 else 0 for a in action]).to(device)
        else:
            action = torch.LongTensor(action).to(device)
            
        reward = torch.FloatTensor(reward).to(device)
        done = torch.FloatTensor(done).to(device)

        q_values = self.model(state)
        next_q_values = self.target_model(next_state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.discount_factor * next_q_value * (1 - done)

        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


    def train(self, env, num_episodes: int, wandb_run_id: Optional[str] = None) -> Dict[str, List[float]]:
        """Train the DQN agent."""
        epsilon = epsilon_start
        episode_metrics = defaultdict(list)
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            self.metrics_collector.reset()
            
            while True:
                action = self.act(state, epsilon)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                self.replay_buffer.push(state, action, reward, next_state, done)
                loss = self.train_step()
                
                # Calculate and collect metrics
                step_metrics = self.calculate_metrics(state, action, reward, next_state, info, loss)
                self.metrics_collector.update(step_metrics)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break

            # Update exploration rate
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            if episode % target_update_freq == 0:
                self.update_target_network()

            # Get average metrics for the episode
            avg_metrics = self.metrics_collector.get_average_metrics()
            
            # Store episode metrics
            for key, value in avg_metrics.items():
                episode_metrics[key].append(value)
            
            # Log to wandb if run_id is provided
            if wandb_run_id:
                wandb.log({
                    "episode": episode,
                    **avg_metrics,
                    "total_reward": total_reward,
                    "steps": steps,
                    "epsilon": epsilon
                })

            logger.info(
                f"Episode {episode + 1}/{num_episodes}, "
                f"Total Reward: {total_reward:.2f}, "
                f"Steps: {steps}, "
                f"Epsilon: {epsilon:.3f}, "
                f"Accuracy: {avg_metrics.get('accuracy', 0):.3f}, "
                f"MSE: {avg_metrics.get('mse', 0):.3f}, "
                f"MRE: {avg_metrics.get('mre', 0):.3f}, "
                f"Avg Latency: {avg_metrics.get('avg_latency', 0):.2f}"
            )

        self.model.save_model()
        self.target_model.save_model(file_name="./dqn_target_model.pth")
        logger.info("Training completed.")
        
        return dict(episode_metrics)
    
    def save_model(self, path: str):
        """Save model parameters to a file."""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'target_model_state_dict': self.target_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon': self.epsilon,
                'batch_size': self.batch_size
            }, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, path: str):
        """Load model from a file."""
        try:
            checkpoint = torch.load(path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.learning_rate = checkpoint.get('learning_rate', self.learning_rate)
            self.discount_factor = checkpoint.get('discount_factor', self.discount_factor)
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.batch_size = checkpoint.get('batch_size', self.batch_size)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise


# Initialize Environment and Agent
if __name__ == "__main__":

    register(
        id="lwmecps-v3",
        entry_point="lwmecps_gym.envs:LWMECPSEnv3",
        max_episode_steps=5,
    )
    minikube = k8s()

    node_name = []

    max_hardware = {
        "cpu": 8,
        "ram": 16000,
        "tx_bandwidth": 1000,
        "rx_bandwidth": 1000,
        "read_disks_bandwidth": 500,
        "write_disks_bandwidth": 500,
        "avg_latency": 300,
    }

    pod_usage = {
        "cpu": 2,
        "ram": 2000,
        "tx_bandwidth": 20,
        "rx_bandwidth": 20,
        "read_disks_bandwidth": 100,
        "write_disks_bandwidth": 100,
    }

    state = minikube.k8s_state()

    max_pods = 10000

    for node in state:
        node_name.append(node)

    avg_latency = 10
    node_info = {}

    for node in state:
        avg_latency = avg_latency + 10
        node_info[node] = {
            "cpu": int(state[node]["cpu"]),
            "ram": round(
                bitmath.KiB(int(re.findall(r"\d+", state[node]["memory"])[0]))
                .to_MB()
                .value
            ),
            "tx_bandwidth": 100,
            "rx_bandwidth": 100,
            "read_disks_bandwidth": 300,
            "write_disks_bandwidth": 300,
            "avg_latency": avg_latency,
        }
        # Работатет только пока находятся в том же порядке.
        max_pods = min(
            [
                min(
                    [
                        val // pod_usage[key]
                        for key, (_, val) in zip(
                            pod_usage.keys(), node_info[node].items()
                        )
                    ]
                ),
                max_pods,
            ]
        )

    env = gym.make(
        "lwmecps-v3",
        node_name=node_name,
        max_hardware=max_hardware,
        pod_usage=pod_usage,
        node_info=node_info,
        num_nodes=len(node_name),
        namespace="default",
        deployments=["lwmecps-testapp-server-bs1", "lwmecps-testapp-server-bs2", "lwmecps-testapp-server-bs3", "lwmecps-testapp-server-bs4"],
        max_pods=max_pods,
        group_id="test-group-1"
    )

    replay_buffer = ReplayBuffer(replay_buffer_size)
    start = time()
    agent = DQNAgent(env, replay_buffer)
    agent.train(env, num_episodes)
    print(f"Training time: {(time() - start)}")

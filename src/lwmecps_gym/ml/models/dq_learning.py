import json
import os
import random
import re
from collections import deque
from time import time
from typing import Dict, List, Optional

import bitmath
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from gymnasium.envs.registration import register
from lwmecps_gym.envs.kubernetes_api import k8s

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


# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

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
    def __init__(self, env, replay_buffer):
        self.env = env
        self.replay_buffer = replay_buffer
        self.model = DQN(env.observation_space.shape[0],
                         env.action_space.n).to(device)
        self.target_model = DQN(
            env.observation_space.shape[0],
            env.action_space.n).to(
            device
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.update_target_network()
        self.metrics_collector = MetricsCollector()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.model(state)
            action = torch.argmax(q_values, dim=1).item()
        else:
            action = self.env.action_space.sample()
        return action

    def calculate_metrics(self, state, action, reward, next_state, info, loss: Optional[float] = None) -> Dict[str, float]:
        """Calculate all required metrics for the current step."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = self.model(state_tensor)
        current_q = q_values[0][action].item()
        
        # Calculate accuracy (1 if reward is positive, 0 otherwise)
        accuracy = 1.0 if reward > 0 else 0.0
        
        # Calculate MSE (Mean Squared Error)
        mse = (reward - current_q) ** 2
        
        # Calculate MRE (Mean Relative Error)
        mre = abs(reward - current_q) / (abs(reward) + 1e-6)
        
        # Get latency from info
        avg_latency = info.get("latency", 0)
        
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
        if self.replay_buffer.size() < batch_size:
            return None

        state, action, reward, next_state, done = \
            self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.LongTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        done = torch.FloatTensor(done).to(device)

        q_values = self.model(state)
        next_q_values = self.target_model(next_state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + gamma * next_q_value * (1 - done)

        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


# Training the DQN Agent
def train_dqn(agent, num_episodes, wandb_run_id: Optional[str] = None):
    epsilon = epsilon_start
    episode_metrics = {}
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        agent.metrics_collector.reset()
        
        for step in range(max_steps_per_episode):
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
            loss = agent.train_step()
            
            # Calculate and collect metrics
            step_metrics = agent.calculate_metrics(state, action, reward, next_state, info, loss)
            agent.metrics_collector.update(step_metrics)
            
            state = next_state
            total_reward += reward
            
            if done:
                break

        # Update exploration rate
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if episode % target_update_freq == 0:
            agent.update_target_network()

        # Get average metrics for the episode
        avg_metrics = agent.metrics_collector.get_average_metrics()
        
        # Store episode metrics
        for key, value in avg_metrics.items():
            if key not in episode_metrics:
                episode_metrics[key] = []
            episode_metrics[key].append(value)
        
        # Log to wandb if run_id is provided
        if wandb_run_id:
            wandb.log({
                "episode": episode,
                **avg_metrics,
                "total_reward": total_reward,
                "steps": step + 1,
                "epsilon": epsilon
            })

        print(
            f"Episode {episode}, Total Reward: {total_reward:.2f}, "
            f"Epsilon: {epsilon:.3f}, "
            f"Accuracy: {avg_metrics.get('accuracy', 0):.3f}, "
            f"MSE: {avg_metrics.get('mse', 0):.3f}, "
            f"MRE: {avg_metrics.get('mre', 0):.3f}, "
            f"Avg Latency: {avg_metrics.get('avg_latency', 0):.2f}"
        )

    agent.model.save_model()
    agent.target_model.save_model(file_name="./dqn_target_model.pth")
    print("Training end.")
    
    return episode_metrics


# Initialize Environment and Agent
if __name__ == "__main__":

    register(
        id="lwmecps-v0",
        entry_point="lwmecps_gym.envs:LWMECPSEnv",
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
        "lwmecps-v1",
        num_nodes=len(node_name),
        node_name=node_name,
        max_hardware=max_hardware,
        pod_usage=pod_usage,
        node_info=node_info,
        deployment_name="mec-test-app",
        namespace="default",
        deployments=["mec-test-app"],
        max_pods=max_pods,
    )

    replay_buffer = ReplayBuffer(replay_buffer_size)
    start = time()
    agent = DQNAgent(env, replay_buffer)
    train_dqn(agent, num_episodes)
    print(f"Training time: {(time() - start)}")

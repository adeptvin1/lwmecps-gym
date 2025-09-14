import json
import pickle
import random
import re
from time import time
import os
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
from collections import defaultdict

import bitmath
import gymnasium as gym
import numpy as np
import wandb
from gymnasium.envs.registration import register
from lwmecps_gym.envs.kubernetes_api import k8s

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.metric_validators = {
            "accuracy": lambda x: 0 <= x <= 1,
            "mse": lambda x: x >= 0,
            "mre": lambda x: x >= 0,
            "avg_latency": lambda x: x >= 0,
            "total_reward": lambda x: True,  # Can be negative
            "exploration_rate": lambda x: 0 <= x <= 1
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

class QLearningAgent:
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        exploration_rate: float = 1.0,
        exploration_decay: float = 0.995,
        min_exploration_rate: float = 0.01,
        max_states: int = 10000
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.max_states = max_states
        self.q_table = {}
        self.state_visits = {}
        self.metrics_collector = MetricsCollector()
        
    def _cleanup_q_table(self):
        """Remove least visited states if table size exceeds max_states."""
        if len(self.q_table) > self.max_states:
            sorted_states = sorted(self.state_visits.items(), key=lambda x: x[1])
            states_to_remove = sorted_states[:len(self.q_table) - self.max_states]
            for state, _ in states_to_remove:
                del self.q_table[state]
                del self.state_visits[state]
    
    def save_q_table(self, file_name="./q_table.json"):
        """Save Q-table to a JSON file."""
        with open(file_name, "w") as file:
            json.dump(self.q_table, file, indent=4)
        logger.info(f"Q-table saved to {file_name}")
    
    def load_q_table(self, file_name="./q_table.json"):
        """Load Q-table from a JSON file."""
        if os.path.exists(file_name):
            with open(file_name, "r") as file:
                self.q_table = json.load(file)
            logger.info(f"Q-table loaded from {file_name}")
        else:
            logger.info(f"No Q-table found at {file_name}")
    
    def _validate_state(self, state):
        """Validate and format state for Q-table."""
        if isinstance(state, dict):
            # Convert dict to string representation
            state_str = json.dumps(state, sort_keys=True)
        else:
            state_str = str(state)
        return state_str
    
    def choose_action(self, state, valid_actions=None):
        """Choose action using epsilon-greedy policy."""
        state_str = self._validate_state(state)
        
        # Get action space size from environment or use default
        action_size = getattr(self, 'action_space_size', 4)
        
        # Initialize state in Q-table if not present
        if state_str not in self.q_table:
            self.q_table[state_str] = {action: 0.0 for action in range(action_size)}
            self.state_visits[state_str] = 0
        
        # Update state visit count
        self.state_visits[state_str] += 1
        
        # For MultiDiscrete action space, we need to choose actions for each deployment
        if hasattr(self, 'env') and hasattr(self.env, 'action_space'):
            if hasattr(self.env.action_space, 'shape') and self.env.action_space.shape[0] > 1:
                # For MultiDiscrete action space, return array of actions
                import numpy as np
                num_deployments = self.env.action_space.shape[0]
                max_replicas = self.env.action_space.nvec[0] - 1  # Get max replicas from action space
                
                actions = np.zeros(num_deployments, dtype=np.int32)
                
                # Choose action for each deployment
                for i in range(num_deployments):
                    if random.random() < self.exploration_rate:
                        # Random action for this deployment
                        actions[i] = random.randint(0, max_replicas)
                    else:
                        # Greedy action for this deployment
                        # For simplicity, use the same Q-table for all deployments
                        # In a more sophisticated approach, you could have separate Q-tables for each deployment
                        action = max(self.q_table[state_str].items(), key=lambda x: x[1])[0]
                        actions[i] = min(action, max_replicas)  # Ensure within bounds
                
                return actions
        
        # For single action space
        if random.random() < self.exploration_rate:
            if valid_actions:
                action = random.choice(valid_actions)
            else:
                action = random.choice(list(self.q_table[state_str].keys()))
        else:
            if valid_actions:
                valid_q_values = {action: self.q_table[state_str][action] for action in valid_actions}
                action = max(valid_q_values.items(), key=lambda x: x[1])[0]
            else:
                action = max(self.q_table[state_str].items(), key=lambda x: x[1])[0]
        
        return action
    
    def update_q_table(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning update rule."""
        state_str = self._validate_state(state)
        next_state_str = self._validate_state(next_state)
        
        # Handle array actions (for MultiDiscrete action space)
        if isinstance(action, (list, tuple, np.ndarray)):
            # For MultiDiscrete action space, we need to update Q-values for each deployment
            # For simplicity, we'll use the average action value
            # In a more sophisticated approach, you could have separate Q-tables for each deployment
            action = int(np.mean(action)) if len(action) > 0 else 0
        else:
            action = int(action)
        
        # Get action space size from environment or use default
        action_size = getattr(self, 'action_space_size', 4)
        
        # Initialize states in Q-table if not present
        if state_str not in self.q_table:
            self.q_table[state_str] = {action: 0.0 for action in range(action_size)}
            self.state_visits[state_str] = 0
        if next_state_str not in self.q_table:
            self.q_table[next_state_str] = {action: 0.0 for action in range(action_size)}
            self.state_visits[next_state_str] = 0
        
        # Ensure action is within valid range
        action = max(0, min(action, action_size - 1))
        
        # Q-learning update
        current_q = self.q_table[state_str][action]
        if done:
            max_next_q = 0
        else:
            max_next_q = max(self.q_table[next_state_str].values())
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state_str][action] = new_q
        
        # Cleanup if necessary
        self._cleanup_q_table()
    
    def get_current_node(self, state):
        """Extract current node from state."""
        if isinstance(state, dict) and "current_node" in state:
            return state["current_node"]
        return None
    
    def calculate_metrics(self, state, action, reward, next_state, info) -> Dict[str, float]:
        """Calculate all required metrics for the current step."""
        state_str = self._validate_state(state)
        
        # Handle array actions (for MultiDiscrete action space)
        if isinstance(action, (list, tuple, np.ndarray)):
            # For MultiDiscrete action space, use the average action value
            action = int(np.mean(action)) if len(action) > 0 else 0
        else:
            action = int(action)
        
        current_q = self.q_table[state_str][action]
        
        # Calculate accuracy (1 if reward is positive, 0 otherwise)
        accuracy = 1.0 if reward > 0 else 0.0
        
        # Calculate MSE (Mean Squared Error)
        mse = (reward - current_q) ** 2
        
        # Calculate MRE (Mean Relative Error)
        mre = abs(reward - current_q) / (abs(reward) + 1e-6)
        
        # Get latency from info and take absolute value
        avg_latency = abs(info.get("latency", 0))
        
        return {
            "accuracy": accuracy,
            "mse": mse,
            "mre": mre,
            "avg_latency": avg_latency,
            "exploration_rate": self.exploration_rate,
            "total_reward": reward
        }
    
    def train(self, env, num_episodes: int, wandb_run_id: str = None) -> Dict[str, List[float]]:
        """Train the agent."""
        # Store environment reference for action conversion
        self.env = env
        
        # Set action space size from environment
        if hasattr(env.action_space, 'n'):
            self.action_space_size = env.action_space.n
        elif hasattr(env.action_space, 'shape'):
            # For MultiDiscrete action space, use the maximum value from nvec
            if hasattr(env.action_space, 'nvec'):
                self.action_space_size = max(env.action_space.nvec)
            else:
                self.action_space_size = env.action_space.shape[0]
        else:
            self.action_space_size = 4  # Default fallback
            
        episode_metrics = defaultdict(list)
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            self.metrics_collector.reset()
            
            while True:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Calculate and collect metrics
                step_metrics = self.calculate_metrics(state, action, reward, next_state, info)
                self.metrics_collector.update(step_metrics)
                
                self.update_q_table(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Update exploration rate
            self.exploration_rate = max(self.min_exploration_rate, 
                                     self.exploration_rate * self.exploration_decay)
            
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
                    "steps": steps
                })
            
            logger.info(
                f"Episode {episode + 1}/{num_episodes}, "
                f"Total Reward: {total_reward:.2f}, "
                f"Steps: {steps}, "
                f"Exploration Rate: {self.exploration_rate:.3f}, "
                f"Accuracy: {avg_metrics.get('accuracy', 0):.3f}, "
                f"MSE: {avg_metrics.get('mse', 0):.3f}, "
                f"MRE: {avg_metrics.get('mre', 0):.3f}, "
                f"Avg Latency: {avg_metrics.get('avg_latency', 0):.2f}"
            )
        
        return dict(episode_metrics)


if __name__ == "__main__":
    # Регистрируем окружение
    register(
        id="lwmecps-v3",
        entry_point="lwmecps_gym.envs:LWMECPSEnv3",
        max_episode_steps=5,
    )

    # Инициализируем Kubernetes клиент
    minikube = k8s()
    state = minikube.k8s_state()
    node_name = list(state.keys())

    # Базовые параметры
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

    # Создаем информацию о узлах
    node_info = {}
    for node in node_name:
        node_info[node] = {
            "cpu": int(state[node]["cpu"]),
            "ram": round(bitmath.KiB(int(re.findall(r"\d+", state[node]["memory"])[0])).to_MB().value),
            "tx_bandwidth": 100,
            "rx_bandwidth": 100,
            "read_disks_bandwidth": 300,
            "write_disks_bandwidth": 300,
            "avg_latency": 10 + (10 * list(node_name).index(node)),  # Увеличиваем задержку для каждого следующего узла
        }

    # Создаем окружение
    env = gym.make(
        "lwmecps-v3",
        node_name=node_name,
        max_hardware=max_hardware,
        pod_usage=pod_usage,
        node_info=node_info,
        num_nodes=len(node_name),
        namespace="default",
        deployment_name="mec-test-app",
        deployments=["mec-test-app"],
        max_pods=10000,
        group_id="test-group-1"
    )

    # Создаем и обучаем агента
    agent = QLearningAgent()
    start = time()
    agent.train(env, num_episodes=100)
    print(f"Training time: {(time() - start)}")
    agent.save_q_table("./q_table.json")

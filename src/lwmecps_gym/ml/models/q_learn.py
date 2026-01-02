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
        
        # Store dimensions for model saving/loading compatibility
        self.obs_dim = None
        self.act_dim = None
        self.max_replicas = None
        self.deployments = []
        
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
            # Convert dict to string representation, handling numpy types
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            try:
                converted_state = convert_numpy(state)
                state_str = json.dumps(converted_state, sort_keys=True)
            except (TypeError, ValueError) as e:
                # Fallback to string representation if JSON serialization fails
                logger.warning(f"Failed to serialize state to JSON: {e}. Using string representation.")
                state_str = str(state)
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
            logger.info(f"New state added to Q-table. Total states: {len(self.q_table)}")
        
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
                is_exploration = random.random() < self.exploration_rate
                
                logger.info(f"MultiDiscrete action space: num_deployments={num_deployments}, max_replicas={max_replicas}, exploration={is_exploration}")
                
                # Choose action for each deployment
                for i in range(num_deployments):
                    if is_exploration:
                        # Random action for this deployment
                        actions[i] = random.randint(0, max_replicas)
                    else:
                        # Greedy action for this deployment
                        # For simplicity, use the same Q-table for all deployments
                        # In a more sophisticated approach, you could have separate Q-tables for each deployment
                        action = max(self.q_table[state_str].items(), key=lambda x: x[1])[0]
                        actions[i] = min(action, max_replicas)  # Ensure within bounds
                
                logger.info(f"Chosen actions: {actions}")
                return actions
        
        # For single action space
        is_exploration = random.random() < self.exploration_rate
        logger.info(f"Single action space: exploration={is_exploration}")
        
        if is_exploration:
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
        
        logger.info(f"Chosen action: {action}")
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
            logger.info(f"Array action converted to: {action}")
        else:
            action = int(action)
        
        # Get action space size from environment or use default
        action_size = getattr(self, 'action_space_size', 4)
        
        # Initialize states in Q-table if not present
        if state_str not in self.q_table:
            self.q_table[state_str] = {action: 0.0 for action in range(action_size)}
            self.state_visits[state_str] = 0
            logger.info(f"New state added to Q-table during update. Total states: {len(self.q_table)}")
        if next_state_str not in self.q_table:
            self.q_table[next_state_str] = {action: 0.0 for action in range(action_size)}
            self.state_visits[next_state_str] = 0
            logger.info(f"New next state added to Q-table. Total states: {len(self.q_table)}")
        
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
        
        logger.info(f"Q-update: state={state_str[:50]}..., action={action}, reward={reward}, current_q={current_q:.4f}, new_q={new_q:.4f}")
        
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
        
        # Ensure state exists in Q-table and action is within bounds
        action_size = getattr(self, 'action_space_size', 4)
        if state_str not in self.q_table:
            self.q_table[state_str] = {action: 0.0 for action in range(action_size)}
            self.state_visits[state_str] = 0
            logger.info(f"State added to Q-table in calculate_metrics. Total states: {len(self.q_table)}")
        
        # Ensure action is within valid range
        action = max(0, min(action, action_size - 1))
        
        current_q = self.q_table[state_str][action]
        
        # Calculate accuracy (1 if reward is positive, 0 otherwise)
        accuracy = 1.0 if reward > 0 else 0.0
        
        # Calculate MSE (Mean Squared Error)
        mse = (reward - current_q) ** 2
        
        # Calculate MRE (Mean Relative Error)
        mre = abs(reward - current_q) / (abs(reward) + 1e-6)
        
        # Get latency from info and take absolute value
        avg_latency = abs(info.get("latency", 0))
        
        # Get additional metrics from info to match other algorithms
        cpu_usage = info.get("cpu_usage", 0)
        ram_usage = info.get("ram_usage", 0)
        network_usage = info.get("network_usage", 0)
        
        return {
            "accuracy": accuracy,
            "mse": mse,
            "mre": mre,
            "avg_latency": avg_latency,
            "total_reward": reward,
            "q_value": current_q,  # Q-value for consistency with other algorithms
            "exploration_rate": self.exploration_rate,
            "cpu_usage": cpu_usage,
            "ram_usage": ram_usage,
            "network_usage": network_usage
        }
    
    def train(self, env, num_episodes: int, wandb_run_id: str = None) -> Dict[str, List[float]]:
        """Train the agent."""
        logger.info(f"Starting Q-learning training for {num_episodes} episodes")
        
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
        
        # Set dimensions for model saving/loading compatibility
        self.act_dim = self.action_space_size
        self.max_replicas = getattr(env.action_space, 'nvec', [10])[0] - 1 if hasattr(env.action_space, 'nvec') else 10
        self.deployments = getattr(env, 'deployments', [])
        
        # Calculate obs_dim from environment
        try:
            sample_obs, _ = env.reset()
            if isinstance(sample_obs, dict):
                # For Dict observation space, calculate dimension
                obs_vector = []
                for node in sample_obs:
                    node_obs = sample_obs[node]
                    obs_vector.extend([
                        node_obs["cpu"], node_obs["ram"], 
                        node_obs["tx_bandwidth"], node_obs["rx_bandwidth"]
                    ])
                    if "deployments" in node_obs:
                        for deployment in node_obs["deployments"]:
                            dep_obs = node_obs["deployments"][deployment]
                            obs_vector.extend([
                                dep_obs["CPU_usage"], dep_obs["RAM_usage"],
                                dep_obs["TX_usage"], dep_obs["RX_usage"],
                                dep_obs["Replicas"]
                            ])
                    obs_vector.append(node_obs.get("latency", 0))
                self.obs_dim = len(obs_vector)
            else:
                self.obs_dim = len(sample_obs) if hasattr(sample_obs, '__len__') else 25
        except Exception as e:
            logger.warning(f"Could not calculate obs_dim: {e}")
            self.obs_dim = 25  # Default fallback
        
        logger.info(f"Action space size: {self.action_space_size}")
        logger.info(f"Initial exploration rate: {self.exploration_rate}")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info(f"Discount factor: {self.discount_factor}")
            
        episode_metrics = defaultdict(list)
        
        # Explicitly start the workload before the training loop
        if hasattr(env, 'start_workload'):
            env.start_workload()
        elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'start_workload'):
            env.unwrapped.start_workload()

        for episode in range(num_episodes):
            logger.info(f"Starting episode {episode + 1}/{num_episodes}")
            state, info = env.reset()
            
            if info.get("group_completed"):
                logger.info("Experiment group finished (detected at reset). Stopping training.")
                break

            total_reward = 0
            steps = 0
            self.metrics_collector.reset()
            
            logger.info(f"Episode {episode + 1}: Initial state type: {type(state)}")
            
            while True:
                action = self.choose_action(state)
                logger.info(f"Episode {episode + 1}, Step {steps + 1}: Action chosen: {action} (type: {type(action)})")
                
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                if info.get("group_completed"):
                    logger.info("Experiment group finished. Stopping training.")
                    done = True

                logger.info(f"Episode {episode + 1}, Step {steps + 1}: Reward: {reward}, Done: {done}")
                logger.info(f"Episode {episode + 1}, Step {steps + 1}: Info: {info}")
                
                # Calculate and collect metrics
                step_metrics = self.calculate_metrics(state, action, reward, next_state, info)
                self.metrics_collector.update(step_metrics)
                
                self.update_q_table(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if info.get("group_completed"):
                    break

                if done:
                    logger.info(f"Episode {episode + 1} completed: Steps: {steps}, Total reward: {total_reward:.2f}")
                    break
            
            if info.get("group_completed"):
                break
            
            # Update exploration rate
            old_exploration_rate = self.exploration_rate
            self.exploration_rate = max(self.min_exploration_rate, 
                                     self.exploration_rate * self.exploration_decay)
            
            # Get average metrics for the episode
            avg_metrics = self.metrics_collector.get_average_metrics()
            
            # Store episode metrics with proper naming for training_service compatibility
            episode_metrics["episode_reward"].append(total_reward)
            episode_metrics["episode_steps"].append(steps)
            episode_metrics["episode_exploration"].append(self.exploration_rate)
            episode_metrics["episode_latency"].append(avg_metrics.get('avg_latency', 0))
            
            # Also store all other metrics for completeness
            for key, value in avg_metrics.items():
                if key not in ["avg_latency"]:  # Already stored as episode_latency
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
                f"Episode {episode + 1}/{num_episodes} completed: "
                f"Reward: {total_reward:.2f}, "
                f"Steps: {steps}, "
                f"Exploration: {old_exploration_rate:.3f} -> {self.exploration_rate:.3f}, "
                f"Q-table size: {len(self.q_table)}, "
                f"Accuracy: {avg_metrics.get('accuracy', 0):.3f}, "
                f"MSE: {avg_metrics.get('mse', 0):.3f}, "
                f"MRE: {avg_metrics.get('mre', 0):.3f}, "
                f"Avg Latency: {avg_metrics.get('avg_latency', 0):.2f}"
            )
        
        logger.info(f"Q-learning training completed!")
        logger.info(f"Final Q-table size: {len(self.q_table)}")
        logger.info(f"Final exploration rate: {self.exploration_rate:.4f}")
        logger.info(f"Total unique states visited: {len(self.state_visits)}")
        
        return dict(episode_metrics)

    def save_model(self, path: str):
        """
        Save Q-learning model to a file.
        This method provides compatibility with training service.
        
        Args:
            path (str): Path to save the model
        """
        try:
            # Save Q-table as JSON
            q_table_path = path.replace('.pth', '_q_table.json')
            self.save_q_table(q_table_path)
            
            # Save model metadata as torch checkpoint for compatibility with torch.load
            import torch
            model_data = {
                'q_table_path': q_table_path,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'exploration_rate': self.exploration_rate,
                'exploration_decay': self.exploration_decay,
                'min_exploration_rate': self.min_exploration_rate,
                'max_states': self.max_states,
                'obs_dim': self.obs_dim,
                'act_dim': self.act_dim,
                'deployments': self.deployments,
                'max_replicas': self.max_replicas
            }
            
            torch.save(model_data, path)
            
            logger.info(f"Q-learning model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving Q-learning model: {str(e)}")
            raise

    def load_model(self, path: str):
        """
        Load Q-learning model from a file.
        This method provides compatibility with training service.
        
        Args:
            path (str): Path to load the model from
        """
        try:
            import torch
            
            # Load model metadata using torch.load
            model_data = torch.load(path, map_location='cpu', weights_only=False)
            
            # Load Q-table
            q_table_path = model_data.get('q_table_path', path.replace('.pth', '_q_table.json'))
            if os.path.exists(q_table_path):
                self.load_q_table(q_table_path)
            else:
                logger.warning(f"Q-table file not found at {q_table_path}")
            
            # Load parameters
            self.learning_rate = model_data.get('learning_rate', self.learning_rate)
            self.discount_factor = model_data.get('discount_factor', self.discount_factor)
            self.exploration_rate = model_data.get('exploration_rate', self.exploration_rate)
            self.exploration_decay = model_data.get('exploration_decay', self.exploration_decay)
            self.min_exploration_rate = model_data.get('min_exploration_rate', self.min_exploration_rate)
            self.max_states = model_data.get('max_states', self.max_states)
            
            # Load dimensions if available (for compatibility)
            if 'obs_dim' in model_data:
                self.obs_dim = model_data['obs_dim']
            if 'act_dim' in model_data:
                self.act_dim = model_data['act_dim']
            if 'deployments' in model_data:
                self.deployments = model_data['deployments']
            if 'max_replicas' in model_data:
                self.max_replicas = model_data['max_replicas']
            
            logger.info(f"Q-learning model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading Q-learning model: {str(e)}")
            raise


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

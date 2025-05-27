import json
import pickle
import random
import re
from time import time
import os
import logging
from typing import Dict, List, Tuple, Union, Optional, Any

import bitmath
import gymnasium as gym
import numpy as np
import wandb
from gymnasium.envs.registration import register
from lwmecps_gym.envs.kubernetes_api import k8s

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QLearningAgent:
    def __init__(
        self,
        env,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        exploration_rate: float = 1.0,
        exploration_decay: float = 0.98,
        min_exploration_rate: float = 0.01,
        max_states: int = 1000,
        wandb_run_id: Optional[str] = None,
    ):
        """
        Initialize Q-learning agent.
        
        Args:
            env: Gymnasium environment
            learning_rate: Learning rate for Q-value updates
            discount_factor: Discount factor for future rewards
            exploration_rate: Initial exploration rate (epsilon)
            exploration_decay: Rate at which exploration rate decays
            min_exploration_rate: Minimum exploration rate
            max_states: Maximum number of states to store in Q-table
            wandb_run_id: Optional Weights & Biases run ID for logging
        """
        self.env = env
        # Get the original environment by unwrapping all wrappers
        self.original_env = env
        while hasattr(self.original_env, 'env'):
            self.original_env = self.original_env.env
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.max_states = max_states
        self.wandb_run_id = wandb_run_id

        # Initialize Q-table for each node and action
        self.q_table = {}
        self.state_visits = {}  # Track state visits for cleanup

        # Initialize wandb
        if self.wandb_run_id:
            try:
                wandb.init(
                    project="lwmecps-gym",
                    id=self.wandb_run_id,
                    config={
                        "learning_rate": learning_rate,
                        "discount_factor": discount_factor,
                        "exploration_rate": exploration_rate,
                        "exploration_decay": exploration_decay,
                        "min_exploration_rate": min_exploration_rate,
                        "max_states": max_states,
                        "model_type": "q_learning",
                    }
                )
                logger.info(f"Successfully initialized wandb run with ID {self.wandb_run_id}")
            except Exception as e:
                logger.error(f"Failed to initialize wandb: {str(e)}")
                self.wandb_run_id = None

    def _cleanup_q_table(self):
        """Remove least visited states from Q-table when it exceeds max_states."""
        if len(self.q_table) <= self.max_states:
            return
            
        # Sort states by visit count
        sorted_states = sorted(self.state_visits.items(), key=lambda x: x[1])
        states_to_remove = len(self.q_table) - self.max_states
        
        # Remove least visited states
        for state, _ in sorted_states[:states_to_remove]:
            del self.q_table[state]
            del self.state_visits[state]
            
        logger.info(f"Cleaned up Q-table, removed {states_to_remove} states")

    def save_q_table(self, file_name: str) -> None:
        """
        Save Q-table to file.
        
        Args:
            file_name: Path to save Q-table
        """
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(file_name)), exist_ok=True)
            
            # Save locally
            with open(file_name, "wb") as f:
                pickle.dump(self.q_table, f)
            logger.info(f"Q-table saved to {file_name}")
            
            # Save to wandb if initialized
            if self.wandb_run_id:
                try:
                    # Save as a wandb artifact
                    artifact = wandb.Artifact('q_table', type='model')
                    artifact.add_file(file_name)
                    wandb.log_artifact(artifact)
                    logger.info("Successfully saved model to wandb as artifact")
                    
                    # Also save directly
                    wandb.save(file_name)
                    logger.info("Successfully saved model file to wandb")
                except Exception as e:
                    logger.error(f"Failed to save model to wandb: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error saving Q-table: {str(e)}")
            raise

    def load_q_table(self, file_name: str) -> None:
        """
        Load Q-table from file.
        
        Args:
            file_name: Path to load Q-table from
        """
        try:
            with open(file_name, "rb") as f:
                self.q_table = pickle.load(f)
            logger.info(f"Q-table loaded from {file_name}")
            
            # Initialize state visits for loaded states
            self.state_visits = {state: 0 for state in self.q_table.keys()}
            
        except Exception as e:
            logger.error(f"Error loading Q-table: {str(e)}")
            raise

    def validate_state(self, state: Dict[str, Any]) -> bool:
        """
        Validate environment state.
        
        Args:
            state: Environment state to validate
            
        Returns:
            bool: True if state is valid, False otherwise
        """
        try:
            if not isinstance(state, dict):
                return False
                
            for node in self.original_env.node_name:
                if node not in state:
                    return False
                node_state = state[node]
                if not isinstance(node_state, dict):
                    return False
                if "deployments" not in node_state or "avg_latency" not in node_state:
                    return False
                if not isinstance(node_state["deployments"], dict):
                    return False
                if not isinstance(node_state["avg_latency"], (int, float)):
                    return False
            return True
        except Exception as e:
            logger.error(f"Error validating state: {str(e)}")
            return False

    def choose_action(self, state: Dict[str, Any]) -> int:
        """
        Choose action based on current state.
        
        Args:
            state: Current environment state
            
        Returns:
            int: Chosen action
        """
        if not self.validate_state(state):
            raise ValueError("Invalid state")
            
        node = self.get_current_node(state)
        if node is None:
            raise ValueError("Could not determine current node")
            
        if random.uniform(0, 1) < self.exploration_rate:
            return self.original_env.action_space.sample()  # Exploration
        else:
            if node not in self.q_table:
                self.q_table[node] = np.zeros(self.original_env.action_space.n)
                self.state_visits[node] = 0
            return np.argmax(self.q_table[node])  # Exploitation

    def update_q_table(self, state: Dict[str, Any], action: int, reward: float, next_state: Dict[str, Any]) -> None:
        """
        Update Q-table based on experience.
        
        Args:
            state: Current state
            action: Taken action
            reward: Received reward
            next_state: Next state
        """
        try:
            if not self.validate_state(state) or not self.validate_state(next_state):
                raise ValueError("Invalid state")
                
            node = self.get_current_node(state)
            next_node = self.get_current_node(next_state)
            
            if node is None or next_node is None:
                raise ValueError("Could not determine current or next node")
                
            # Initialize Q-values for new states
            if node not in self.q_table:
                self.q_table[node] = np.zeros(self.original_env.action_space.n)
                self.state_visits[node] = 0
            if next_node not in self.q_table:
                self.q_table[next_node] = np.zeros(self.original_env.action_space.n)
                self.state_visits[next_node] = 0
                
            # Update state visit count
            self.state_visits[node] += 1
            
            # Cleanup if needed
            self._cleanup_q_table()
            
            best_next_action = np.argmax(self.q_table[next_node])

            # Update Q-value
            q_value = self.q_table[node][action]
            self.q_table[node][action] = q_value + self.learning_rate * (
                reward
                + self.discount_factor * self.q_table[next_node][best_next_action]
                - q_value
            )

            # Log Q-value update to wandb
            if self.wandb_run_id:
                try:
                    wandb.log({
                        f"q_value/{node}/{action}": self.q_table[node][action],
                        f"q_value_change/{node}/{action}": self.q_table[node][action] - q_value,
                        "exploration_rate": self.exploration_rate,
                    })
                except Exception as e:
                    logger.error(f"Failed to log to wandb: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error updating Q-table: {str(e)}")
            raise

    def get_current_node(self, state: Dict[str, Any]) -> Optional[str]:
        """
        Get current node from state.
        
        Args:
            state: Environment state
            
        Returns:
            Optional[str]: Current node name or None if not found
        """
        try:
            if not self.validate_state(state):
                return None
                
            for node in self.original_env.node_name:
                if node in state:
                    deployments = state[node]["deployments"]
                    if self.original_env.deployment_name in deployments:
                        if deployments[self.original_env.deployment_name]["replicas"] > 0:
                            return node
            return None
        except Exception as e:
            logger.error(f"Error getting current node: {str(e)}")
            return None

    def train(self, episodes: int) -> Dict[str, Dict[int, Any]]:
        """
        Train the agent.
        
        Args:
            episodes: Number of episodes to train
            
        Returns:
            Dict[str, Dict[int, Any]]: Training results
        """
        episode_latency = {}
        episode_reward = {}
        episode_steps = {}
        episode_exploration = {}
        
        try:
            for episode in range(episodes):
                logger.info(f"\nStarting episode {episode + 1}/{episodes}")
                state = self.env.reset()[0]  # Get only observation
                total_reward = 0
                steps = 0
                
                while True:
                    action = self.choose_action(state)
                    next_state, reward, done, truncated, info = self.env.step(action)
                    self.update_q_table(state, action, reward, next_state)
                    
                    total_reward += reward
                    steps += 1
                    
                    # Log current state and reward
                    current_node = self.get_current_node(state)
                    logger.info(f"Step {steps}:")
                    logger.info(f"  Current node: {current_node}")
                    logger.info(f"  Action: {action} (node {self.original_env.node_name[action]})")
                    logger.info(f"  Reward: {reward}")
                    logger.info(f"  Total reward so far: {total_reward}")
                    
                    if done or truncated:
                        break
                    
                    state = next_state
                
                # Save episode metrics
                current_node = self.get_current_node(state)
                if current_node:
                    episode_latency[episode] = state[current_node]["avg_latency"]
                episode_reward[episode] = total_reward
                episode_steps[episode] = steps
                episode_exploration[episode] = self.exploration_rate
                
                # Update exploration rate
                self.exploration_rate = max(
                    self.min_exploration_rate,
                    self.exploration_rate * self.exploration_decay
                )
                
                # Log episode results
                logger.info(f"\nEpisode {episode + 1} completed:")
                logger.info(f"  Total steps: {steps}")
                logger.info(f"  Total reward: {total_reward}")
                if current_node:
                    logger.info(f"  Final latency: {state[current_node]['avg_latency']}ms")
                logger.info(f"  Exploration rate: {self.exploration_rate}")
                
                # Log metrics to wandb
                if self.wandb_run_id:
                    try:
                        metrics = {
                            "episode": episode,
                            "total_reward": total_reward,
                            "steps": steps,
                            "epsilon": self.exploration_rate,
                        }
                        if current_node:
                            metrics["latency"] = state[current_node]["avg_latency"]
                        wandb.log(metrics)
                        logger.info(f"Successfully logged metrics for episode {episode}")
                    except Exception as e:
                        logger.error(f"Failed to log metrics to wandb: {str(e)}")
            
            return {
                "episode_latency": episode_latency,
                "episode_reward": episode_reward,
                "episode_steps": episode_steps,
                "episode_exploration": episode_exploration
            }
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise


if __name__ == "__main__":
    # Регистрируем окружение
    register(
        id="lwmecps-v3",
        entry_point="lwmecps_gym.envs:LWMECPSEnv3",
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
    agent = QLearningAgent(env)
    start = time()
    agent.train(episodes=100)
    print(f"Training time: {(time() - start)}")
    agent.save_q_table("./q_table.pkl")

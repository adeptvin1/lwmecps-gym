"""
Simple heuristic baseline algorithms for pod placement in MEC infrastructure.

This module provides non-learning baseline strategies for comparison with RL algorithms:
- Uniform: Distributes replicas evenly across nodes
- Static: Uses fixed replica count
- Greedy Latency: Places replicas on nodes with minimum latency
- Greedy Load: Places replicas on nodes with minimum load
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
import gymnasium as gym

logger = logging.getLogger(__name__)


class HeuristicBaseline:
    """
    Base class for heuristic baseline algorithms.
    Provides common interface compatible with RL agents.
    """
    
    def __init__(
        self,
        heuristic_type: str = "uniform",
        num_deployments: int = 4,
        max_replicas: int = 10,
        static_replicas: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize heuristic baseline.
        
        Args:
            heuristic_type: Type of heuristic ("uniform", "static", "greedy_latency", "greedy_load")
            num_deployments: Number of deployments to manage
            max_replicas: Maximum number of replicas per deployment
            static_replicas: Fixed number of replicas for static heuristic
        """
        self.heuristic_type = heuristic_type
        self.num_deployments = num_deployments
        self.max_replicas = max_replicas
        self.static_replicas = static_replicas or (max_replicas // 2)
        
        logger.info(f"Initialized {heuristic_type} heuristic with {num_deployments} deployments, max_replicas={max_replicas}")
    
    def _extract_state_info(self, observation: Dict) -> Tuple[List[str], Dict[str, Dict]]:
        """
        Extract node names and state information from observation.
        
        Args:
            observation: Observation dictionary from environment
            
        Returns:
            Tuple of (node_names, node_states)
        """
        if "nodes" not in observation:
            raise ValueError("Observation must contain 'nodes' key")
        
        nodes = observation["nodes"]
        node_names = list(nodes.keys())
        node_states = {}
        
        for node_name, node_data in nodes.items():
            node_states[node_name] = {
                "cpu": node_data.get("CPU", 0.0),
                "ram": node_data.get("RAM", 0.0),
                "tx": node_data.get("TX", 0.0),
                "rx": node_data.get("RX", 0.0),
                "avg_latency": node_data.get("avg_latency", 0.0),
                "deployments": node_data.get("deployments", {})
            }
        
        return node_names, node_states
    
    def _calculate_node_load(self, node_state: Dict, deployment_name: str) -> float:
        """
        Calculate current load on a node.
        
        Args:
            node_state: State information for the node
            deployment_name: Name of the deployment
            
        Returns:
            Load value (0.0 to 1.0)
        """
        deployments = node_state.get("deployments", {})
        if deployment_name not in deployments:
            return 0.0
        
        deployment = deployments[deployment_name]
        replicas = deployment.get("Replicas", 0)
        cpu_usage = deployment.get("CPU_usage", 0.0)
        ram_usage = deployment.get("RAM_usage", 0.0)
        
        # Simple load metric: weighted sum of resource usage
        cpu_load = (replicas * cpu_usage) / max(node_state.get("cpu", 1.0), 1.0)
        ram_load = (replicas * ram_usage) / max(node_state.get("ram", 1.0), 1.0)
        
        return (cpu_load + ram_load) / 2.0
    
    def choose_action(self, observation: Dict) -> np.ndarray:
        """
        Choose action based on heuristic strategy.
        
        Args:
            observation: Current state observation from environment
            
        Returns:
            Action array of shape (num_deployments,) with replica counts
        """
        try:
            node_names, node_states = self._extract_state_info(observation)
            num_nodes = len(node_names)
            
            if self.heuristic_type == "uniform":
                return self._uniform_action(num_nodes)
            elif self.heuristic_type == "static":
                return self._static_action()
            elif self.heuristic_type == "greedy_latency":
                return self._greedy_latency_action(observation, node_names, node_states)
            elif self.heuristic_type == "greedy_load":
                return self._greedy_load_action(observation, node_names, node_states)
            else:
                raise ValueError(f"Unknown heuristic type: {self.heuristic_type}")
        except Exception as e:
            logger.error(f"Error in choose_action: {e}")
            # Return safe default action
            return np.array([self.static_replicas] * self.num_deployments, dtype=np.int32)
    
    def _uniform_action(self, num_nodes: int) -> np.ndarray:
        """
        Uniform distribution: distribute replicas evenly.
        For simplicity, uses same replica count for all deployments.
        """
        # Distribute total replicas evenly across deployments
        replicas_per_deployment = max(1, self.max_replicas // self.num_deployments)
        return np.array([replicas_per_deployment] * self.num_deployments, dtype=np.int32)
    
    def _static_action(self) -> np.ndarray:
        """
        Static configuration: fixed replica count.
        """
        return np.array([self.static_replicas] * self.num_deployments, dtype=np.int32)
    
    def _greedy_latency_action(
        self, 
        observation: Dict, 
        node_names: List[str], 
        node_states: Dict[str, Dict]
    ) -> np.ndarray:
        """
        Greedy latency: place replicas on nodes with minimum latency.
        """
        action = np.zeros(self.num_deployments, dtype=np.int32)
        
        # Get deployment names from observation
        if "nodes" in observation and node_names:
            first_node = node_names[0]
            if first_node in observation["nodes"]:
                deployments = observation["nodes"][first_node].get("deployments", {})
                deployment_names = list(deployments.keys())
            else:
                deployment_names = [f"deployment_{i}" for i in range(self.num_deployments)]
        else:
            deployment_names = [f"deployment_{i}" for i in range(self.num_deployments)]
        
        # For each deployment, find node with minimum latency
        for i, deployment_name in enumerate(deployment_names[:self.num_deployments]):
            min_latency = float('inf')
            best_node = None
            
            for node_name in node_names:
                latency = node_states[node_name].get("avg_latency", float('inf'))
                if latency < min_latency:
                    min_latency = latency
                    best_node = node_name
            
            # Place replicas on best node (simplified: just set replica count)
            # In real implementation, would need to track which node has which deployment
            action[i] = min(self.static_replicas, self.max_replicas)
        
        return action
    
    def _greedy_load_action(
        self,
        observation: Dict,
        node_names: List[str],
        node_states: Dict[str, Dict]
    ) -> np.ndarray:
        """
        Greedy load: place replicas on nodes with minimum load.
        """
        action = np.zeros(self.num_deployments, dtype=np.int32)
        
        # Get deployment names from observation
        if "nodes" in observation and node_names:
            first_node = node_names[0]
            if first_node in observation["nodes"]:
                deployments = observation["nodes"][first_node].get("deployments", {})
                deployment_names = list(deployments.keys())
            else:
                deployment_names = [f"deployment_{i}" for i in range(self.num_deployments)]
        else:
            deployment_names = [f"deployment_{i}" for i in range(self.num_deployments)]
        
        # For each deployment, find node with minimum load
        for i, deployment_name in enumerate(deployment_names[:self.num_deployments]):
            min_load = float('inf')
            best_node = None
            
            for node_name in node_names:
                load = self._calculate_node_load(node_states[node_name], deployment_name)
                if load < min_load:
                    min_load = load
                    best_node = node_name
            
            # Place replicas on best node (simplified: just set replica count)
            action[i] = min(self.static_replicas, self.max_replicas)
        
        return action
    
    def train(
        self,
        env: gym.Env,
        total_episodes: int,
        wandb_run_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, List[float]]:
        """
        Run heuristic baseline for specified number of episodes.
        Note: This is not actual training, just evaluation.
        
        Args:
            env: Gymnasium environment
            total_episodes: Number of episodes to run
            wandb_run_id: Optional Weights & Biases run ID for logging
            
        Returns:
            Dictionary with training metrics (compatible with RL agents)
        """
        logger.info(f"Running {self.heuristic_type} heuristic for {total_episodes} episodes")
        
        episode_rewards = []
        episode_lengths = []
        episode_latencies = []
        
        for episode in range(total_episodes):
            observation, info = env.reset()
            episode_reward = 0.0
            episode_length = 0
            episode_latency_sum = 0.0
            episode_latency_count = 0
            
            done = False
            truncated = False
            
            while not (done or truncated):
                # Choose action using heuristic
                action = self.choose_action(observation)
                
                # Step environment
                observation, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                # Collect latency if available
                if "latency" in info:
                    episode_latency_sum += info["latency"]
                    episode_latency_count += 1
            
            avg_latency = episode_latency_sum / episode_latency_count if episode_latency_count > 0 else 0.0
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_latencies.append(avg_latency)
            
            if (episode + 1) % 10 == 0:
                logger.info(
                    f"Episode {episode + 1}/{total_episodes}: "
                    f"reward={episode_reward:.2f}, "
                    f"length={episode_length}, "
                    f"avg_latency={avg_latency:.2f}ms"
                )
        
        # Return results in format compatible with RL agents
        results = {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "episode_latencies": episode_latencies,
            "mean_rewards": [np.mean(episode_rewards[:i+1]) for i in range(len(episode_rewards))],
            "mean_lengths": [np.mean(episode_lengths[:i+1]) for i in range(len(episode_lengths))],
            # Dummy values for compatibility with PPO/TD3/SAC format
            "actor_losses": [0.0] * total_episodes,
            "critic_losses": [0.0] * total_episodes,
            "total_losses": [0.0] * total_episodes,
        }
        
        logger.info(
            f"Heuristic baseline completed. "
            f"Mean reward: {np.mean(episode_rewards):.2f}, "
            f"Mean latency: {np.mean(episode_latencies):.2f}ms"
        )
        
        return results


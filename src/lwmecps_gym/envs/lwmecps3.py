import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple, Union
import logging
from lwmecps_gym.envs.testapp_api import start_experiment_group, get_metrics


class LWMECPSEnv3(gym.Env):
    """LWMECPS Environment with test application API integration."""
    
    def __init__(
        self,
        node_name: str,
        max_hardware: Dict[str, float],
        pod_usage: Dict[str, float],
        node_info: Dict[str, Dict[str, float]],
        num_nodes: int,
        namespace: str,
        deployment_name: str,
        deployments: List[str],
        max_pods: int,
        group_id: str,
        base_url: str = "http://localhost:8001"
    ):
        super().__init__()
        
        self.node_name = node_name
        self.max_hardware = max_hardware
        self.pod_usage = pod_usage
        self.node_info = node_info
        self.num_nodes = num_nodes
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.deployments = deployments
        self.max_pods = max_pods
        self.group_id = group_id
        self.base_url = base_url
        
        # Action space: 0 = scale down, 1 = no change, 2 = scale up
        self.action_space = gym.spaces.Discrete(3)
        
        # Observation space
        self.observation_space = gym.spaces.Dict({
            "cpu": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "ram": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "bandwidth_in": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "bandwidth_out": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "avg_latency": gym.spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            "concurrent_users": gym.spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            "deployment_replicas": gym.spaces.Box(low=0, high=max_pods, shape=(1,), dtype=np.int32)
        })
        
        # Initialize state variables
        self.current_replicas = 1
        self.state = {
            "cpu": np.array([0.0], dtype=np.float32),
            "ram": np.array([0.0], dtype=np.float32),
            "bandwidth_in": np.array([0.0], dtype=np.float32),
            "bandwidth_out": np.array([0.0], dtype=np.float32),
            "avg_latency": np.array([0.0], dtype=np.float32),
            "concurrent_users": np.array([0], dtype=np.float32),
            "deployment_replicas": np.array([1], dtype=np.int32)
        }
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        try:
            # Start the experiment group
            start_experiment_group(self.group_id, self.base_url)
            
            # Get initial metrics
            metrics = get_metrics(self.group_id, self.base_url)
            node_metrics = metrics.get(self.node_name, {})
            
            # Update state with initial metrics
            self.state.update({
                "avg_latency": np.array([node_metrics.get("avg_latency", 0.0)], dtype=np.float32),
                "concurrent_users": np.array([node_metrics.get("concurrent_users", 0)], dtype=np.float32)
            })
            
            self.current_replicas = 1
            self.state["deployment_replicas"] = np.array([self.current_replicas], dtype=np.int32)
            
            return self.state, {}
            
        except Exception as e:
            self.logger.error(f"Failed to reset environment: {str(e)}")
            raise
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one time step within the environment."""
        try:
            # Execute action
            if action == 0 and self.current_replicas > 1:  # Scale down
                self.current_replicas -= 1
            elif action == 2 and self.current_replicas < self.max_pods:  # Scale up
                self.current_replicas += 1
            
            self.state["deployment_replicas"] = np.array([self.current_replicas], dtype=np.int32)
            
            # Get updated metrics
            metrics = get_metrics(self.group_id, self.base_url)
            node_metrics = metrics.get(self.node_name, {})
            
            # Update state with new metrics
            self.state.update({
                "avg_latency": np.array([node_metrics.get("avg_latency", 0.0)], dtype=np.float32),
                "concurrent_users": np.array([node_metrics.get("concurrent_users", 0)], dtype=np.float32)
            })
            
            # Calculate reward
            reward = self._calculate_reward()
            
            # Check if episode is done
            done = False
            
            return self.state, reward, done, False, {}
            
        except Exception as e:
            self.logger.error(f"Failed to execute step: {str(e)}")
            raise
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on current state."""
        latency = self.state["avg_latency"][0]
        users = self.state["concurrent_users"][0]
        replicas = self.state["deployment_replicas"][0]
        
        # Reward for low latency and high user count
        reward = (1000 / (latency + 1)) * (users / 100)
        
        # Penalty for using too many replicas
        reward -= replicas * 10
        
        return float(reward)
    
    def close(self) -> None:
        """Clean up the environment."""
        pass 
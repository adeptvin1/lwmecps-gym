import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple, Union
import logging
from lwmecps_gym.envs.testapp_api import start_experiment_group, get_metrics
from lwmecps_gym.envs.kubernetes_api import k8s


class LWMECPSEnv3(gym.Env):
    """LWMECPS Environment with test application API integration and Q-learning support."""
    
    def __init__(
        self,
        node_name: List[str],  # Changed to List[str] to support multiple nodes
        max_hardware: Dict[str, float],
        pod_usage: Dict[str, float],
        node_info: Dict[str, Dict[str, float]],
        num_nodes: int,
        namespace: str,
        deployment_name: str,
        deployments: List[str],
        max_pods: int,
        group_id: str,
        base_url: str = "http://localhost:8001",
        env_config: Dict[str, Any] = None
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
        self.base_url = env_config.get("base_url", base_url) if env_config else base_url
        
        # Initialize Kubernetes client
        self.minikube = k8s()
        
        # Action space: each action corresponds to moving pod to a specific node
        self.action_space = gym.spaces.Discrete(num_nodes)
        
        # Observation space for each node - simplified for Q-learning
        self.observation_space = gym.spaces.Dict({
            node: gym.spaces.Dict({
                "deployments": gym.spaces.Dict({
                    deployment: gym.spaces.Dict({
                        "replicas": gym.spaces.Box(low=0, high=max_pods, shape=(), dtype=np.float32)
                    }) for deployment in deployments
                }),
                "avg_latency": gym.spaces.Box(low=0, high=float('inf'), shape=(), dtype=np.float32)
            }) for node in node_name
        })
        
        # Initialize state
        self.state = None
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        try:
            # Start the experiment group
            start_experiment_group(self.group_id, self.base_url)
            
            # Initialize state for each node - simplified for Q-learning
            self.state = {
                node: {
                    "deployments": {
                        deployment: {"replicas": 0} for deployment in self.deployments
                    },
                    "avg_latency": self.node_info[node]["avg_latency"]
                } for node in self.node_name
            }
            
            # Place initial pod on the first node
            self.minikube.k8s_action(
                namespace=self.namespace,
                deployment_name=self.deployment_name,
                replicas=1,
                node=self.node_name[0]
            )
            
            # Update state with initial metrics
            metrics = get_metrics(self.group_id, self.base_url)
            for node in self.node_name:
                node_metrics = metrics.get(node, {})
                if node_metrics:
                    self.state[node]["avg_latency"] = node_metrics.get("avg_latency", self.node_info[node]["avg_latency"])
            
            return self.state, {}
            
        except Exception as e:
            self.logger.error(f"Failed to reset environment: {str(e)}")
            raise
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one time step within the environment."""
        try:
            # Move pod to the selected node
            target_node = self.node_name[action]
            self.minikube.k8s_action(
                namespace=self.namespace,
                deployment_name=self.deployment_name,
                replicas=1,
                node=target_node
            )
            
            # Get updated metrics
            metrics = get_metrics(self.group_id, self.base_url)
            
            # Update state with new metrics
            for node in self.node_name:
                node_metrics = metrics.get(node, {})
                if node_metrics:
                    latency = node_metrics.get("avg_latency", self.node_info[node]["avg_latency"])
                    self.state[node]["avg_latency"] = latency
                    self.logger.info(f"Node {node} latency: {latency}ms")
            
            # Calculate reward
            reward = self._calculate_reward(target_node)
            
            # Check if episode is done
            done = False
            
            return self.state, reward, done, False, {}
            
        except Exception as e:
            self.logger.error(f"Failed to execute step: {str(e)}")
            raise
    
    def _calculate_reward(self, target_node: str) -> float:
        """Calculate reward based on current state and target node."""
        latency = self.state[target_node]["avg_latency"]
        
        # Base reward is inverse to latency
        reward = 1000 / (latency + 1)
        
        return float(reward)
    
    def close(self) -> None:
        """Clean up the environment."""
        pass 
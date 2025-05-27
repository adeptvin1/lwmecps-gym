import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple, Union, Any
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
            "current_node": gym.spaces.Text(max_length=50),  # Add current_node to observation space
            "nodes": gym.spaces.Dict({
                node: gym.spaces.Dict({
                    "deployments": gym.spaces.Dict({
                        deployment: gym.spaces.Dict({
                            "replicas": gym.spaces.Box(low=0, high=max_pods, shape=(), dtype=np.float32)
                        }) for deployment in deployments
                    }),
                    "avg_latency": gym.spaces.Box(low=0, high=float('inf'), shape=(), dtype=np.float32)
                }) for node in node_name
            })
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
            # Validate node_name
            if not self.node_name:
                raise ValueError("node_name list cannot be empty")
            
            # Start the experiment group
            self.logger.info(f"Starting experiment group {self.group_id}")
            start_experiment_group(self.group_id, self.base_url)
            self.logger.info(f"Successfully started experiment group {self.group_id}")
            
            # Initialize state for each node - simplified for Q-learning
            self.logger.info("Initializing state for each node")
            self.state = {
                "current_node": self.node_name[0],  # Track current node
                "nodes": {
                    node: {
                        "deployments": {
                            deployment: {"replicas": 0} for deployment in self.deployments
                        },
                        "avg_latency": self.node_info[node]["avg_latency"]
                    } for node in self.node_name
                }
            }
            self.logger.info(f"Initial state: {self.state}")
            
            # Place initial pod on the first node
            initial_node = self.node_name[0]
            self.logger.info(f"Placing initial pod on node {initial_node}")
            self.minikube.k8s_action(
                namespace=self.namespace,
                deployment_name=self.deployment_name,
                replicas=1,
                node=initial_node
            )
            self.logger.info(f"Successfully placed initial pod on node {initial_node}")
            
            # Update state with initial metrics
            self.logger.info("Getting initial metrics")
            metrics = get_metrics(self.group_id, self.base_url)
            self.logger.info(f"Received initial metrics: {metrics}")
            
            # Use group metrics for all nodes
            if 'group' in metrics:
                group_metrics = metrics['group']
                latency = group_metrics.get('avg_latency', 0.0)
                if latency < 0:
                    self.logger.warning(f"Received negative latency: {latency}, using 0.0")
                    latency = 0.0
                for node in self.node_name:
                    self.state["nodes"][node]["avg_latency"] = latency
                    self.logger.info(f"Node {node} initial latency: {latency}ms")
            else:
                self.logger.warning("No group metrics found, using default latencies")
            
            # Set initial node's replicas to 1
            for deployment in self.deployments:
                self.state["nodes"][initial_node]["deployments"][deployment]["replicas"] = 1
            
            return self.state, {}
            
        except Exception as e:
            self.logger.error(f"Failed to reset environment: {str(e)}")
            raise
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one time step within the environment."""
        try:
            # Validate action
            if not 0 <= action < self.num_nodes:
                raise ValueError(f"Invalid action {action}. Must be between 0 and {self.num_nodes-1}")
            
            # Move pod to the selected node
            target_node = self.node_name[action]
            self.logger.info(f"Moving pod to node {target_node}")
            
            # Update state: set all nodes to 0 replicas
            for node in self.node_name:
                for deployment in self.deployments:
                    self.state["nodes"][node]["deployments"][deployment]["replicas"] = 0
            
            # Move pod to target node
            self.minikube.k8s_action(
                namespace=self.namespace,
                deployment_name=self.deployment_name,
                replicas=1,
                node=target_node
            )
            self.logger.info(f"Successfully moved pod to node {target_node}")
            
            # Update state: set target node to 1 replica and update current_node
            for deployment in self.deployments:
                self.state["nodes"][target_node]["deployments"][deployment]["replicas"] = 1
            self.state["current_node"] = target_node
            
            # Get updated metrics
            self.logger.info(f"Getting metrics for group {self.group_id}")
            metrics = get_metrics(self.group_id, self.base_url)
            self.logger.info(f"Received metrics: {metrics}")
            
            # Update state with group metrics
            if 'group' in metrics:
                group_metrics = metrics['group']
                latency = group_metrics.get('avg_latency', 0.0)
                if latency < 0:
                    self.logger.warning(f"Received negative latency: {latency}, using 0.0")
                    latency = 0.0
                for node in self.node_name:
                    self.state["nodes"][node]["avg_latency"] = latency
                    self.logger.info(f"Node {node} latency: {latency}ms")
            else:
                self.logger.warning("No group metrics found, using default latencies")
            
            # Calculate reward
            reward = self._calculate_reward(target_node)
            self.logger.info(f"Calculated reward for node {target_node}: {reward}")
            
            # Check if episode is done
            done = False
            
            return self.state, reward, done, False, {}
            
        except Exception as e:
            self.logger.error(f"Failed to execute step: {str(e)}")
            raise
    
    def _calculate_reward(self, target_node: str) -> float:
        """Calculate reward based on current state and target node."""
        latency = self.state["nodes"][target_node]["avg_latency"]
        
        # Ensure latency is non-negative
        if latency < 0:
            self.logger.warning(f"Negative latency detected: {latency}, using 0.0")
            latency = 0.0
        
        # Base reward is inverse to latency
        reward = 1000 / (latency + 1)
        
        return float(reward)
    
    def close(self) -> None:
        """Clean up the environment."""
        pass 
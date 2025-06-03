import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple, Union, Any
import logging
import time
import re
import bitmath
from lwmecps_gym.envs.testapp_api import start_experiment_group, get_metrics
from lwmecps_gym.envs.kubernetes_api import k8s

# Resource metrics constants
NODE_NETWORK_SPEED = 1024  # 1Gb/s in Mb/s
POD_RESOURCE_USAGE = {
    "CPU_usage": 0.1,  # 10% of CPU core
    "RAM_usage": 128,  # 128MB
    "TX_usage": 10,    # 10Mb/s
    "RX_usage": 10     # 10Mb/s
}

class LWMECPSEnv3(gym.Env):
    """LWMECPS Environment with test application API integration and Q-learning support."""
    
    def __init__(
        self,
        node_name: List[str],
        max_hardware: Dict[str, float],
        pod_usage: Dict[str, float],
        node_info: Dict[str, Dict[str, float]],
        num_nodes: int,
        namespace: str,
        deployments: List[str],
        max_pods: int,
        group_id: str,
        base_url: str = "http://localhost:8001",
        env_config: Dict[str, Any] = None,
        stabilization_time: int = 10
    ):
        super().__init__()
        self.node_name = node_name
        self.max_hardware = max_hardware
        self.pod_usage = pod_usage
        self.node_info = node_info
        self.num_nodes = num_nodes
        self.namespace = namespace
        self.deployments = deployments
        self.max_pods = max_pods
        self.group_id = group_id
        self.base_url = env_config.get("base_url", base_url) if env_config else base_url
        self.stabilization_time = int(env_config.get("stabilization_time", stabilization_time)) if env_config else stabilization_time
        self.minikube = k8s()
        self.max_replicas = int(self.max_hardware["cpu"] / self.pod_usage["cpu"])
        self.action_space = gym.spaces.MultiDiscrete(
            [self.max_replicas + 1] * len(deployments)
        )
        self.observation_space = gym.spaces.Dict({
            "nodes": gym.spaces.Dict({
                node: gym.spaces.Dict({
                    "CPU": gym.spaces.Box(low=0, high=float('inf'), shape=(), dtype=np.float32),
                    "RAM": gym.spaces.Box(low=0, high=float('inf'), shape=(), dtype=np.float32),
                    "TX": gym.spaces.Box(low=0, high=NODE_NETWORK_SPEED, shape=(), dtype=np.float32),
                    "RX": gym.spaces.Box(low=0, high=NODE_NETWORK_SPEED, shape=(), dtype=np.float32),
                    "deployments": gym.spaces.Dict({
                        deployment: gym.spaces.Dict({
                            "CPU_usage": gym.spaces.Box(low=0, high=float('inf'), shape=(), dtype=np.float32),
                            "RAM_usage": gym.spaces.Box(low=0, high=float('inf'), shape=(), dtype=np.float32),
                            "TX_usage": gym.spaces.Box(low=0, high=float('inf'), shape=(), dtype=np.float32),
                            "RX_usage": gym.spaces.Box(low=0, high=float('inf'), shape=(), dtype=np.float32),
                            "Replicas": gym.spaces.Box(low=0, high=max_pods, shape=(), dtype=np.int32)
                        }) for deployment in deployments
                    }),
                    "avg_latency": gym.spaces.Box(low=0, high=float('inf'), shape=(), dtype=np.float32)
                }) for node in node_name
            })
        })
        self.state = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
    def _parse_memory(self, memory_str: str) -> float:
        """
        Parse memory string from Kubernetes format to MB.
        
        Args:
            memory_str (str): Memory string in Kubernetes format (e.g., "1Ki", "1Mi", "1Gi")
            
        Returns:
            float: Memory in MB
        """
        try:
            # Extract number and unit
            match = re.match(r"(\d+)([KMG]i)", memory_str)
            if not match:
                raise ValueError(f"Invalid memory format: {memory_str}")
                
            number = int(match.group(1))
            unit = match.group(2)
            
            # Convert to MB
            if unit == "Ki":
                return round(bitmath.KiB(number).to_MB().value)
            elif unit == "Mi":
                return number
            elif unit == "Gi":
                return round(bitmath.GiB(number).to_MB().value)
            else:
                raise ValueError(f"Unknown memory unit: {unit}")
        except Exception as e:
            self.logger.error(f"Failed to parse memory string '{memory_str}': {str(e)}")
            raise

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
            
            # Get initial Kubernetes state
            k8s_state = self.minikube.k8s_state()
            if not k8s_state:
                raise ValueError("Failed to get initial Kubernetes state")
            
            # Initialize state with resource metrics
            self.state = {
                "nodes": {
                    node: {
                        "CPU": float(k8s_state[node]['cpu']),
                        "RAM": self._parse_memory(k8s_state[node]['memory']),
                        "TX": NODE_NETWORK_SPEED,
                        "RX": NODE_NETWORK_SPEED,
                        "deployments": {
                            deployment: {
                                "CPU_usage": POD_RESOURCE_USAGE["CPU_usage"],
                                "RAM_usage": POD_RESOURCE_USAGE["RAM_usage"],
                                "TX_usage": POD_RESOURCE_USAGE["TX_usage"],
                                "RX_usage": POD_RESOURCE_USAGE["RX_usage"],
                                "Replicas": 0
                            } for deployment in self.deployments
                        },
                        "avg_latency": float(self.node_info[node]["avg_latency"])
                    } for node in self.node_name
                }
            }
            
            # Place initial pod
            self.logger.info(f"Setting initial replicas to 1 for {self.deployments[0]}")
            self.minikube.k8s_action(
                namespace=self.namespace,
                deployment_name=self.deployments[0],  # Use first deployment
                replicas=1
            )
            
            # Update state with initial metrics
            metrics = get_metrics(self.group_id, self.base_url)
            if 'group' in metrics:
                group_metrics = metrics['group']
                latency = group_metrics.get('avg_latency', 0.0)
                if latency < 0:
                    self.logger.warning(f"Received negative latency: {latency}, using 0.0")
                    latency = 0.0
                for node in self.node_name:
                    self.state["nodes"][node]["avg_latency"] = float(latency)
            
            # Set initial replicas to 1 for first deployment
            for node in self.node_name:
                self.state["nodes"][node]["deployments"][self.deployments[0]]["Replicas"] = 1
            
            return self.state, {}
            
        except Exception as e:
            self.logger.error(f"Failed to reset environment: {str(e)}")
            raise
    
    def _validate_node_resources(self, node: str, deployment: str, replicas: int) -> bool:
        """
        Validate if node has enough resources for the requested number of replicas.
        
        Args:
            node (str): Node name
            deployment (str): Deployment name
            replicas (int): Number of replicas to add
            
        Returns:
            bool: True if node has enough resources, False otherwise
        """
        # Get current resource usage on the node
        current_usage = {
            "CPU": 0.0,
            "RAM": 0.0,
            "TX": 0.0,
            "RX": 0.0
        }
        
        # Calculate current usage from existing replicas
        for dep in self.deployments:
            current_replicas = self.state["nodes"][node]["deployments"][dep]["Replicas"]
            current_usage["CPU"] += current_replicas * self.state["nodes"][node]["deployments"][dep]["CPU_usage"]
            current_usage["RAM"] += current_replicas * self.state["nodes"][node]["deployments"][dep]["RAM_usage"]
            current_usage["TX"] += current_replicas * self.state["nodes"][node]["deployments"][dep]["TX_usage"]
            current_usage["RX"] += current_replicas * self.state["nodes"][node]["deployments"][dep]["RX_usage"]
        
        # Add usage from new replicas
        current_usage["CPU"] += replicas * self.state["nodes"][node]["deployments"][deployment]["CPU_usage"]
        current_usage["RAM"] += replicas * self.state["nodes"][node]["deployments"][deployment]["RAM_usage"]
        current_usage["TX"] += replicas * self.state["nodes"][node]["deployments"][deployment]["TX_usage"]
        current_usage["RX"] += replicas * self.state["nodes"][node]["deployments"][deployment]["RX_usage"]
        
        # Check against node limits
        if current_usage["CPU"] > self.state["nodes"][node]["CPU"]:
            self.logger.warning(f"CPU limit exceeded on node {node}: {current_usage['CPU']} > {self.state['nodes'][node]['CPU']}")
            return False
            
        if current_usage["RAM"] > self.state["nodes"][node]["RAM"]:
            self.logger.warning(f"RAM limit exceeded on node {node}: {current_usage['RAM']} > {self.state['nodes'][node]['RAM']}")
            return False
            
        if current_usage["TX"] > self.state["nodes"][node]["TX"]:
            self.logger.warning(f"TX limit exceeded on node {node}: {current_usage['TX']} > {self.state['nodes'][node]['TX']}")
            return False
            
        if current_usage["RX"] > self.state["nodes"][node]["RX"]:
            self.logger.warning(f"RX limit exceeded on node {node}: {current_usage['RX']} > {self.state['nodes'][node]['RX']}")
            return False
            
        return True

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one time step within the environment."""
        try:
            # Validate action
            if not isinstance(action, np.ndarray) or action.shape != (len(self.deployments),):
                raise ValueError(f"Invalid action shape. Expected shape ({len(self.deployments)},), got {action.shape if isinstance(action, np.ndarray) else type(action)}")
            
            if not all(0 <= a <= self.max_replicas for a in action):
                raise ValueError(f"Invalid action values. All values must be between 0 and {self.max_replicas}")
            
            # Update replicas for each deployment
            for i, deployment in enumerate(self.deployments):
                replicas = int(action[i])
                self.logger.info(f"Setting {replicas} replicas for deployment {deployment}")
                
                # Update state: set all nodes to 0 replicas for this deployment
                for node in self.node_name:
                    self.state["nodes"][node]["deployments"][deployment]["Replicas"] = 0
                
                # If replicas > 0, validate resources and place them
                if replicas > 0:
                    # Validate resources before applying changes
                    if not self._validate_node_resources(self.node_name[0], deployment, replicas):
                        self.logger.warning(f"Not enough resources for {replicas} replicas of {deployment}")
                        # Return negative reward for invalid action
                        return self.state, -100.0, False, False, {"error": "Not enough resources"}
                    
                    self.minikube.k8s_action(
                        namespace=self.namespace,
                        deployment_name=deployment,
                        replicas=replicas
                    )
                    # Update state for all nodes
                    for node in self.node_name:
                        self.state["nodes"][node]["deployments"][deployment]["Replicas"] = replicas
            
            # Wait for system stabilization
            self.logger.info(f"Waiting {self.stabilization_time} seconds for system stabilization...")
            time.sleep(self.stabilization_time)
            
            # Get updated metrics
            metrics = get_metrics(self.group_id, self.base_url)
            if 'group' in metrics:
                group_metrics = metrics['group']
                latency = group_metrics.get('avg_latency', 0.0)
                if latency < 0:
                    self.logger.warning(f"Received negative latency: {latency}, using 0.0")
                    latency = 0.0
                for node in self.node_name:
                    self.state["nodes"][node]["avg_latency"] = float(latency)
            
            # Calculate reward
            reward = self._calculate_reward()
            self.logger.info(f"Calculated reward: {reward}")
            
            # Check if episode is done
            done = False
            
            return self.state, reward, done, False, {}
            
        except Exception as e:
            self.logger.error(f"Failed to execute step: {str(e)}")
            raise
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on current state."""
        # Calculate total replicas
        total_replicas = sum(
            self.state["nodes"][node]["deployments"][deployment]["Replicas"]
            for node in self.node_name
            for deployment in self.deployments
        )
        
        # Calculate average latency
        avg_latency = sum(
            self.state["nodes"][node]["avg_latency"]
            for node in self.node_name
        ) / len(self.node_name)
        
        # Calculate resource imbalances
        resource_types = ["CPU", "RAM", "TX", "RX"]
        total_imbalance = 0.0
        
        for resource in resource_types:
            # Get resource usage values for each node
            values = []
            for node in self.node_name:
                node_usage = 0.0
                for deployment in self.deployments:
                    replicas = self.state["nodes"][node]["deployments"][deployment]["Replicas"]
                    if resource == "CPU":
                        node_usage += replicas * self.state["nodes"][node]["deployments"][deployment]["CPU_usage"]
                    elif resource == "RAM":
                        node_usage += replicas * self.state["nodes"][node]["deployments"][deployment]["RAM_usage"]
                    elif resource == "TX":
                        node_usage += replicas * self.state["nodes"][node]["deployments"][deployment]["TX_usage"]
                    elif resource == "RX":
                        node_usage += replicas * self.state["nodes"][node]["deployments"][deployment]["RX_usage"]
                values.append(node_usage)
            
            # Calculate imbalance for this resource
            mean = sum(values) / len(values)
            imbalance = sum((x - mean) ** 2 for x in values) / len(values)
            total_imbalance += imbalance
        
        # Calculate final reward
        alpha = 1.0  # Weight for latency
        beta = 0.1   # Weight for total replicas
        gamma = 0.1  # Weight for imbalance
        
        reward = alpha * avg_latency - beta * total_replicas - gamma * total_imbalance
        
        return float(reward)
    
    def close(self) -> None:
        """Clean up the environment."""
        pass 
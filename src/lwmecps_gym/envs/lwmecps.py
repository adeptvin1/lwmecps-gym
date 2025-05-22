import re
from time import sleep
import random

import bitmath
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from lwmecps_gym.envs.kubernetes_api import k8s
import logging

logger = logging.getLogger(__name__)


class LWMECPSEnv(gym.Env):

    def __init__(
        self,
        node_name,
        max_hardware,
        pod_usage,
        node_info,
        num_nodes,
        namespace,
        deployment_name,
        deployments,
        max_pods,
    ):
        super(LWMECPSEnv, self).__init__()
        logger.info("[LWMECPSEnv.__init__] Initializing environment")
        self.num_nodes = num_nodes
        self.node_name = node_name
        self.max_hardware = max_hardware
        self.pod_usage = pod_usage
        self.node_info = node_info
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.deployments = deployments
        self.max_pods = max_pods

        logger.info("[LWMECPSEnv.__init__] Initializing Kubernetes client")
        self.minikube = k8s()

        self.prev_latency = None
        self.current_latency = None
        self.rew_amm = 0
        # self.render_mode = render_mode
        # self.window_size = window_size

        # Define action space
        # Action space: scale up/down pods (discrete actions)
        self.action_space = spaces.Discrete(3)  # 0: scale down, 1: no change, 2: scale up

        logger.info("[LWMECPSEnv.__init__] Setting up observation space")
        self.observation_space = spaces.Dict(
            {
                node: spaces.Dict(
                    {
                        "cpu": spaces.Box(
                            low=0,
                            high=self.max_hardware["cpu"],
                            shape=(),
                            dtype=np.float32,
                        ),
                        "ram": spaces.Box(
                            low=0,
                            high=self.max_hardware["ram"],
                            shape=(),
                            dtype=np.float32,
                        ),
                        "tx_bandwidth": spaces.Box(
                            low=0,
                            high=self.max_hardware["tx_bandwidth"],
                            shape=(),
                            dtype=np.float32,
                        ),
                        "rx_bandwidth": spaces.Box(
                            low=0,
                            high=self.max_hardware["rx_bandwidth"],
                            shape=(),
                            dtype=np.float32,
                        ),
                        "read_disks_bandwidth": spaces.Box(
                            low=0,
                            high=self.max_hardware["read_disks_bandwidth"],
                            shape=(),
                            dtype=np.float32,
                        ),
                        "write_disks_bandwidth": spaces.Box(
                            low=0,
                            high=self.max_hardware["write_disks_bandwidth"],
                            shape=(),
                            dtype=np.float32,
                        ),
                        "avg_latency": spaces.Box(
                            low=0,
                            high=self.max_hardware["avg_latency"],
                            shape=(),
                            dtype=np.float32,
                        ),
                        "deployments": spaces.Dict(
                            {
                                deployment: spaces.Dict(
                                    {
                                        "replicas": spaces.Box(
                                            low=0,
                                            high=self.max_pods,
                                            shape=(),
                                            dtype=np.float32,
                                        )
                                    }
                                )
                                for deployment in self.deployments
                            }
                        ),
                    }
                )
                for node in self.node_name
            }
        )
        self.state = None
        logger.info("[LWMECPSEnv.__init__] Environment initialization completed")
        self.reset()

    def _parse_memory(self, memory_str):
        """
        Parse memory string from Kubernetes format to MB.
        
        Args:
            memory_str (str): Memory string in Kubernetes format (e.g., "1Ki", "1Mi", "1Gi")
            
        Returns:
            int: Memory in MB
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
            logger.error(f"Failed to parse memory string '{memory_str}': {str(e)}")
            raise

    def reset(self, seed=None, options=None):
        """
        Сброс состояния среды.
        """
        try:
            # Получаем информацию о нодах
            node_info = self.minikube.k8s_state()
            if not node_info:
                raise Exception("Не удалось получить информацию о нодах")

            # Инициализируем состояние
            self.state = {}
            for node in node_info:
                # Получаем информацию о ресурсах ноды
                cpu = int(node_info[node]["cpu"])
                memory = node_info[node]["memory"]
                
                # Конвертируем память из Kubernetes формата (Ki) в MB
                memory_mb = self._parse_memory(memory)
                
                # Инициализируем состояние для ноды
                self.state[node] = {
                    "cpu": cpu,
                    "ram": memory_mb,
                    "tx_bandwidth": int(node_info[node].get("tx_bandwidth", 0)),
                    "rx_bandwidth": int(node_info[node].get("rx_bandwidth", 0)),
                    "read_disks_bandwidth": int(node_info[node].get("read_disks_bandwidth", 0)),
                    "write_disks_bandwidth": int(node_info[node].get("write_disks_bandwidth", 0)),
                    "avg_latency": float(node_info[node].get("avg_latency", 0.0)),
                    "deployments": {},
                }
                
                # Добавляем информацию о деплойментах
                if "deployments" in node_info[node]:
                    for namespace, deployments in node_info[node]["deployments"].items():
                        if namespace not in self.state[node]["deployments"]:
                            self.state[node]["deployments"][namespace] = {}
                        for deployment, info in deployments.items():
                            self.state[node]["deployments"][namespace][deployment] = {
                                "replicas": info.get("replicas", 0),
                                "cpu_request": info.get("cpu_request", 0),
                                "memory_request": info.get("memory_request", 0)
                            }

            # Устанавливаем начальное состояние
            self.current_step = 0
            self.total_reward = 0
            self.done = False
            self.prev_latency = None
            self.current_latency = None
            self.rew_amm = 0

            # Если указан seed, устанавливаем его
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)

            # Преобразуем состояние в формат, соответствующий observation_space
            observation = {}
            for node in self.node_name:
                if node in self.state:
                    node_state = self.state[node]
                    observation[node] = {
                        "cpu": node_state["cpu"],
                        "ram": node_state["ram"],
                        "tx_bandwidth": node_state["tx_bandwidth"],
                        "rx_bandwidth": node_state["rx_bandwidth"],
                        "read_disks_bandwidth": node_state["read_disks_bandwidth"],
                        "write_disks_bandwidth": node_state["write_disks_bandwidth"],
                        "avg_latency": node_state["avg_latency"],
                        "deployments": {
                            deployment: {
                                "replicas": node_state["deployments"].get(self.namespace, {}).get(deployment, {}).get("replicas", 0)
                            }
                            for deployment in self.deployments
                        }
                    }
                else:
                    # Если нода не найдена, создаем пустое состояние
                    observation[node] = {
                        "cpu": 0,
                        "ram": 0,
                        "tx_bandwidth": 0,
                        "rx_bandwidth": 0,
                        "read_disks_bandwidth": 0,
                        "write_disks_bandwidth": 0,
                        "avg_latency": 0.0,
                        "deployments": {
                            deployment: {"replicas": 0}
                            for deployment in self.deployments
                        }
                    }

            return observation, {}  # Возвращаем состояние и пустой словарь info согласно новому API
        except Exception as e:
            logger.error(f"Ошибка при сбросе состояния: {str(e)}")
            raise

    def step(self, action):
        logger.info(f"[LWMECPSEnv.step] Starting step with action {action}")
        assert self.action_space.contains(action), "Invalid action"
        
        # Move pod to the new node
        pod_node = self.node_name[action]
        try:
            logger.info(f"[LWMECPSEnv.step] Moving deployment to node {pod_node}")
            self.minikube.k8s_action(
                namespace=self.namespace,
                deployment_name=self.deployment_name,
                replicas=1,
                node=pod_node,
            )
            logger.info(f"[LWMECPSEnv.step] Successfully moved deployment to node {pod_node}")
        except Exception as e:
            logger.error(f"[LWMECPSEnv.step] Failed to move deployment to node {pod_node}: {str(e)}")
            return self.state, -1000, True, {"error": str(e)}
        
        # Wait for deployment to stabilize with exponential backoff
        logger.info("[LWMECPSEnv.step] Waiting for deployment to stabilize")
        max_wait = 40  # Maximum wait time in seconds
        wait_time = 5  # Initial wait time
        while wait_time <= max_wait:
            try:
                # Check if deployment is ready
                deployment = self.minikube.app_api.read_namespaced_deployment(
                    name=self.deployment_name,
                    namespace=self.namespace
                )
                if deployment.status.ready_replicas == deployment.spec.replicas:
                    break
                logger.info(f"[LWMECPSEnv.step] Deployment not ready yet, waiting {wait_time} seconds")
                sleep(wait_time)
                wait_time *= 2  # Exponential backoff
            except Exception as e:
                logger.error(f"[LWMECPSEnv.step] Error checking deployment status: {str(e)}")
                sleep(wait_time)
                wait_time *= 2
        
        # Update state
        logger.info("[LWMECPSEnv.step] Updating environment state")
        self.state = self._get_k8s_state()
        logger.info("[LWMECPSEnv.step] State updated successfully")
        
        # Calculate reward and check if episode should end
        logger.info("[LWMECPSEnv.step] Calculating reward")
        reward, done = self.reward()
        
        # Check if latency increased
        if (self.prev_latency is not None) and (self.current_latency >= self.prev_latency):
            logger.warning(f"[LWMECPSEnv.step] Latency increased from {self.prev_latency} to {self.current_latency}")
            done = True
            reward -= 100  # Negative reward for degradation
            
        # Update previous latency
        self.prev_latency = self.current_latency
        
        # Преобразуем состояние в формат, соответствующий observation_space
        observation = {}
        for node in self.node_name:
            if node in self.state:
                node_state = self.state[node]
                observation[node] = {
                    "cpu": node_state["cpu"],
                    "ram": node_state["ram"],
                    "tx_bandwidth": node_state["tx_bandwidth"],
                    "rx_bandwidth": node_state["rx_bandwidth"],
                    "read_disks_bandwidth": node_state["read_disks_bandwidth"],
                    "write_disks_bandwidth": node_state["write_disks_bandwidth"],
                    "avg_latency": node_state["avg_latency"],
                    "deployments": {
                        deployment: {
                            "replicas": node_state["deployments"].get(self.namespace, {}).get(deployment, {}).get("replicas", 0)
                        }
                        for deployment in self.deployments
                    }
                }
            else:
                # Если нода не найдена, создаем пустое состояние
                observation[node] = {
                    "cpu": 0,
                    "ram": 0,
                    "tx_bandwidth": 0,
                    "rx_bandwidth": 0,
                    "read_disks_bandwidth": 0,
                    "write_disks_bandwidth": 0,
                    "avg_latency": 0.0,
                    "deployments": {
                        deployment: {"replicas": 0}
                        for deployment in self.deployments
                    }
                }
        
        info = {"latency": self.current_latency}
        logger.info(f"[LWMECPSEnv.step] Step completed. Reward: {reward}, Done: {done}")
        return observation, reward, done, info

    def _get_k8s_state(self):
        """
        Get current state from Kubernetes and convert it to gym format.
        """
        logger.info("[LWMECPSEnv._get_k8s_state] Fetching Kubernetes state")
        k8s_state_now = self.minikube.k8s_state()
        
        state = {}
        for node in self.node_name:
            try:
                state[node] = {
                    "cpu": int(k8s_state_now[node]["cpu"]),
                    "ram": self._parse_memory(k8s_state_now[node]["memory"]),
                    "tx_bandwidth": int(k8s_state_now[node].get("tx_bandwidth", 0)),
                    "rx_bandwidth": int(k8s_state_now[node].get("rx_bandwidth", 0)),
                    "read_disks_bandwidth": int(k8s_state_now[node].get("read_disks_bandwidth", 0)),
                    "write_disks_bandwidth": int(k8s_state_now[node].get("write_disks_bandwidth", 0)),
                    "avg_latency": float(k8s_state_now[node].get("avg_latency", 0.0)),
                    "deployments": {},
                }
                
                # Process deployments if available
                if "deployments" in k8s_state_now[node]:
                    for namespace, deployments in k8s_state_now[node]["deployments"].items():
                        if namespace not in state[node]["deployments"]:
                            state[node]["deployments"][namespace] = {}
                        for deployment, info in deployments.items():
                            state[node]["deployments"][namespace][deployment] = {
                                "replicas": info.get("replicas", 0)
                            }
            except Exception as e:
                logger.error(f"[LWMECPSEnv._get_k8s_state] Failed to process state for node {node}: {str(e)}")
                raise
                
        logger.debug(f"[LWMECPSEnv._get_k8s_state] Current state: {state}")
        return state

    def reward(self):
        """
        Calculate reward based on current state.
        
        Returns:
            tuple: (reward, done)
        """
        logger.info("[LWMECPSEnv.reward] Calculating reward")
        total_latency = 0
        total_pods = 0
        
        for node in self.node_name:
            try:
                # Safely get deployment replicas
                namespace_deployments = self.state[node]["deployments"].get(self.namespace, {})
                deployment_info = namespace_deployments.get(self.deployment_name, {})
                pods = deployment_info.get("replicas", 0)
                
                if pods > 0:
                    total_pods += pods
                    # TODO: Надо сделать запрос средней Latency с тестового окружения (lwmecps-testapp)
                    # TODO: Надо посмотреть на reward
                    total_latency += self.state[node]["avg_latency"] * pods
                    logger.debug(f"[LWMECPSEnv.reward] Node {node}: {pods} pods, latency contribution: {self.state[node]['avg_latency'] * pods}")
            except Exception as e:
                logger.warning(f"[LWMECPSEnv.reward] Error processing node {node}: {str(e)}")
                continue

        # TODO: Надо посмотреть на reward
        if total_pods > 0:
            total_latency /= total_pods
            self.current_latency = total_latency
            done = False
            self.rew_amm -= self.current_latency  # Invert latency for positive reward
            logger.info(f"[LWMECPSEnv.reward] Total pods: {total_pods}, Average latency: {total_latency}, Reward: {self.rew_amm}")
        else:
            done = True
            self.rew_amm = -1000
            logger.warning("[LWMECPSEnv.reward] No pods found, ending episode with negative reward")

        return self.rew_amm, done

    def render(self, mode="human"):
        """
        Render the environment.
        
        Args:
            mode (str): Rendering mode
        """
        if mode == "human":
            nodes_state = {
                node: {
                    "cpu": self.state[node]["cpu"],
                    "ram": self.state[node]["ram"],
                    "avg_latency": self.state[node]["avg_latency"],
                    "deployments": self.state[node]["deployments"],
                }
                for node in self.node_name
            }
            logger.info(f"[LWMECPSEnv.render] Current environment state:\n{nodes_state}")
        elif mode == "rgb_array":
            logger.info("[LWMECPSEnv.render] RGB array rendering mode not implemented")
            # Implement RGB array rendering if needed
            pass

    def close(self):
        """
        Clean up resources.
        """
        logger.info("[LWMECPSEnv.close] Closing environment")
        pass

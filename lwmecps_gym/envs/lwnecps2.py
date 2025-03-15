import re
from time import sleep

import bitmath
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from lwmecps_gym.envs.kubernetes_api import k8s


class LWMECPSEnv2(gym.Env):

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
        super(LWMECPSEnv2, self).__init__()
        self.num_nodes = num_nodes
        self.node_name = node_name
        self.max_hardware = max_hardware
        self.pod_usage = pod_usage
        self.node_info = node_info
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.deployments = deployments
        self.max_pods = max_pods

        self.minikube = k8s()

        self.prev_latency = None
        self.current_latency = None
        self.rew_amm = 0

        self.action_space = spaces.Discrete(self.num_nodes)

        # Здесь мы создаем одномерное пространство состояний
        observation_dim = len(node_name) * (
            6 + len(deployments)
        )  # Размерность пространства состояний
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(observation_dim,), dtype=np.float32
        )

        self.state = None
        self.reset()

    def reset(self):
        self.prev_latency = None
        self.current_latency = None
        self.rew_amm = 0
        self.state = {
            node: {
                "cpu": self.node_info[node]["cpu"],
                "ram": self.node_info[node]["ram"],
                "tx_bandwidth": self.node_info[node]["tx_bandwidth"],
                "rx_bandwidth": self.node_info[node]["rx_bandwidth"],
                "read_disks_bandwidth":
                    self.node_info[node]["read_disks_bandwidth"],
                "write_disks_bandwidth":
                    self.node_info[node]["write_disks_bandwidth"],
                "avg_latency": self.node_info[node]["avg_latency"],
                "deployments": {
                    deployment: {"replicas": 0}
                        for deployment in self.deployments
                },
            }
            for node in self.node_name
        }
        for node in self.state:
            self.minikube.k8s_action(
                namespace=self.namespace,
                deployment_name=self.deployment_name,
                replicas=1,
                node=node,
            )

        return self._get_observation()

    def _get_observation(self):
        # Преобразуем состояние в одномерный вектор
        state_vector = []
        for node in self.node_name:
            node_state = self.state[node]
            state_vector.append(node_state["cpu"] / self.max_hardware["cpu"])
            state_vector.append(node_state["ram"] / self.max_hardware["ram"])
            state_vector.append(
                node_state["tx_bandwidth"] / self.max_hardware["tx_bandwidth"]
            )
            state_vector.append(
                node_state["rx_bandwidth"] / self.max_hardware["rx_bandwidth"]
            )
            state_vector.append(
                node_state["read_disks_bandwidth"]
                / self.max_hardware["read_disks_bandwidth"]
            )
            state_vector.append(
                node_state["write_disks_bandwidth"]
                / self.max_hardware["write_disks_bandwidth"]
            )
            for deployment in self.deployments:
                state_vector.append(
                    node_state["deployments"][deployment]["replicas"] /
                    self.max_pods
                )

        return np.array(state_vector, dtype=np.float32)

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        pod_node = self.node_name[action]
        self.minikube.k8s_action(
            namespace=self.namespace,
            deployment_name=self.deployment_name,
            replicas=1,
            node=pod_node,
        )
        sleep(40)
        self.state = self.k8s_state_gym()
        print("step is passed")
        sleep(3)
        reward, done = self.reward()

        info = {"latency": self.current_latency}

        if (self.prev_latency is not None) and (
            self.current_latency >= self.prev_latency
        ):
            done = True
            reward -= 100

        self.prev_latency = self.current_latency
        return self._get_observation(), reward, done, info

    def reward(self):
        total_latency = 0
        total_pods = 0
        for node in self.node_name:
            pods = 0
            try:
                pods = (
                    self.state[node]["deployments"]
                    [self.deployment_name]["replicas"]
                )
            except KeyError:
                pass
            else:
                total_pods += pods
                total_latency += self.state[node]["avg_latency"] * pods

        print("total pods", total_pods)
        if total_pods > 0:
            total_latency /= total_pods
            print("total pods", total_pods, "total latency", total_latency)
            self.current_latency = total_latency
            done = False
            self.rew_amm -= self.current_latency

        else:
            done = True
            self.rew_amm = -1000

        return self.rew_amm, done

    def k8s_state_gym(self):
        k8s_state_now = self.minikube.k8s_state()

        self.state = {
            node: {
                "cpu": int(k8s_state_now[node]["cpu"]),
                "ram": round(
                    bitmath.KiB(
                        int(re.findall(
                            r"\d+", k8s_state_now[node]["memory"])[0]
                            )
                    )
                    .to_MB()
                    .value
                ),
                "tx_bandwidth": self.node_info[node]["tx_bandwidth"],
                "rx_bandwidth": self.node_info[node]["rx_bandwidth"],
                "read_disks_bandwidth":
                    self.node_info[node]["read_disks_bandwidth"],
                "write_disks_bandwidth":
                    self.node_info[node]["write_disks_bandwidth"],
                "avg_latency": self.node_info[node]["avg_latency"],
                "deployments": (
                    {
                        deployment: {
                            "replicas": k8s_state_now[node]["deployments"]
                            .get(self.namespace, {})
                            .get(deployment, {"replicas": 0})["replicas"]
                        }
                        for deployment in self.deployments
                    }
                    if "deployments" in k8s_state_now[node]
                    else {}
                ),
            }
            for node in self.node_name
        }
        print(self.state)
        return self.state

    def render(self, mode="human"):
        nodes_state = {
            node: {
                "cpu": self.state[node]["cpu"],
                "ram": self.state[node]["ram"],
                "avg_latency": self.state[node]["avg_latency"],
                "deployments": self.state[node]["deployments"],
            }
            for node in self.node_name
        }
        return nodes_state

    def close(self):
        pass

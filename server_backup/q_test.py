from q_learn import QLearningAgent
import pickle
import random
import re
import time

import bitmath
import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from lwmecps_gym.envs.kubernetes_api import k8s
from lwmecps_gym.envs.lwnecps import LWMECPSEnv

if __name__ == "__main__":

    register(
        id="lwmecps-v0",
        entry_point="lwmecps_gym.envs:LWMECPSEnv",
    )
    minikube = k8s()

    node_name = []

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
        "rx_bandwidth": 80,
        "read_disks_bandwidth": 100,
        "write_disks_bandwidth": 100,
    }

    state = minikube.k8s_state()

    for node in state:
        node_name.append(node)

    avg_latency = 100
    node_info = {}

    for node in state:
        avg_latency = avg_latency + 10
        node_info[node] = {
            "cpu": int(state[node]["cpu"]),
            "ram": round(
                bitmath.KiB(int(re.findall(r"\d+", state[node]["memory"])[0]))
                .to_MB()
                .value
            ),
            "tx_bandwidth": 100,
            "rx_bandwidth": 100,
            "read_disks_bandwidth": 300,
            "write_disks_bandwidth": 300,
            "avg_latency": avg_latency,
        }

    env = gym.make(
        "lwmecps-v0",
        max_pods=5,
        num_nodes=len(node_name),
        node_name=node_name,
        max_hardware=max_hardware,
        pod_usage=pod_usage,
        node_info=node_info,
        deployment_name="mec-test-app",
        namespace="default",
        deployments=["mec-test-app"],
    )
    agent = QLearningAgent(env, exploration_rate=0)
    agent.load_q_table("./q_table.pkl")
    agent.train(1)

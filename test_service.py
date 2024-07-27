
from lwmecps_gym.envs.kubernetes_api import k8s
from gymnasium.envs.registration import register
import gymnasium as gym
import bitmath
import re
import time


register(
    id='lwmecps-v0',
    entry_point='lwmecps_gym.envs:LWMECPSEnv',
)
minikube = k8s()

node_name = []

max_hardware = {
    'cpu': 8,
    'ram': 16000,
    'tx_bandwidth': 1000,
    'rx_bandwidth': 1000,
    'read_disks_bandwidth': 500,
    'write_disks_bandwidth': 500,
    'avg_latency': 300
}

pod_usage = {
    'cpu': 2,
    'ram': 2000,
    'tx_bandwidth': 20,
    'rx_bandwidth': 80,
    'read_disks_bandwidth': 100,
    'write_disks_bandwidth': 100
}

state = minikube.k8s_state()


for node in state:
    node_name.append(node)

avg_latency = 10
node_info = {}

for node in state:
    avg_latency = avg_latency + 10 
    node_info[node] = {
        'cpu': int(state[node]['cpu']),
        'ram': round(bitmath.KiB(int(re.findall(r'\d+', state[node]['memory'])[0])).to_MB().value),
        'tx_bandwidth': 100,
        'rx_bandwidth': 100,
        'read_disks_bandwidth': 300,
        'write_disks_bandwidth': 300,
        'avg_latency': avg_latency
    }

# Использование окружения
env = gym.make('lwmecps-v0',num_nodes = len(node_name), node_name = node_name, max_hardware = max_hardware, pod_usage = pod_usage, node_info = node_info, deployment_name = 'mec-test-app', namespace = 'default' )


for _ in range(5):
    observation = env.reset()
    done = False
    while not done:
        env.render()
        print(env.action_space.sample())
        action = env.action_space.sample()  # Случайное действие
        observation, reward, done, info = env.step(action)

env.close()
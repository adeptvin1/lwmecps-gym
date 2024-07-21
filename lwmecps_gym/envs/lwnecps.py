import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.registration import register

class LWMECPSEnv(gym.Env):

    def __init__(self, node_name , max_hardware, pod_usage, node_info, num_nodes):
        super(LWMECPSEnv, self).__init__()
        self.num_nodes = num_nodes
        self.node_name = node_name
        self.max_hardware = max_hardware
        self.pod_usage = pod_usage
        self.node_info = node_info

        # self.render_mode = render_mode
        # self.window_size = window_size
        
        # spaces https://gymnasium.farama.org/api/spaces/composite/
        self.action_space = spaces.Discrete(self.num_nodes)
        
        # spaces https://gymnasium.farama.org/api/spaces/composite/
        self.observation_space = spaces.Dict(
            {
                node: spaces.Dict(
                    {
                        'cpu': spaces.Box(low=0, high=self.max_hardware['cpu'], shape=(), dtype=np.float32),
                        'ram': spaces.Box(low=0, high=self.max_hardware['ram'], shape=(), dtype=np.float32),
                        'tx_bandwidth': spaces.Box(low=0, high=self.max_hardware['tx_bandwidth'], shape=(), dtype=np.float32),
                        'rx_bandwidth': spaces.Box(low=0, high=self.max_hardware['rx_bandwidth'], shape=(), dtype=np.float32),
                        'read_disks_bandwidth': spaces.Box(low=0, high=self.max_hardware['read_disks_bandwidth'], shape=(), dtype=np.float32),
                        'write_disks_bandwidth': spaces.Box(low=0, high=self.max_hardware['write_disks_bandwidth'], shape=(), dtype=np.float32),
                        'avg_latency': spaces.Box(low=0, high=self.max_hardware['avg_latency'], shape=(), dtype=np.float32)
                    }
                # Тут реализован цикл проходящий по именам нод и создающий словарь описания нод 
                ) for node in self.node_name
            }
        )
        self.observation_space['pod_node']  = spaces.Text(64)
        self.state = None
        self.reset()

    def reset(self):
        self.state = {
            node: {
                'cpu': self.node_info[node]['cpu'],
                'ram': self.node_info[node]['ram'],
                'tx_bandwidth': self.node_info[node]['tx_bandwidth'],
                'rx_bandwidth': self.node_info[node]['rx_bandwidth'],
                'read_disks_bandwidth': self.node_info[node]['read_disks_bandwidth'],
                'write_disks_bandwidth': self.node_info[node]['write_disks_bandwidth'],
                'avg_latency': self.node_info[node]['avg_latency']
            } for node in self.node_name
            
        }
        self.state['pod_node'] = self.node_name[0]

        return self.state
    

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        # Текущая нода, на которой находится pod
        current_pod_node = self.state['pod_node']
        
        # Перемещение pod на новую ноду
        pod_node = self.node_name[action]
        node_parameters = ['cpu', 'ram', 'tx_bandwidth', 'rx_bandwidth', 'read_disks_bandwidth', 'write_disks_bandwidth']
        # Освобождение ресурсов на текущей ноде
        for param in node_parameters:
            self.state[current_pod_node][param] = min(self.state[current_pod_node][param] + self.pod_usage[param], self.node_info[current_pod_node][param])
       
        
        # Потребление ресурсов на новой ноде
        for param in node_parameters:
            self.state[pod_node][param] -= self.pod_usage[param]

        # Проверка переполнения ресурсов на новой ноде
        if self.state[pod_node]['cpu'] < 0 or self.state[pod_node]['ram'] < 0 or self.state[pod_node]['tx_bandwidth'] < 0 or self.state[pod_node]['rx_bandwidth'] < 0 or self.state[pod_node]['read_disks_bandwidth'] < 0 or self.state[pod_node]['write_disks_bandwidth'] < 0 :
            done = True
            reward = -100  # Негативное вознаграждение за попытку переполнения ресурсов
        else:
            done = False
            reward = 0  # Можно настроить вознаграждение по-другому в зависимости от задачи

        # Обновление pod node
        self.state['pod_node'] = self.node_name[action]

        info = {}

        return self.state, reward, done, info

    def render(self, mode='human'):
        nodes_state = {node: {'cpu': self.state[node]['cpu'], 'ram': self.state[node]['ram'], 'avg_latency': self.state[node]['avg_latency'] } for node in self.node_name}
        print(f"Nodes: {nodes_state}, Pod Node: {self.state['pod_node']}")


    def close(self):
        pass




register(
    id='lwmecps-v0',
    entry_point='__main__:LWMECPSEnv',
)
node_name = ['node0', 'node1', 'node2', 'node3']
max_hardware = {
    'cpu': 8,
    'ram': 16,
    'tx_bandwidth': 1000,
    'rx_bandwidth': 1000,
    'read_disks_bandwidth': 500,
    'write_disks_bandwidth': 500,
    'avg_latency': 300
}
pod_usage = {
    'cpu': 2,
    'ram': 4,
    'tx_bandwidth': 20,
    'rx_bandwidth': 80,
    'read_disks_bandwidth': 100,
    'write_disks_bandwidth': 100
}

node_info = {
    'node0' : {
        'cpu': 4,
        'ram': 8,
        'tx_bandwidth': 100,
        'rx_bandwidth': 100,
        'read_disks_bandwidth': 300,
        'write_disks_bandwidth': 300,
        'avg_latency': 10
    },
    'node1' : {
        'cpu': 1,
        'ram': 1,
        'tx_bandwidth': 100,
        'rx_bandwidth': 100,
        'read_disks_bandwidth': 300,
        'write_disks_bandwidth': 300,
        'avg_latency': 20
    },
    'node2' : {
        'cpu': 8,
        'ram': 16,
        'tx_bandwidth': 100,
        'rx_bandwidth': 100,
        'read_disks_bandwidth': 300,
        'write_disks_bandwidth': 300,
        'avg_latency': 10
    },
    'node3' : {
        'cpu': 2,
        'ram': 4,
        'tx_bandwidth': 100,
        'rx_bandwidth': 100,
        'read_disks_bandwidth': 300,
        'write_disks_bandwidth': 300,
        'avg_latency': 30
    },    
}
# Использование окружения
env = gym.make('lwmecps-v0',num_nodes = len(node_name), node_name = node_name, max_hardware = max_hardware, pod_usage = pod_usage, node_info = node_info )


for _ in range(5):
    observation = env.reset()
    done = False
    while not done:
        env.render()
        print(env.action_space.sample())
        action = env.action_space.sample()  # Случайное действие
        observation, reward, done, info = env.step(action)

env.close()
import re
from time import sleep

import bitmath
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from lwmecps_gym.envs.kubernetes_api import k8s


class LWMECPSEnv(gym.Env):

    def __init__(self, node_name , max_hardware, pod_usage, node_info, num_nodes, namespace, deployment_name, deployments):
        super(LWMECPSEnv, self).__init__()
        self.num_nodes = num_nodes
        self.node_name = node_name
        self.max_hardware = max_hardware
        self.pod_usage = pod_usage
        self.node_info = node_info
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.deployments = deployments

        self.minikube = k8s()
        
        self.prev_latency = None
        self.current_latency = None
        self.rew_amm = 0
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
                        'avg_latency': spaces.Box(low=0, high=self.max_hardware['avg_latency'], shape=(), dtype=np.float32),
                        'deployments': spaces.Dict({
                            deployment: spaces.Dict({
                                'replicas': spaces.Box(low=0, high=10, shape=(), dtype=np.float32)
                            }) for deployment in self.deployments
                        })
                    }
                # Тут реализован цикл проходящий по именам нод и создающий словарь описания нод 
                ) for node in self.node_name
            }
        )
        self.state = None
        self.reset()

    def reset(self):
        self.prev_latency = None
        self.current_latency = None
        self.rew_amm = 0
        self.state = {
            node: {
                'cpu': self.node_info[node]['cpu'],
                'ram': self.node_info[node]['ram'],
                'tx_bandwidth': self.node_info[node]['tx_bandwidth'],
                'rx_bandwidth': self.node_info[node]['rx_bandwidth'],
                'read_disks_bandwidth': self.node_info[node]['read_disks_bandwidth'],
                'write_disks_bandwidth': self.node_info[node]['write_disks_bandwidth'],
                'avg_latency': self.node_info[node]['avg_latency'],
                'deployments': {
                    deployment: {
                        'replicas': 0  # Пример числового значения
                    } for deployment in self.deployments
                }
            } for node in self.node_name
        }
        for node in self.state:
            # пока не нужно
            # for deployment_name in self.state[node]['deployments']:
                self.minikube.k8s_action(namespace=self.namespace, deployment_name=self.deployment_name, replicas=1, node=node)

        return self.state
    

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"
        # Перемещение pod на новую ноду
        pod_node = self.node_name[action]
        self.minikube.k8s_action(namespace=self.namespace, deployment_name=self.deployment_name, replicas=1, node=pod_node)
        self.state = self.k8s_state_gym()
        print('step is passed')
        sleep(3)
        reward, done = self.reward()
        # node_parameters = ['cpu', 'ram', 'tx_bandwidth', 'rx_bandwidth', 'read_disks_bandwidth', 'write_disks_bandwidth']
        # # Освобождение ресурсов на текущей ноде
        # for param in node_parameters:
        #     self.state[current_pod_node][param] = min(self.state[current_pod_node][param] + self.pod_usage[param], self.node_info[current_pod_node][param])
       
        
        # # Потребление ресурсов на новой ноде
        # for param in node_parameters:
        #     self.state[pod_node][param] -= self.pod_usage[param]

        # # Проверка переполнения ресурсов на новой ноде
        # if self.state[pod_node]['cpu'] < 0 or self.state[pod_node]['ram'] < 0 or self.state[pod_node]['tx_bandwidth'] < 0 or self.state[pod_node]['rx_bandwidth'] < 0 or self.state[pod_node]['read_disks_bandwidth'] < 0 or self.state[pod_node]['write_disks_bandwidth'] < 0 :
        #     done = True
        #     reward = -100  # Негативное вознаграждение за попытку переполнения ресурсов
        # else:
        #     done = False
        #     reward = 0  # Можно настроить вознаграждение по-другому в зависимости от задачи

        # # Обновление pod node
        # self.state['pod_node'] = self.node_name[action]

        info = {}
        print(self.state)
        # Если задержка увеличилась, то завершаем эпизод, так как это ухудшение
        if (self.prev_latency is not None) and (self.current_latency >= self.prev_latency):
            done = True
            reward -= 100  # Негативное вознаграждение за ухудшение

        # Обновление предыдущей задержки
        self.prev_latency = self.current_latency
        return self.state, reward, done, info

    def reward(self):
        total_latency = 0
        total_pods = 0
        for node in self.node_name:
            pods = 0
            # #Смотрим количество выделенных подов. Чем больше подов == тем больше пользователей. В базе считаем что под полностью занят пользователями.
            # print("node", node, "cpu", self.state[node]['cpu'], "pod", self.pod_usage['cpu'])
            # # node_info - начальное состояние нод, state - текущее состояние окружения (свободные ресурсы) деленные на pod usage['cpu']
            # # По факту это условное количество подов ?
            # resource_usage = (self.node_info[node]["cpu"] - self.state[node]['cpu']) // self.pod_usage['cpu']
            # total_resources += resource_usage
            # # сумма всех задержек умноженных на использование ресурсов на каждой ноде (если ресурсов будет меньше = лучше)
            # total_latency += self.state[node]['avg_latency'] * resource_usage
            try:
                pods = self.state[node]['deployments'][self.deployment_name]['replicas']
            except KeyError:
                pass
            else:
                total_pods += pods
                total_latency += self.state[node]['avg_latency'] * pods
         

        print("total pods", total_pods)
        #Слабо представляю момент, когда будет 0 ресурсов, только если 0 нод в системе есть, либо если не выделен ни один под. Но перестраховаться никогда не плохо
        if total_pods > 0:
            total_latency /= total_pods
            print("total pods", total_pods, "total latency", total_latency)
            self.current_latency = total_latency 
            done = False
            self.rew_amm -= self.current_latency # Инверсируем задержку, чтобы минимизация давала положительный reward

        else: 
            done = True
            self.rew_amm = -1000

        
        return self.rew_amm, done

    def k8s_state_gym(self):
        k8s_state_now = self.minikube.k8s_state()

        self.state = {
            node: {
                'cpu': int(k8s_state_now[node]['cpu']),
                'ram': round(bitmath.KiB(int(re.findall(r'\d+', k8s_state_now[node]['memory'])[0])).to_MB().value),
                'tx_bandwidth': self.node_info[node]['tx_bandwidth'],
                'rx_bandwidth': self.node_info[node]['rx_bandwidth'],
                'read_disks_bandwidth': self.node_info[node]['read_disks_bandwidth'],
                'write_disks_bandwidth': self.node_info[node]['write_disks_bandwidth'],
                'avg_latency': self.node_info[node]['avg_latency'],
                'deployments': {
                    deployment: {
                        'replicas': k8s_state_now[node]['deployments'].get(self.namespace, {}).get(deployment, {'replicas': 0})['replicas']
                    } for deployment in self.deployments
                } if 'deployments' in k8s_state_now[node] else {}
            } for node in self.node_name
        }
        print(self.state)
        return self.state

    def render(self, mode='human'):
        nodes_state = {node: {'cpu': self.state[node]['cpu'], 'ram': self.state[node]['ram'], 'avg_latency': self.state[node]['avg_latency'], 'deployments': self.state[node]['deployments']  } for node in self.node_name}


    def close(self):
        pass

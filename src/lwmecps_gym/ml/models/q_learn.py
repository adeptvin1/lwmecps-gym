import json
import pickle
import random
import re
from time import time

import bitmath
import gymnasium as gym
import numpy as np
import wandb
from gymnasium.envs.registration import register
from lwmecps_gym.envs.kubernetes_api import k8s


class QLearningAgent:
    def __init__(
        self,
        env,
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=1.0,
        exploration_decay=0.98,
        wandb_run_id=None,
    ):
        self.env = env
        # Get the original environment by unwrapping all wrappers
        self.original_env = env
        while hasattr(self.original_env, 'env'):
            self.original_env = self.original_env.env
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.wandb_run_id = wandb_run_id

        # Инициализация Q-таблицы для каждой ноды и каждого действия
        self.q_table = {
            node: np.zeros(self.original_env.action_space.n) for node in self.original_env.node_name
        }

        # Инициализация wandb
        if self.wandb_run_id:
            wandb.init(
                project="lwmecps-gym",
                id=self.wandb_run_id,
                config={
                    "learning_rate": learning_rate,
                    "discount_factor": discount_factor,
                    "exploration_rate": exploration_rate,
                    "exploration_decay": exploration_decay,
                    "model_type": "q_learning",
                }
            )

    def save_q_table(self, file_name):
        try:
            with open(file_name, "wb") as f:
                pickle.dump(self.q_table, f)
            print(f"Q-таблица сохранена в {file_name}")
        except Exception as e:
            print(f"Ошибка при сохранении Q-таблицы: {str(e)}")
            raise

    def load_q_table(self, file_name):
        try:
            with open(file_name, "rb") as f:
                self.q_table = pickle.load(f)
            print(f"Q-таблица загружена из {file_name}")
        except Exception as e:
            print(f"Ошибка при загрузке Q-таблицы: {str(e)}")
            raise

    def choose_action(self, state):
        node = self.get_current_node(state)
        if random.uniform(0, 1) < self.exploration_rate:
            return self.original_env.action_space.sample()  # Исследование
        else:
            return np.argmax(self.q_table[node])  # Эксплуатация

    def update_q_table(self, state, action, reward, next_state):
        try:
            node = self.get_current_node(state)
            next_node = self.get_current_node(next_state)
            
            # Проверяем, существуют ли состояния в Q-таблице
            if node not in self.q_table or next_node not in self.q_table:
                print(f"Предупреждение: узел {node} или {next_node} не найден в Q-таблице")
                return
                
            best_next_action = np.argmax(self.q_table[next_node])

            # Обновление Q-значения
            q_value = self.q_table[node][action]
            self.q_table[node][action] = q_value + self.learning_rate * (
                reward
                + self.discount_factor * self.q_table[next_node][best_next_action]
                - q_value
            )

            # Логируем обновление Q-значения в wandb
            if self.wandb_run_id:
                wandb.log({
                    f"q_value/{node}/{action}": self.q_table[node][action],
                    f"q_value_change/{node}/{action}": self.q_table[node][action] - q_value,
                })
        except Exception as e:
            print(f"Ошибка при обновлении Q-таблицы: {str(e)}")
            raise

    def get_current_node(self, state):
        """
        Получение текущего узла на основе состояния.
        Ищем узел, где находится под в данный момент.
        """
        try:
            for node in self.original_env.node_name:
                if node in state:
                    namespace_deployments = state[node]["deployments"].get(self.original_env.namespace, {})
                    deployment_info = namespace_deployments.get(self.original_env.deployment_name, {})
                    if deployment_info.get("replicas", 0) > 0:
                        return node
            # Если под не найден, возвращаем первую ноду
            return next(iter(state))
        except Exception as e:
            print(f"Ошибка при определении текущего узла: {str(e)}")
            raise

    def train(self, episodes):
        episode_latency = {}
        episode_reward = {}
        try:
            for episode in range(episodes):
                # В Gymnasium reset возвращает (state, info)
                state, _ = self.env.reset()
                done = False
                episode_reward[episode] = 0
                episode_steps = 0

                while not done:
                    action = self.choose_action(state)
                    next_state, reward, done, info = self.env.step(action)
                    if not done:
                        print(f"Episode {episode + 1}, Reward: {reward}")
                        self.update_q_table(state, action, reward, next_state)
                        state = next_state
                        episode_reward[episode] += reward
                        episode_latency[episode] = info["latency"]
                        episode_steps += 1

                self.exploration_rate *= self.exploration_decay
                print(f"Episode {episode + 1}: exploration rate = {self.exploration_rate}")

                # Сохраняем модель после каждого эпизода
                model_file = f"./q_table_episode_{episode}.pkl"
                self.save_q_table(model_file)
                
                # Сохраняем модель в wandb
                if self.wandb_run_id:
                    wandb.save(model_file)
                    # Логируем метрики эпизода в wandb
                    wandb.log({
                        "episode": episode,
                        "episode_reward": episode_reward[episode],
                        "episode_latency": episode_latency[episode],
                        "episode_steps": episode_steps,
                        "exploration_rate": self.exploration_rate,
                    })

                # Сохраняем результаты после каждого эпизода
                try:
                    with open("./q_episode_latency.json", "w") as file:
                        json.dump(episode_latency, file, indent=4)

                    with open("./q_episode_reward.json", "w") as file:
                        json.dump(episode_reward, file, indent=4)

                    # Сохраняем файлы в wandb
                    if self.wandb_run_id:
                        wandb.save("./q_episode_latency.json")
                        wandb.save("./q_episode_reward.json")
                except Exception as e:
                    print(f"Ошибка при сохранении результатов эпизода: {str(e)}")

        except Exception as e:
            print(f"Ошибка во время обучения: {str(e)}")
            raise
        finally:
            # Завершаем сессию wandb
            if self.wandb_run_id:
                wandb.finish()


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
        "rx_bandwidth": 20,
        "read_disks_bandwidth": 100,
        "write_disks_bandwidth": 100,
    }

    state = minikube.k8s_state()

    max_pods = 10000

    for node in state:
        node_name.append(node)

    avg_latency = 10
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
        # Работатет только пока находятся в том же порядке.
        max_pods = min(
            [
                min(
                    [
                        val // pod_usage[key]
                        for key, (_, val) in zip(
                            pod_usage.keys(), node_info[node].items()
                        )
                    ]
                ),
                max_pods,
            ]
        )

    env = gym.make(
        "lwmecps-v0",
        num_nodes=len(node_name),
        node_name=node_name,
        max_hardware=max_hardware,
        pod_usage=pod_usage,
        node_info=node_info,
        deployment_name="mec-test-app",
        namespace="default",
        deployments=["mec-test-app"],
        max_pods=max_pods,
    )
    agent = QLearningAgent(env)
    start = time()
    agent.train(episodes=100)
    print(f"Training time: {(time() - start)}")
    agent.save_q_table("./q_table.pkl")

import json
import pickle
import random
import re
from time import time
import os

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
            try:
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
                print(f"Successfully initialized wandb run with ID {self.wandb_run_id}")
            except Exception as e:
                print(f"Failed to initialize wandb: {str(e)}")
                self.wandb_run_id = None

    def save_q_table(self, file_name):
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(file_name)), exist_ok=True)
            
            # Save locally
            with open(file_name, "wb") as f:
                pickle.dump(self.q_table, f)
            print(f"Q-таблица сохранена в {file_name}")
            
            # Save to wandb if initialized
            if self.wandb_run_id:
                try:
                    # Save as a wandb artifact
                    artifact = wandb.Artifact('q_table', type='model')
                    artifact.add_file(file_name)
                    wandb.log_artifact(artifact)
                    print(f"Successfully saved model to wandb as artifact")
                    
                    # Also save directly
                    wandb.save(file_name)
                    print(f"Successfully saved model file to wandb")
                except Exception as e:
                    print(f"Failed to save model to wandb: {str(e)}")
                
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
                    # Проверяем наличие пода на узле через deployments
                    deployments = state[node]["deployments"]
                    if self.original_env.deployment_name in deployments:
                        if deployments[self.original_env.deployment_name]["replicas"] > 0:
                            return node
            # Если под не найден, возвращаем первую ноду
            return next(iter(state))
        except Exception as e:
            print(f"Ошибка при определении текущего узла: {str(e)}")
            raise

    def train(self, episodes):
        episode_latency = {}
        episode_reward = {}
        episode_steps = {}
        episode_exploration = {}
        
        try:
            for episode in range(episodes):
                print(f"\nStarting episode {episode + 1}/{episodes}")
                state = self.env.reset()[0]  # Получаем только observation
                total_reward = 0
                steps = 0
                
                while True:
                    action = self.choose_action(state)
                    next_state, reward, done, truncated, info = self.env.step(action)
                    self.update_q_table(state, action, reward, next_state)
                    
                    total_reward += reward
                    steps += 1
                    
                    # Логируем текущее состояние и награду
                    current_node = self.get_current_node(state)
                    print(f"Step {steps}:")
                    print(f"  Current node: {current_node}")
                    print(f"  Action: {action} (node {self.original_env.node_name[action]})")
                    print(f"  Reward: {reward}")
                    print(f"  State: {state}")
                    print(f"  Total reward so far: {total_reward}")
                    
                    if done or truncated:
                        break
                    
                    state = next_state
                
                # Сохраняем метрики эпизода
                current_node = self.get_current_node(state)
                episode_latency[episode] = state[current_node]["avg_latency"]
                episode_reward[episode] = total_reward
                episode_steps[episode] = steps
                episode_exploration[episode] = self.exploration_rate
                
                # Обновляем epsilon
                self.exploration_rate = max(0.01, self.exploration_rate * self.exploration_decay)
                
                # Логируем итоги эпизода
                print(f"\nEpisode {episode + 1} completed:")
                print(f"  Total steps: {steps}")
                print(f"  Total reward: {total_reward}")
                print(f"  Final latency: {state[current_node]['avg_latency']}ms")
                print(f"  Exploration rate: {self.exploration_rate}")
                
                # Логируем метрики в wandb
                if self.wandb_run_id:
                    try:
                        metrics = {
                            "episode": episode,
                            "total_reward": total_reward,
                            "steps": steps,
                            "epsilon": self.exploration_rate,
                            "latency": state[current_node]["avg_latency"]
                        }
                        wandb.log(metrics)
                        print(f"Successfully logged metrics for episode {episode}")
                    except Exception as e:
                        print(f"Failed to log metrics to wandb: {str(e)}")
            
            return {
                "episode_latency": episode_latency,
                "episode_reward": episode_reward,
                "episode_steps": episode_steps,
                "episode_exploration": episode_exploration
            }
            
        except Exception as e:
            print(f"Ошибка при обучении: {str(e)}")
            raise


if __name__ == "__main__":
    # Регистрируем окружение
    register(
        id="lwmecps-v3",
        entry_point="lwmecps_gym.envs:LWMECPSEnv3",
    )

    # Инициализируем Kubernetes клиент
    minikube = k8s()
    state = minikube.k8s_state()
    node_name = list(state.keys())

    # Базовые параметры
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

    # Создаем информацию о узлах
    node_info = {}
    for node in node_name:
        node_info[node] = {
            "cpu": int(state[node]["cpu"]),
            "ram": round(bitmath.KiB(int(re.findall(r"\d+", state[node]["memory"])[0])).to_MB().value),
            "tx_bandwidth": 100,
            "rx_bandwidth": 100,
            "read_disks_bandwidth": 300,
            "write_disks_bandwidth": 300,
            "avg_latency": 10 + (10 * list(node_name).index(node)),  # Увеличиваем задержку для каждого следующего узла
        }

    # Создаем окружение
    env = gym.make(
        "lwmecps-v3",
        node_name=node_name,
        max_hardware=max_hardware,
        pod_usage=pod_usage,
        node_info=node_info,
        num_nodes=len(node_name),
        namespace="default",
        deployment_name="mec-test-app",
        deployments=["mec-test-app"],
        max_pods=10000,
        group_id="test-group-1"
    )

    # Создаем и обучаем агента
    agent = QLearningAgent(env)
    start = time()
    agent.train(episodes=100)
    print(f"Training time: {(time() - start)}")
    agent.save_q_table("./q_table.pkl")

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
        episode_steps = {}
        episode_exploration = {}
        
        try:
            for episode in range(episodes):
                state = self.env.reset()[0]  # Получаем только observation
                total_reward = 0
                steps = 0
                
                while True:
                    action = self.choose_action(state)
                    next_state, reward, done, info = self.env.step(action)
                    self.update_q_table(state, action, reward, next_state)
                    
                    total_reward += reward
                    steps += 1
                    
                    if done:
                        break
                    
                    state = next_state
                
                # Сохраняем метрики эпизода
                episode_latency[episode] = info.get("latency", 0)
                episode_reward[episode] = total_reward
                episode_steps[episode] = steps
                episode_exploration[episode] = self.exploration_rate
                
                # Обновляем epsilon
                self.exploration_rate = max(0.01, self.exploration_rate * self.exploration_decay)
                
                # Логируем метрики в wandb
                if self.wandb_run_id:
                    try:
                        metrics = {
                            "episode": episode,
                            "total_reward": total_reward,
                            "steps": steps,
                            "epsilon": self.exploration_rate,
                            "latency": info.get("latency", 0)
                        }
                        wandb.log(metrics)
                        print(f"Successfully logged metrics for episode {episode}")
                    except Exception as e:
                        print(f"Failed to log metrics to wandb: {str(e)}")
                
                # Сохраняем результаты после каждого эпизода
                try:
                    # Ensure the directory exists
                    os.makedirs("metrics", exist_ok=True)
                    
                    # Save metrics files
                    latency_file = os.path.join("metrics", "q_episode_latency.json")
                    reward_file = os.path.join("metrics", "q_episode_reward.json")
                    
                    with open(latency_file, "w") as file:
                        json.dump(episode_latency, file, indent=4)
                    with open(reward_file, "w") as file:
                        json.dump(episode_reward, file, indent=4)

                    # Сохраняем файлы в wandb
                    if self.wandb_run_id:
                        try:
                            wandb.save(latency_file)
                            wandb.save(reward_file)
                            print(f"Successfully saved metric files to wandb")
                        except Exception as e:
                            print(f"Failed to save metric files to wandb: {str(e)}")
                except Exception as e:
                    print(f"Ошибка при сохранении результатов эпизода: {str(e)}")
            
            # Возвращаем метрики в двух форматах
            return {
                # Метрики для TrainingTask (списки)
                "task_metrics": {
                    "final_latency": [float(episode_latency.get(episodes - 1, 0))],
                    "final_reward": [float(episode_reward.get(episodes - 1, 0))],
                    "final_steps": [float(episode_steps.get(episodes - 1, 0))],
                    "final_exploration": [float(episode_exploration.get(episodes - 1, 0))],
                    "avg_latency": [float(sum(episode_latency.values()) / len(episode_latency) if episode_latency else 0)],
                    "avg_reward": [float(sum(episode_reward.values()) / len(episode_reward) if episode_reward else 0)],
                    "avg_steps": [float(sum(episode_steps.values()) / len(episode_steps) if episode_steps else 0)]
                },
                # Метрики для TrainingResult (числа)
                "result_metrics": {
                    "final_latency": float(episode_latency.get(episodes - 1, 0)),
                    "final_reward": float(episode_reward.get(episodes - 1, 0)),
                    "final_steps": float(episode_steps.get(episodes - 1, 0)),
                    "final_exploration": float(episode_exploration.get(episodes - 1, 0)),
                    "avg_latency": float(sum(episode_latency.values()) / len(episode_latency) if episode_latency else 0),
                    "avg_reward": float(sum(episode_reward.values()) / len(episode_reward) if episode_reward else 0),
                    "avg_steps": float(sum(episode_steps.values()) / len(episode_steps) if episode_steps else 0)
                }
            }

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

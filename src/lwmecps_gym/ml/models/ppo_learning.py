import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gymnasium import spaces
import logging
from typing import List, Dict, Union, Tuple, Optional
import wandb

from lwmecps_gym.envs import LWMECPSEnv

logger = logging.getLogger(__name__)

class ActorCritic(nn.Module):
    """
    Архитектура нейронной сети для PPO, состоящая из двух частей:
    1. Actor (политика) - определяет распределение действий
    2. Critic (функция ценности) - оценивает ожидаемую награду
    
    Args:
        obs_dim (int): Размерность вектора наблюдения
        act_dim (int): Размерность пространства действий
        hidden_size (int): Размер скрытых слоев
    """
    def __init__(self, obs_dim, act_dim, hidden_size=64):
        super().__init__()

        # Актер (Policy) - определяет распределение действий
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),  # Tanh для лучшей стабильности обучения
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, act_dim),
            nn.Softmax(dim=-1)  # Преобразует выход в вероятности действий
        )

        # Критик (Value function) - оценивает ожидаемую награду
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)  # Одна выходная величина - оценка ценности состояния
        )

    def forward(self, x):
        """
        Прямой проход через сеть.
        
        Args:
            x (torch.Tensor): Входной тензор состояния
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (распределение действий, оценка ценности)
        """
        action_probs = self.actor(x)
        value = self.critic(x)
        return action_probs, value

    def get_action_and_value(self, x):
        """
        Получение действия, его логарифмической вероятности и оценки ценности.
        
        Args:
            x (torch.Tensor): Входной тензор состояния
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.distributions.Distribution, torch.Tensor]:
                (действие, log_prob, распределение, оценка ценности)
        """
        action_probs, value = self(x)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, dist, value


class RolloutBuffer:
    """
    Буфер для хранения траекторий опыта.
    Используется для сбора данных перед обновлением политики.
    
    Args:
        n_steps (int): Количество шагов для сбора перед обновлением
        obs_dim (int): Размерность вектора наблюдения
    """
    def __init__(self, n_steps, obs_dim):
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.reset()

    def reset(self):
        """Сброс буфера для нового сбора данных."""
        self.states = np.zeros((self.n_steps, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.n_steps, dtype=np.int64)
        self.rewards = np.zeros(self.n_steps, dtype=np.float32)
        self.values = np.zeros(self.n_steps, dtype=np.float32)
        self.log_probs = np.zeros(self.n_steps, dtype=np.float32)
        self.dones = np.zeros(self.n_steps, dtype=np.float32)
        self.advantages = np.zeros(self.n_steps, dtype=np.float32)
        self.returns = np.zeros(self.n_steps, dtype=np.float32)
        self.pos = 0

    def add(self, state, action, reward, value, log_prob, done):
        """
        Добавление одного шага опыта в буфер.
        
        Args:
            state (np.ndarray): Состояние
            action (int): Выбранное действие
            reward (float): Полученная награда
            value (float): Оценка ценности состояния
            log_prob (float): Логарифмическая вероятность действия
            done (bool): Флаг завершения эпизода
        """
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.dones[self.pos] = done
        self.pos += 1

    def is_full(self):
        """Проверка, заполнен ли буфер."""
        return self.pos >= self.n_steps

    def compute_advantages(self, gamma, lam):
        """
        Вычисление преимуществ (advantages) с использованием GAE.
        
        Args:
            gamma (float): Коэффициент дисконтирования
            lam (float): Параметр lambda для GAE
        """
        gae = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_value = 0
            else:
                next_value = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            self.advantages[t] = gae
            self.returns[t] = self.advantages[t] + self.values[t]


class MetricsCollector:
    def __init__(self):
        self.metrics = {}
    
    def update(self, metrics_dict: Dict[str, float]):
        """Update metrics with new values."""
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Calculate average values for all metrics."""
        return {key: np.mean(values) for key, values in self.metrics.items() if values}
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get the most recent values for all metrics."""
        return {key: values[-1] for key, values in self.metrics.items() if values}
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()


class PPO:
    """
    Реализация алгоритма Proximal Policy Optimization.
    
    Args:
        obs_dim (int): Размерность вектора наблюдения
        act_dim (int): Размерность пространства действий
        hidden_size (int): Размер скрытых слоев
        lr (float): Скорость обучения
        gamma (float): Коэффициент дисконтирования
        lam (float): Параметр lambda для GAE
        clip_eps (float): Параметр клиппинга для PPO
        ent_coef (float): Коэффициент энтропии
        vf_coef (float): Коэффициент функции ценности
        n_steps (int): Количество шагов для сбора перед обновлением
        batch_size (int): Размер батча для обновления
        n_epochs (int): Количество эпох обновления
        device (str): Устройство для вычислений (cpu/cuda)
        deployments (List[str]): Список развертываний для обработки
    """
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_size: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        device: str = "cpu",
        deployments: List[str] = None
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lr = lr
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device
        self.deployments = deployments or []
        self.metrics_collector = MetricsCollector()

        # Инициализация модели и оптимизатора
        self.model = ActorCritic(obs_dim, act_dim, hidden_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.buffer = RolloutBuffer(n_steps, obs_dim)

    def _flatten_observation(self, obs):
        """
        Преобразует наблюдение в одномерный вектор.
        
        Args:
            obs: Наблюдение из среды
            
        Returns:
            np.ndarray: Одномерный вектор наблюдения
        """
        if isinstance(obs, np.ndarray):
            return obs
            
        # Сортируем узлы для стабильного порядка
        sorted_nodes = sorted(obs.keys())
        
        # Создаем вектор наблюдения
        obs_vector = []
        for node in sorted_nodes:
            node_obs = obs[node]
            # Добавляем метрики оборудования
            obs_vector.extend([
                float(node_obs["cpu"]),
                float(node_obs["ram"]),
                float(node_obs["tx_bandwidth"]),
                float(node_obs["rx_bandwidth"]),
                float(node_obs["read_disks_bandwidth"]),
                float(node_obs["write_disks_bandwidth"]),
                float(node_obs["avg_latency"])
            ])
            # Добавляем метрики развертываний
            for deployment in self.deployments:
                obs_vector.append(float(node_obs["deployments"][deployment]["replicas"]))
        
        return np.array(obs_vector, dtype=np.float32)

    def select_action(self, state: Union[np.ndarray, Dict, Tuple]) -> Tuple[int, float, float]:
        """
        Выбор действия на основе текущего состояния.
        
        Args:
            state: Текущее состояние
            
        Returns:
            Tuple[int, float, float]: (выбранное действие, log_prob, оценка ценности)
        """
        try:
            # Преобразование состояния в тензор
            state = self._flatten_observation(state)
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Получение действия и оценки ценности
            action, log_prob, _, value = self.model.get_action_and_value(state)
            
            return action.item(), log_prob.item(), value.item()
        except Exception as e:
            logger.error(f"Error in select_action: {str(e)}")
            raise

    def collect_trajectories(self, env) -> Tuple[float, int]:
        """
        Сбор траекторий опыта из среды.
        
        Args:
            env: Среда для взаимодействия
            
        Returns:
            Tuple[float, int]: (награда за эпизод, длина эпизода)
        """
        self.buffer.reset()
        state, _ = env.reset()  # Получаем начальное состояние и info
        done = False
        episode_reward = 0
        episode_length = 0

        try:
            while not self.buffer.is_full():
                action, log_prob, value = self.select_action(state)
                next_state, reward, done, info = env.step(action)
                # Преобразуем состояние в плоский массив перед добавлением в буфер
                flattened_state = self._flatten_observation(state)
                self.buffer.add(flattened_state, action, reward, value, log_prob, done)
                state = next_state
                episode_reward += reward
                episode_length += 1

                if done:
                    state, _ = env.reset()  # Получаем новое начальное состояние и info
                    episode_reward = 0
                    episode_length = 0

            self.buffer.compute_advantages(self.gamma, self.lam)
            return episode_reward, episode_length
        except Exception as e:
            logger.error(f"Error in collect_trajectories: {str(e)}")
            raise

    def calculate_metrics(self, state, action, reward, value, log_prob, info, actor_loss: Optional[float] = None, critic_loss: Optional[float] = None) -> Dict[str, float]:
        """Calculate all required metrics for the current step."""
        # Calculate accuracy (1 if reward is positive, 0 otherwise)
        accuracy = 1.0 if reward > 0 else 0.0
        
        # Calculate MSE (Mean Squared Error) between reward and value
        mse = (reward - value) ** 2
        
        # Calculate MRE (Mean Relative Error)
        mre = abs(reward - value) / (abs(reward) + 1e-6)
        
        # Get latency from info
        avg_latency = info.get("latency", 0)
        
        metrics = {
            "accuracy": accuracy,
            "mse": mse,
            "mre": mre,
            "avg_latency": avg_latency,
            "total_reward": reward,
            "value": value,
            "log_prob": log_prob
        }
        
        if actor_loss is not None:
            metrics["actor_loss"] = actor_loss
        if critic_loss is not None:
            metrics["critic_loss"] = critic_loss
            
        return metrics

    def update(self) -> Dict[str, float]:
        """Обновление политики на основе собранных данных."""
        # Вычисление преимуществ
        self.buffer.compute_advantages(self.gamma, self.lam)
        
        # Подготовка данных для обновления
        states = torch.FloatTensor(self.buffer.states).to(self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        old_values = torch.FloatTensor(self.buffer.values).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        advantages = torch.FloatTensor(self.buffer.advantages).to(self.device)
        returns = torch.FloatTensor(self.buffer.returns).to(self.device)
        
        # Нормализация преимуществ
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        # Обновление в течение нескольких эпох
        for _ in range(self.n_epochs):
            # Создание батчей
            indices = np.random.permutation(self.n_steps)
            for start_idx in range(0, self.n_steps, self.batch_size):
                idx = indices[start_idx:start_idx + self.batch_size]
                
                # Получение новых значений
                action_probs, values = self.model(states[idx])
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(actions[idx])
                entropy = dist.entropy().mean()
                
                # Вычисление отношения вероятностей
                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                
                # PPO loss
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[idx]
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                critic_loss = 0.5 * (returns[idx] - values.squeeze()).pow(2).mean()
                
                # Общий loss
                loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy
                
                # Обновление параметров
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Сбор метрик
                step_metrics = self.calculate_metrics(
                    states[idx].cpu().numpy(),
                    actions[idx].cpu().numpy(),
                    self.buffer.rewards[idx],
                    values.squeeze().detach().cpu().numpy(),
                    new_log_probs.detach().cpu().numpy(),
                    {"latency": self.buffer.rewards[idx].mean()},
                    actor_loss.item(),
                    critic_loss.item()
                )
                self.metrics_collector.update(step_metrics)
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
        
        # Сброс буфера
        self.buffer.reset()
        
        # Возврат средних значений loss
        return {
            "actor_loss": total_actor_loss / (self.n_epochs * (self.n_steps // self.batch_size)),
            "critic_loss": total_critic_loss / (self.n_epochs * (self.n_steps // self.batch_size)),
            "entropy": total_entropy / (self.n_epochs * (self.n_steps // self.batch_size))
        }

    def train(self, env, total_timesteps: int, wandb_run_id: Optional[str] = None) -> Dict[str, List[float]]:
        """Обучение агента."""
        timesteps_so_far = 0
        episode_metrics = {}
        
        while timesteps_so_far < total_timesteps:
            # Сбор траекторий
            episode_reward, episode_length = self.collect_trajectories(env)
            timesteps_so_far += episode_length
            
            # Обновление политики
            update_metrics = self.update()
            
            # Получение средних метрик
            avg_metrics = self.metrics_collector.get_average_metrics()
            
            # Объединение всех метрик
            all_metrics = {**avg_metrics, **update_metrics}
            
            # Сохранение метрик
            for key, value in all_metrics.items():
                if key not in episode_metrics:
                    episode_metrics[key] = []
                episode_metrics[key].append(value)
            
            # Логирование в wandb
            if wandb_run_id:
                wandb.log({
                    "timesteps": timesteps_so_far,
                    "episode_reward": episode_reward,
                    "episode_length": episode_length,
                    **all_metrics
                })
            
            # Вывод информации
            logger.info(
                f"Timesteps: {timesteps_so_far}/{total_timesteps}, "
                f"Episode Reward: {episode_reward:.2f}, "
                f"Episode Length: {episode_length}, "
                f"Actor Loss: {update_metrics['actor_loss']:.3f}, "
                f"Critic Loss: {update_metrics['critic_loss']:.3f}, "
                f"Entropy: {update_metrics['entropy']:.3f}, "
                f"Accuracy: {avg_metrics.get('accuracy', 0):.3f}, "
                f"MSE: {avg_metrics.get('mse', 0):.3f}, "
                f"MRE: {avg_metrics.get('mre', 0):.3f}, "
                f"Avg Latency: {avg_metrics.get('avg_latency', 0):.2f}"
            )
        
        return episode_metrics

    def save_model(self, path: str):
        """
        Сохранение модели.
        
        Args:
            path (str): Путь для сохранения
        """
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'obs_dim': self.obs_dim,
                'act_dim': self.act_dim,
                'deployments': self.deployments
            }, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, path: str):
        """
        Загрузка модели.
        
        Args:
            path (str): Путь к сохраненной модели
        """
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.obs_dim = checkpoint['obs_dim']
            self.act_dim = checkpoint['act_dim']
            self.deployments = checkpoint['deployments']
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise


def main():
    # Создаем окружение
    env = LWMECPSEnv(
        num_nodes=3,
        node_name=["node1", "node2", "node3"],
        max_hardware={
            "cpu": 8,
            "ram": 16000,
            "tx_bandwidth": 1000,
            "rx_bandwidth": 1000,
            "read_disks_bandwidth": 500,
            "write_disks_bandwidth": 500,
            "avg_latency": 300,
        },
        pod_usage={
            "cpu": 2,
            "ram": 2000,
            "tx_bandwidth": 20,
            "rx_bandwidth": 20,
            "read_disks_bandwidth": 100,
            "write_disks_bandwidth": 100,
        },
        node_info={},
        deployment_name="mec-test-app",
        namespace="default",
        deployments=["mec-test-app"],
        max_pods=10000,
    )

    # Вычисляем размерность наблюдения
    obs_dim = 0
    for node in env.node_name:
        # Добавляем метрики оборудования
        obs_dim += 7  # cpu, ram, tx_bandwidth, rx_bandwidth, read_disks_bandwidth, write_disks_bandwidth, avg_latency
        # Добавляем метрики развертываний
        obs_dim += len(env.deployments)  # replicas для каждого развертывания

    act_dim = env.action_space.n  # 3 действия: scale down, no change, scale up

    # Создаём агент PPO
    ppo_agent = PPO(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_size=64,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        device="cpu",
        deployments=env.deployments
    )

    # Запускаем обучение на 10 итераций
    print("Starting PPO training for 10 iterations...")
    ppo_agent.train(env, total_timesteps=20480, wandb_run_id="ppo_training")  # 2048 * 10 = 20480 timesteps

    # Тестируем обученную модель
    print("\nTesting trained model...")
    state = env.reset()
    done = False
    cum_reward = 0.0
    while not done:
        action, _, _ = ppo_agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        cum_reward += reward
        state = next_state

    env.render()
    print(f"Final cumulative reward: {cum_reward:.2f}")
    print(f"Episode info: {info}")


if __name__ == "__main__":
    main()
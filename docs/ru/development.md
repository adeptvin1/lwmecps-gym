# Разработка алгоритмов

## Обзор

LWMECPS Gym предоставляет среду для разработки и тестирования алгоритмов обучения с подкреплением для оптимизации размещения сервисов в Kubernetes кластере. В этом документе описаны основные концепции и примеры реализации алгоритмов.

## Архитектура

### Компоненты

1. **Среда (Environment)**
   - `LWMECPSEnv` - базовая среда
   - `LWMECPSEnv2` - улучшенная версия
   - `LWMECPSEnv3` - последняя версия

2. **Агенты (Agents)**
   - `QLearningAgent` - Q-learning
   - `DQNAgent` - Deep Q-Network
   - `PPO` - Proximal Policy Optimization

3. **Сервисы**
   - `TrainingService` - управление обучением
   - `ModelService` - управление моделями
   - `MetricsService` - сбор метрик

## Разработка нового алгоритма

### 1. Создание агента

```python
from typing import Dict, Any
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

class YourAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
        wandb_run_id: str = None,
        **kwargs
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.wandb_run_id = wandb_run_id
        
        # Инициализация нейронной сети
        self.network = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, env.action_space.n)
        )
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Инициализация W&B
        if self.wandb_run_id:
            wandb.init(
                project="lwmecps-gym",
                id=self.wandb_run_id,
                config={
                    "learning_rate": learning_rate,
                    "discount_factor": discount_factor,
                    "epsilon": epsilon,
                    "model_type": "your_agent",
                    **kwargs
                }
            )

    def choose_action(self, state: np.ndarray) -> int:
        """
        Выбор действия на основе текущего состояния
        
        Args:
            state: Текущее состояние
            
        Returns:
            int: Выбранное действие
        """
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.network(state_tensor)
        return q_values.argmax().item()

    def update_model(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Обновление модели на основе опыта
        
        Args:
            state: Текущее состояние
            action: Выполненное действие
            reward: Полученная награда
            next_state: Следующее состояние
            done: Флаг завершения эпизода
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        current_q = self.network(state_tensor)[0][action]
        next_q = self.network(next_state_tensor).max().item()
        
        target_q = reward + (1 - done) * self.discount_factor * next_q
        
        loss = nn.MSELoss()(current_q, torch.tensor([target_q]))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, episodes: int) -> Dict[str, Any]:
        """
        Обучение агента
        
        Args:
            episodes: Количество эпизодов обучения
            
        Returns:
            Dict[str, Any]: Метрики обучения
        """
        metrics = {
            "episode_rewards": [],
            "episode_steps": [],
            "episode_latencies": []
        }
        
        for episode in range(episodes):
            state = self.env.reset()[0]
            total_reward = 0
            steps = 0
            
            while True:
                action = self.choose_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                self.update_model(state, action, reward, next_state, done)
                
                total_reward += reward
                steps += 1
                
                if done:
                    break
                    
                state = next_state
            
            # Логирование метрик
            metrics["episode_rewards"].append(total_reward)
            metrics["episode_steps"].append(steps)
            metrics["episode_latencies"].append(info.get("latency", 0))
            
            # Логирование в W&B
            if self.wandb_run_id:
                wandb.log({
                    "episode": episode,
                    "total_reward": total_reward,
                    "steps": steps,
                    "latency": info.get("latency", 0)
                })
        
        return metrics

    def save_model(self, file_name: str):
        """
        Сохранение модели
        
        Args:
            file_name: Путь для сохранения
        """
        torch.save({
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon
        }, file_name)
        
        if self.wandb_run_id:
            wandb.save(file_name)

    def load_model(self, file_name: str):
        """
        Загрузка модели
        
        Args:
            file_name: Путь к файлу модели
        """
        checkpoint = torch.load(file_name)
        self.network.load_state_dict(checkpoint["network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epsilon = checkpoint["epsilon"]
```

### 2. Регистрация агента

```python
from lwmecps_gym.ml.models.your_agent import YourAgent

class TrainingService:
    async def _run_training(self, task_id: str, task: TrainingTask):
        # ...
        if task.model_type == "your_agent":
            agent = YourAgent(env, **task.parameters)
        # ...
```

## Использование среды

### Состояние

Состояние среды включает:
- Информацию о нодах кластера
- Текущее размещение подов
- Метрики производительности

```python
state = env.reset()[0]
print(f"State shape: {state.shape}")
print(f"State: {state}")
```

### Действия

Доступные действия:
- Размещение пода на определенной ноде
- Масштабирование количества подов
- Перемещение подов между нодами

```python
action = env.action_space.sample()
print(f"Action: {action}")
```

### Награда

Награда включает:
- Задержку обработки запросов
- Использование ресурсов
- Баланс нагрузки

```python
next_state, reward, done, info = env.step(action)
print(f"Reward: {reward}")
print(f"Info: {info}")
```

## Метрики и логирование

### W&B метрики

```python
wandb.log({
    "episode": episode,
    "total_reward": total_reward,
    "steps": steps,
    "latency": info.get("latency", 0),
    "resource_usage": info.get("resource_usage", {}),
    "pod_count": info.get("pod_count", 0)
})
```

### Сохранение моделей

```python
# Локальное сохранение
agent.save_model("models/your_agent.pt")

# Сохранение в W&B
wandb.save("models/your_agent.pt")
```

## Примеры алгоритмов

### Q-Learning

```python
class QLearningAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.1
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        self.q_table = np.zeros((
            env.observation_space.n,
            env.action_space.n
        ))

    def choose_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state])

    def update_model(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        
        new_value = (1 - self.learning_rate) * old_value + \
            self.learning_rate * (reward + self.discount_factor * next_max)
        
        self.q_table[state, action] = new_value
```

### Deep Q-Network

```python
class DQNAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
        memory_size: int = 10000,
        batch_size: int = 32
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.memory_size = memory_size
        self.batch_size = batch_size
        
        self.memory = []
        self.network = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, env.action_space.n)
        )
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([x[0] for x in batch])
        actions = torch.LongTensor([x[1] for x in batch])
        rewards = torch.FloatTensor([x[2] for x in batch])
        next_states = torch.FloatTensor([x[3] for x in batch])
        dones = torch.FloatTensor([x[4] for x in batch])
        
        current_q = self.network(states).gather(1, actions.unsqueeze(1))
        next_q = self.network(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.discount_factor * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

### PPO

```python
class PPO:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.0003,
        discount_factor: float = 0.99,
        epsilon: float = 0.2,
        epochs: int = 10
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epochs = epochs
        
        self.actor = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, env.action_space.n),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate
        )

    def choose_action(self, state: np.ndarray) -> Tuple[int, float]:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state_tensor)
        action = torch.multinomial(action_probs, 1).item()
        return action, action_probs[0][action].item()

    def update_model(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        next_states: List[np.ndarray],
        dones: List[bool]
    ):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        for _ in range(self.epochs):
            # Вычисление преимуществ
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            advantages = rewards + (1 - dones) * self.discount_factor * next_values - values
            
            # Вычисление вероятностей действий
            action_probs = self.actor(states)
            old_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze()
            
            # Обновление политики
            for _ in range(10):
                action_probs = self.actor(states)
                new_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze()
                ratio = new_probs / old_probs
                
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                critic_loss = nn.MSELoss()(values, rewards + (1 - dones) * self.discount_factor * next_values)
                
                loss = actor_loss + 0.5 * critic_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
```

## Тестирование

### Unit тесты

```python
import unittest
import numpy as np
from lwmecps_gym.ml.models.your_agent import YourAgent
from lwmecps_gym.envs.lwmecps3 import LWMECPSEnv3

class TestYourAgent(unittest.TestCase):
    def setUp(self):
        self.env = LWMECPSEnv3()
        self.agent = YourAgent(self.env)

    def test_choose_action(self):
        state = self.env.reset()[0]
        action = self.agent.choose_action(state)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.env.action_space.n)

    def test_update_model(self):
        state = self.env.reset()[0]
        action = self.agent.choose_action(state)
        next_state, reward, done, _ = self.env.step(action)
        
        self.agent.update_model(state, action, reward, next_state, done)
        # Проверка обновления модели
```

### Интеграционные тесты

```python
import pytest
from lwmecps_gym.ml.training_service import TrainingService
from lwmecps_gym.core.models import TrainingTask

@pytest.mark.asyncio
async def test_training():
    service = TrainingService()
    task = TrainingTask(
        name="test-training",
        model_type="your_agent",
        parameters={
            "learning_rate": 0.001,
            "discount_factor": 0.99
        },
        total_episodes=100
    )
    
    task_id = await service.create_task(task)
    await service.start_training(task_id)
    
    # Проверка результатов обучения
```

## Оптимизация

### Гиперпараметры

```python
from ray import tune

def train_with_hyperparams(config):
    env = LWMECPSEnv3()
    agent = YourAgent(
        env,
        learning_rate=config["learning_rate"],
        discount_factor=config["discount_factor"],
        epsilon=config["epsilon"]
    )
    
    metrics = agent.train(episodes=1000)
    tune.report(
        mean_reward=np.mean(metrics["episode_rewards"]),
        mean_steps=np.mean(metrics["episode_steps"])
    )

analysis = tune.run(
    train_with_hyperparams,
    config={
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "discount_factor": tune.uniform(0.9, 0.999),
        "epsilon": tune.uniform(0.01, 0.3)
    },
    num_samples=100
)
```

### Производительность

```python
import cProfile
import pstats

def profile_training():
    env = LWMECPSEnv3()
    agent = YourAgent(env)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    agent.train(episodes=100)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats()
```

## Документация

### Docstrings

```python
def train(self, episodes: int) -> Dict[str, Any]:
    """
    Обучение агента
    
    Args:
        episodes (int): Количество эпизодов обучения
        
    Returns:
        Dict[str, Any]: Метрики обучения, включая:
            - episode_rewards (List[float]): Награды за каждый эпизод
            - episode_steps (List[int]): Количество шагов в каждом эпизоде
            - episode_latencies (List[float]): Задержки в каждом эпизоде
            
    Raises:
        ValueError: Если количество эпизодов меньше 1
    """
```

### Типы

```python
from typing import Dict, List, Tuple, Any, Optional

class YourAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
        wandb_run_id: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        pass
``` 
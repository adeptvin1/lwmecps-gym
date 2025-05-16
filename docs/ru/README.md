# LWMECPS Gym - Документация

## Содержание
1. [Обзор системы](#обзор-системы)
2. [Архитектура](#архитектура)
3. [Компоненты системы](#компоненты-системы)
4. [Разработка алгоритмов](#разработка-алгоритмов)
5. [API](#api)
6. [Развертывание](#развертывание)

## Обзор системы

LWMECPS Gym - это среда для обучения с подкреплением (Reinforcement Learning), предназначенная для оптимизации размещения сервисов в Kubernetes кластере. Система позволяет разрабатывать и тестировать алгоритмы машинного обучения для оптимизации размещения вычислительных сервисов в распределенных узлах обработки данных.

### Основные возможности
- Интеграция с Kubernetes для управления размещением сервисов
- Сбор метрик производительности и состояния кластера
- Отслеживание экспериментов через Weights & Biases
- Хранение данных в MongoDB
- REST API для управления обучением и мониторинга

## Архитектура

Система состоит из следующих основных компонентов:

1. **API слой** (`src/lwmecps_gym/api/`)
   - REST API для управления обучением
   - Эндпоинты для мониторинга и управления

2. **ML компоненты** (`src/lwmecps_gym/ml/`)
   - Реализации алгоритмов обучения
   - Сервис для управления обучением

3. **Core компоненты** (`src/lwmecps_gym/core/`)
   - Базовые модели данных
   - Конфигурация системы
   - Интеграция с W&B и MongoDB

4. **Environment** (`src/lwmecps_gym/envs/`)
   - Среда обучения с подкреплением
   - Интеграция с Kubernetes

## Компоненты системы

### API слой
API слой предоставляет REST интерфейс для взаимодействия с системой. Основные эндпоинты:
- `/api/v1/training` - управление обучением
- `/api/v1/models` - управление моделями
- `/api/v1/metrics` - получение метрик

### ML компоненты
ML компоненты отвечают за реализацию алгоритмов обучения. Основные классы:
- `TrainingService` - управление процессом обучения
- `QLearningAgent` - реализация Q-learning
- `DQNAgent` - реализация Deep Q-Network
- `PPO` - реализация Proximal Policy Optimization

### Core компоненты
Core компоненты содержат базовую функциональность:
- `models.py` - модели данных
- `database.py` - работа с MongoDB
- `wandb_config.py` - конфигурация W&B
- `config.py` - общая конфигурация

### Environment
Environment предоставляет среду для обучения:
- `LWMECPSEnv` - базовая среда
- `LWMECPSEnv2` - улучшенная версия
- `LWMECPSEnv3` - последняя версия
- `kubernetes_api.py` - интеграция с Kubernetes

## Разработка алгоритмов

### Создание нового алгоритма

1. Создайте новый файл в `src/lwmecps_gym/ml/models/`:
```python
from typing import Dict, Any
import gymnasium as gym
import wandb

class YourAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        wandb_run_id: str = None,
        **kwargs
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.wandb_run_id = wandb_run_id
        
        # Инициализация W&B
        if self.wandb_run_id:
            wandb.init(
                project="lwmecps-gym",
                id=self.wandb_run_id,
                config={
                    "learning_rate": learning_rate,
                    "discount_factor": discount_factor,
                    "model_type": "your_agent",
                    **kwargs
                }
            )

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
                # Ваша логика выбора действия
                action = self.choose_action(state)
                
                # Выполнение действия
                next_state, reward, done, info = self.env.step(action)
                
                # Обновление модели
                self.update_model(state, action, reward, next_state)
                
                # Сохранение метрик
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

    def choose_action(self, state):
        """
        Выбор действия на основе текущего состояния
        
        Args:
            state: Текущее состояние
            
        Returns:
            int: Выбранное действие
        """
        # Ваша логика выбора действия
        pass

    def update_model(self, state, action, reward, next_state):
        """
        Обновление модели на основе опыта
        
        Args:
            state: Текущее состояние
            action: Выполненное действие
            reward: Полученная награда
            next_state: Следующее состояние
        """
        # Ваша логика обновления модели
        pass

    def save_model(self, file_name: str):
        """
        Сохранение модели
        
        Args:
            file_name: Путь для сохранения
        """
        # Ваша логика сохранения модели
        pass

    def load_model(self, file_name: str):
        """
        Загрузка модели
        
        Args:
            file_name: Путь к файлу модели
        """
        # Ваша логика загрузки модели
        pass
```

2. Зарегистрируйте ваш алгоритм в `TrainingService`:
```python
from lwmecps_gym.ml.models.your_agent import YourAgent

class TrainingService:
    async def _run_training(self, task_id: str, task: TrainingTask):
        # ...
        if task.model_type == "your_agent":
            agent = YourAgent(env, **task.parameters)
        # ...
```

### Использование среды

Среда предоставляет следующие возможности:

1. **Состояние**:
   - Информация о нодах кластера
   - Текущее размещение подов
   - Метрики производительности

2. **Действия**:
   - Размещение пода на определенной ноде
   - Масштабирование количества подов
   - Перемещение подов между нодами

3. **Награда**:
   - Задержка обработки запросов
   - Использование ресурсов
   - Баланс нагрузки

### Метрики и логирование

1. **W&B метрики**:
   - Награда за эпизод
   - Количество шагов
   - Задержка
   - Использование ресурсов

2. **Сохранение моделей**:
   - Локальное сохранение
   - Сохранение в W&B как артефакт
   - Версионирование моделей

## API

### Эндпоинты

1. **Управление обучением**:
   ```http
   POST /api/v1/training
   {
     "name": "string",
     "model_type": "string",
     "parameters": {
       "learning_rate": 0.1,
       "discount_factor": 0.9
     },
     "total_episodes": 1000
   }
   ```

2. **Мониторинг**:
   ```http
   GET /api/v1/training/{task_id}/progress
   GET /api/v1/training/{task_id}/metrics
   ```

3. **Управление моделями**:
   ```http
   POST /api/v1/models/{model_id}/deploy
   GET /api/v1/models/{model_id}/metrics
   ```

## Развертывание

### Требования
- Kubernetes кластер
- Helm 3
- MongoDB
- Weights & Biases аккаунт

### Установка
1. Добавьте репозиторий:
   ```bash
   helm repo add lwmecps-gym https://your-repo-url
   ```

2. Установите chart:
   ```bash
   helm install lwmecps-gym lwmecps-gym/lwmecps-gym \
     --set mongodb.auth.rootPassword=your-password \
     --set wandb.apiKey=your-api-key
   ```

### Конфигурация
Основные параметры конфигурации:
- `mongodb.auth` - настройки MongoDB
- `wandb` - настройки W&B
- `training` - параметры обучения
- `kubernetes` - настройки кластера 
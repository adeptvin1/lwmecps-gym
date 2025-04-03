# LWMECPSEnv3

LWMECPSEnv3 - это среда Gymnasium для обучения агентов масштабированию микросервисов в Kubernetes кластере с использованием метрик задержки и количества пользователей из тестового приложения.

## Описание

LWMECPSEnv3 расширяет базовую среду Gymnasium и предоставляет интерфейс для:
- Масштабирования количества реплик подов
- Получения метрик задержки и количества пользователей
- Вычисления награды на основе текущего состояния

## Параметры инициализации

```python
env = LWMECPSEnv3(
    node_name: str,                    # Имя ноды
    max_hardware: Dict[str, float],    # Максимальные ресурсы ноды
    pod_usage: Dict[str, float],       # Использование ресурсов подом
    node_info: Dict[str, Dict[str, float]],  # Информация о нодах
    num_nodes: int,                    # Количество нод
    namespace: str,                    # Kubernetes namespace
    deployment_name: str,              # Имя deployment
    deployments: List[str],            # Список deployments
    max_pods: int,                     # Максимальное количество подов
    group_id: str,                     # ID группы экспериментов
    base_url: str = "http://localhost:8001"  # URL тестового приложения
)
```

## Пространство действий

Действия агента:
- 0: Уменьшить количество реплик
- 1: Оставить без изменений
- 2: Увеличить количество реплик

## Пространство наблюдений

Наблюдения включают:
- CPU использование (0-1)
- RAM использование (0-1)
- Входящая пропускная способность (0-1)
- Исходящая пропускная способность (0-1)
- Средняя задержка (0-∞)
- Количество пользователей (0-∞)
- Количество реплик (0-max_pods)

## Награда

Награда вычисляется на основе:
- Задержки (чем меньше, тем лучше)
- Количества пользователей (чем больше, тем лучше)
- Количества реплик (штраф за использование ресурсов)

Формула награды:
```
reward = (1000 / (latency + 1)) * (users / 100) - replicas * 10
```

## Методы

### reset()
- Запускает группу экспериментов
- Получает начальные метрики
- Сбрасывает количество реплик до 1
- Возвращает начальное состояние

### step(action)
- Выполняет действие масштабирования
- Получает обновленные метрики
- Вычисляет награду
- Возвращает новое состояние, награду и флаги завершения

### close()
- Очищает ресурсы среды

## Пример использования

```python
import gymnasium as gym
from lwmecps_gym.envs import LWMECPSEnv3

env = LWMECPSEnv3(
    node_name="node-1",
    max_hardware={"cpu": 4.0, "memory": 16.0},
    pod_usage={"cpu": 0.5, "memory": 1.0},
    node_info={"node-1": {"cpu": 4.0, "memory": 16.0}},
    num_nodes=1,
    namespace="default",
    deployment_name="test-app",
    deployments=["test-app"],
    max_pods=10,
    group_id="test-group"
)

observation, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
env.close()
``` 
# API Документация

## Общая информация

API LWMECPS Gym предоставляет REST интерфейс для управления обучением моделей, мониторинга и управления развернутыми моделями. Все эндпоинты доступны по базовому URL `/api/v1/`.

## Аутентификация

Все запросы к API должны включать заголовок `Authorization` с токеном доступа:

```http
Authorization: Bearer <your-token>
```

## Эндпоинты

### Управление обучением

#### Создание задачи обучения

```http
POST /api/v1/training
```

Создает новую задачу обучения.

**Тело запроса:**
```json
{
  "name": "string",
  "model_type": "string",
  "parameters": {
    "learning_rate": 0.1,
    "discount_factor": 0.9,
    "batch_size": 32,
    "epochs": 100
  },
  "total_episodes": 1000,
  "wandb_project": "string",
  "wandb_run_name": "string"
}
```

**Параметры:**
- `name` (string, required) - название задачи
- `model_type` (string, required) - тип модели (q_learning, dqn, ppo)
- `parameters` (object, required) - параметры модели
- `total_episodes` (integer, required) - количество эпизодов обучения
- `wandb_project` (string, optional) - название проекта в W&B
- `wandb_run_name` (string, optional) - название запуска в W&B

**Ответ:**
```json
{
  "task_id": "string",
  "status": "created",
  "created_at": "2024-03-20T12:00:00Z"
}
```

#### Получение статуса задачи

```http
GET /api/v1/training/{task_id}
```

Возвращает текущий статус задачи обучения.

**Параметры пути:**
- `task_id` (string, required) - ID задачи

**Ответ:**
```json
{
  "task_id": "string",
  "status": "running",
  "progress": 0.5,
  "current_episode": 500,
  "total_episodes": 1000,
  "metrics": {
    "episode_rewards": [1.0, 2.0, 3.0],
    "episode_steps": [100, 90, 80],
    "episode_latencies": [0.1, 0.09, 0.08]
  },
  "created_at": "2024-03-20T12:00:00Z",
  "updated_at": "2024-03-20T12:30:00Z"
}
```

#### Отмена задачи

```http
POST /api/v1/training/{task_id}/cancel
```

Отменяет выполнение задачи обучения.

**Параметры пути:**
- `task_id` (string, required) - ID задачи

**Ответ:**
```json
{
  "task_id": "string",
  "status": "cancelled",
  "updated_at": "2024-03-20T12:35:00Z"
}
```

### Управление моделями

#### Список моделей

```http
GET /api/v1/models
```

Возвращает список доступных моделей.

**Параметры запроса:**
- `page` (integer, optional) - номер страницы
- `per_page` (integer, optional) - количество моделей на странице
- `status` (string, optional) - фильтр по статусу (trained, deployed, archived)

**Ответ:**
```json
{
  "models": [
    {
      "model_id": "string",
      "name": "string",
      "type": "string",
      "status": "trained",
      "metrics": {
        "accuracy": 0.95,
        "latency": 0.1
      },
      "created_at": "2024-03-20T12:00:00Z"
    }
  ],
  "total": 100,
  "page": 1,
  "per_page": 10
}
```

#### Развертывание модели

```http
POST /api/v1/models/{model_id}/deploy
```

Развертывает модель в Kubernetes кластере.

**Параметры пути:**
- `model_id` (string, required) - ID модели

**Тело запроса:**
```json
{
  "replicas": 3,
  "resources": {
    "cpu": "1",
    "memory": "1Gi"
  },
  "strategy": "rolling"
}
```

**Ответ:**
```json
{
  "model_id": "string",
  "status": "deploying",
  "deployment": {
    "name": "string",
    "namespace": "string",
    "replicas": 3
  },
  "created_at": "2024-03-20T12:00:00Z"
}
```

#### Получение метрик модели

```http
GET /api/v1/models/{model_id}/metrics
```

Возвращает метрики производительности развернутой модели.

**Параметры пути:**
- `model_id` (string, required) - ID модели

**Параметры запроса:**
- `start_time` (string, optional) - начальное время в формате ISO 8601
- `end_time` (string, optional) - конечное время в формате ISO 8601
- `interval` (string, optional) - интервал агрегации (1m, 5m, 1h)

**Ответ:**
```json
{
  "model_id": "string",
  "metrics": {
    "latency": {
      "values": [0.1, 0.09, 0.08],
      "timestamps": ["2024-03-20T12:00:00Z", "2024-03-20T12:01:00Z", "2024-03-20T12:02:00Z"]
    },
    "throughput": {
      "values": [100, 110, 120],
      "timestamps": ["2024-03-20T12:00:00Z", "2024-03-20T12:01:00Z", "2024-03-20T12:02:00Z"]
    },
    "resource_usage": {
      "cpu": {
        "values": [0.5, 0.6, 0.7],
        "timestamps": ["2024-03-20T12:00:00Z", "2024-03-20T12:01:00Z", "2024-03-20T12:02:00Z"]
      },
      "memory": {
        "values": [0.4, 0.45, 0.5],
        "timestamps": ["2024-03-20T12:00:00Z", "2024-03-20T12:01:00Z", "2024-03-20T12:02:00Z"]
      }
    }
  }
}
```

### Мониторинг

#### Статус системы

```http
GET /api/v1/health
```

Возвращает статус системы и ее компонентов.

**Ответ:**
```json
{
  "status": "healthy",
  "components": {
    "api": "healthy",
    "database": "healthy",
    "kubernetes": "healthy",
    "wandb": "healthy"
  },
  "version": "1.0.0",
  "uptime": "24h"
}
```

#### Метрики системы

```http
GET /api/v1/metrics
```

Возвращает метрики системы.

**Параметры запроса:**
- `start_time` (string, optional) - начальное время в формате ISO 8601
- `end_time` (string, optional) - конечное время в формате ISO 8601
- `interval` (string, optional) - интервал агрегации (1m, 5m, 1h)

**Ответ:**
```json
{
  "system": {
    "cpu_usage": {
      "values": [0.5, 0.6, 0.7],
      "timestamps": ["2024-03-20T12:00:00Z", "2024-03-20T12:01:00Z", "2024-03-20T12:02:00Z"]
    },
    "memory_usage": {
      "values": [0.4, 0.45, 0.5],
      "timestamps": ["2024-03-20T12:00:00Z", "2024-03-20T12:01:00Z", "2024-03-20T12:02:00Z"]
    }
  },
  "kubernetes": {
    "node_count": 3,
    "pod_count": 10,
    "resource_usage": {
      "cpu": 0.6,
      "memory": 0.5
    }
  },
  "training": {
    "active_tasks": 2,
    "completed_tasks": 100,
    "average_training_time": "2h"
  }
}
```

## Коды ошибок

| Код | Описание |
|-----|----------|
| 400 | Неверный запрос |
| 401 | Не авторизован |
| 403 | Доступ запрещен |
| 404 | Ресурс не найден |
| 409 | Конфликт |
| 422 | Неверные данные |
| 500 | Внутренняя ошибка сервера |

## Примеры использования

### Python

```python
import requests

API_URL = "http://localhost:8000/api/v1"
TOKEN = "your-token"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

# Создание задачи обучения
response = requests.post(
    f"{API_URL}/training",
    headers=headers,
    json={
        "name": "test-training",
        "model_type": "dqn",
        "parameters": {
            "learning_rate": 0.001,
            "discount_factor": 0.99
        },
        "total_episodes": 1000
    }
)
task_id = response.json()["task_id"]

# Получение статуса задачи
response = requests.get(
    f"{API_URL}/training/{task_id}",
    headers=headers
)
status = response.json()["status"]

# Развертывание модели
response = requests.post(
    f"{API_URL}/models/{model_id}/deploy",
    headers=headers,
    json={
        "replicas": 3,
        "resources": {
            "cpu": "1",
            "memory": "1Gi"
        }
    }
)
```

### cURL

```bash
# Создание задачи обучения
curl -X POST "http://localhost:8000/api/v1/training" \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test-training",
    "model_type": "dqn",
    "parameters": {
      "learning_rate": 0.001,
      "discount_factor": 0.99
    },
    "total_episodes": 1000
  }'

# Получение статуса задачи
curl -X GET "http://localhost:8000/api/v1/training/task-id" \
  -H "Authorization: Bearer your-token"

# Развертывание модели
curl -X POST "http://localhost:8000/api/v1/models/model-id/deploy" \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{
    "replicas": 3,
    "resources": {
      "cpu": "1",
      "memory": "1Gi"
    }
  }'
``` 
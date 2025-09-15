# Meta-Learning в LWMECPS Gym

## Обзор

LWMECPS Gym теперь поддерживает мета-обучение (meta-learning) - подход к машинному обучению, который позволяет алгоритмам быстро адаптироваться к новым задачам, используя знания, полученные при обучении на множестве связанных задач.

## Поддерживаемые алгоритмы

### Базовые алгоритмы с мета-обучением
- **Meta-PPO** - Proximal Policy Optimization с мета-обучением
- **Meta-SAC** - Soft Actor-Critic с мета-обучением  
- **Meta-TD3** - Twin Delayed Deep Deterministic Policy Gradient с мета-обучением
- **Meta-DQN** - Deep Q-Network с мета-обучением

### Методы мета-обучения
- **MAML** (Model-Agnostic Meta-Learning) - универсальный подход, который изучает хорошие начальные параметры
- **FOMAML** (First-Order MAML) - упрощенная версия MAML, использующая только градиенты первого порядка

## Архитектура

### Базовые классы

```python
# Базовый класс для мета-обучения
BaseMetaLearning
├── MAML          # Реализация MAML
├── FOMAML        # Реализация FOMAML
└── MetaLearningWrapper  # Обертка для существующих алгоритмов
    ├── MetaPPO   # PPO с мета-обучением
    ├── MetaSAC   # SAC с мета-обучением
    ├── MetaTD3   # TD3 с мета-обучением
    └── MetaDQN   # DQN с мета-обучением
```

### Принцип работы

1. **Мета-обучение**: Алгоритм обучается на множестве задач, изучая хорошие начальные параметры
2. **Адаптация**: При появлении новой задачи, алгоритм быстро адаптируется за несколько шагов градиентного спуска
3. **Обобщение**: Изученные параметры позволяют эффективно работать на новых, но похожих задачах

## API Эндпоинты

### Создание задачи мета-обучения

```http
POST /meta-learning/meta-tasks
Content-Type: application/json

{
  "name": "Meta-PPO Training",
  "description": "Meta-learning training using PPO with MAML",
  "model_type": "meta_ppo",
  "meta_method": "maml",
  "tasks": [
    {
      "task_id": "task_1",
      "name": "High CPU Task",
      "node_name": ["node1", "node2", "node3"],
      "max_hardware": {...},
      "pod_usage": {...},
      "episodes": 50
    },
    // ... другие задачи
  ],
  "meta_parameters": {
    "meta_lr": 0.01,
    "inner_lr": 0.01,
    "num_inner_steps": 1,
    "num_meta_epochs": 100
  },
  "parameters": {
    "hidden_size": 64,
    "learning_rate": 3e-4,
    // ... другие параметры базового алгоритма
  }
}
```

### Запуск мета-обучения

```http
POST /meta-learning/meta-tasks/{task_id}/start
```

### Мониторинг прогресса

```http
GET /meta-learning/meta-tasks/{task_id}/progress
```

### Адаптация к новой задаче

```http
POST /meta-learning/meta-tasks/{task_id}/adapt
Content-Type: application/json

{
  "node_name": ["node1", "node2", "node3"],
  "max_hardware": {...},
  "pod_usage": {...},
  "adaptation_episodes": 20
}
```

## Пример использования

### 1. Создание задачи мета-обучения

```python
import requests

# Конфигурация задач для мета-обучения
tasks = [
    {
        "task_id": "high_cpu_task",
        "name": "High CPU Task",
        "node_name": ["node1", "node2", "node3"],
        "max_hardware": {
            "cpu": 8, "ram": 16000, "tx_bandwidth": 1000,
            "rx_bandwidth": 1000, "avg_latency": 300
        },
        "pod_usage": {
            "cpu": 4, "ram": 4000, "tx_bandwidth": 40,
            "rx_bandwidth": 40
        },
        "episodes": 50
    },
    {
        "task_id": "high_memory_task", 
        "name": "High Memory Task",
        "node_name": ["node1", "node2", "node3"],
        "max_hardware": {
            "cpu": 8, "ram": 16000, "tx_bandwidth": 1000,
            "rx_bandwidth": 1000, "avg_latency": 300
        },
        "pod_usage": {
            "cpu": 2, "ram": 6000, "tx_bandwidth": 20,
            "rx_bandwidth": 20
        },
        "episodes": 50
    }
]

# Создание задачи мета-обучения
task_data = {
    "name": "Meta-PPO Training",
    "description": "Meta-learning training using PPO with MAML",
    "model_type": "meta_ppo",
    "meta_method": "maml",
    "tasks": tasks,
    "meta_parameters": {
        "meta_lr": 0.01,
        "inner_lr": 0.01,
        "num_inner_steps": 1,
        "num_meta_epochs": 100
    },
    "parameters": {
        "hidden_size": 64,
        "learning_rate": 3e-4,
        "discount_factor": 0.99,
        "batch_size": 64
    }
}

response = requests.post(
    "http://localhost:8000/meta-learning/meta-tasks",
    json=task_data
)
task = response.json()
task_id = task["id"]
```

### 2. Запуск мета-обучения

```python
# Запуск обучения
response = requests.post(
    f"http://localhost:8000/meta-learning/meta-tasks/{task_id}/start"
)
```

### 3. Мониторинг прогресса

```python
# Проверка прогресса
response = requests.get(
    f"http://localhost:8000/meta-learning/meta-tasks/{task_id}/progress"
)
progress = response.json()

print(f"State: {progress['state']}")
print(f"Meta Method: {progress['meta_method']}")
print(f"Number of Tasks: {progress['num_tasks']}")
```

### 4. Адаптация к новой задаче

```python
# Новая задача для адаптации
new_task_config = {
    "node_name": ["node1", "node2", "node3"],
    "max_hardware": {
        "cpu": 8, "ram": 16000, "tx_bandwidth": 1000,
        "rx_bandwidth": 1000, "avg_latency": 200
    },
    "pod_usage": {
        "cpu": 3, "ram": 3000, "tx_bandwidth": 30,
        "rx_bandwidth": 30
    },
    "adaptation_episodes": 20
}

# Адаптация к новой задаче
response = requests.post(
    f"http://localhost:8000/meta-learning/meta-tasks/{task_id}/adapt",
    json=new_task_config
)
adaptation_result = response.json()

print(f"Adaptation Metrics: {adaptation_result['adaptation_metrics']}")
```

## Параметры мета-обучения

### Общие параметры

- `meta_lr` (float): Скорость обучения для мета-обновлений (по умолчанию: 0.01)
- `inner_lr` (float): Скорость обучения для внутренних обновлений (по умолчанию: 0.01)
- `num_inner_steps` (int): Количество шагов градиентного спуска во внутреннем цикле (по умолчанию: 1)
- `num_meta_epochs` (int): Количество эпох мета-обучения (по умолчанию: 100)

### Параметры базовых алгоритмов

Каждый базовый алгоритм поддерживает свои специфические параметры:

#### Meta-PPO
- `hidden_size`: Размер скрытых слоев
- `learning_rate`: Скорость обучения
- `gamma`: Коэффициент дисконтирования
- `clip_epsilon`: Параметр обрезки для PPO
- `entropy_coef`: Коэффициент энтропии
- `value_function_coef`: Коэффициент функции ценности

#### Meta-SAC
- `hidden_size`: Размер скрытых слоев
- `learning_rate`: Скорость обучения
- `gamma`: Коэффициент дисконтирования
- `tau`: Коэффициент мягкого обновления
- `alpha`: Коэффициент энтропии
- `auto_entropy`: Автоматическая настройка энтропии

#### Meta-TD3
- `hidden_size`: Размер скрытых слоев
- `learning_rate`: Скорость обучения
- `gamma`: Коэффициент дисконтирования
- `tau`: Коэффициент мягкого обновления
- `policy_delay`: Задержка обновления политики
- `noise_clip`: Обрезка шума

#### Meta-DQN
- `learning_rate`: Скорость обучения
- `discount_factor`: Коэффициент дисконтирования
- `epsilon`: Скорость исследования
- `memory_size`: Размер буфера воспроизведения
- `batch_size`: Размер батча

## Преимущества мета-обучения

### 1. Быстрая адаптация
- Модель может быстро адаптироваться к новым задачам за несколько шагов градиентного спуска
- Значительно сокращает время обучения для новых задач

### 2. Лучшая эффективность выборки
- Требует меньше данных для достижения хорошей производительности на новых задачах
- Более эффективное использование имеющихся данных

### 3. Улучшенная обобщающая способность
- Изученные параметры позволяют лучше работать на новых, но похожих задачах
- Снижает риск переобучения

### 4. Гибкость
- Один и тот же подход мета-обучения работает с разными базовыми алгоритмами
- Можно комбинировать разные методы мета-обучения с разными алгоритмами

## Когда использовать мета-обучение

### Подходящие случаи:
- Множество связанных задач с похожей структурой
- Необходимость быстрой адаптации к новым средам
- Ограниченные данные для новых задач
- Желание улучшить обобщающую способность

### Неподходящие случаи:
- Задачи сильно отличаются друг от друга
- Достаточно данных для обучения с нуля
- Простые задачи, не требующие сложной адаптации

## Мониторинг и отладка

### Метрики мета-обучения

- `meta_loss`: Потеря мета-обучения
- `avg_task_loss`: Средняя потеря по задачам
- `adaptation_loss`: Потеря при адаптации
- `test_avg_reward`: Средняя награда при тестировании
- `test_avg_length`: Средняя длина эпизода при тестировании

### Логирование

Все метрики автоматически логируются в Weights & Biases для мониторинга и анализа.

## Ограничения и рекомендации

### Ограничения:
- Требует больше вычислительных ресурсов для начального обучения
- Сложность настройки гиперпараметров
- Может быть избыточным для простых задач

### Рекомендации:
- Начните с FOMAML для более быстрого обучения
- Используйте MAML для лучшей производительности
- Тщательно подбирайте параметры мета-обучения
- Мониторьте прогресс обучения

## Заключение

Мета-обучение в LWMECPS Gym предоставляет мощный инструмент для создания алгоритмов, которые могут быстро адаптироваться к новым задачам. Это особенно полезно в контексте оптимизации размещения сервисов в Kubernetes, где условия могут часто изменяться, и требуется быстрая адаптация к новым средам и требованиям.

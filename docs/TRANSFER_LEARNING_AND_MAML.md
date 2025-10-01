# Transfer Learning и Meta-Learning в LWMECPS GYM

## 🎯 Обзор

LWMECPS GYM теперь поддерживает передовые алгоритмы машинного обучения для быстрой адаптации к новым задачам:

- **Transfer Learning** - перенос знаний между похожими задачами
- **MAML (Model-Agnostic Meta-Learning)** - обучение моделей, способных быстро адаптироваться
- **FOMAML (First-Order MAML)** - эффективная версия MAML

## 🔄 Transfer Learning

### Концепция

Transfer Learning позволяет использовать знания, полученные при решении одной задачи, для решения похожей задачи. В контексте Kubernetes это означает:

- Адаптация модели, обученной на одном кластере, к другому кластеру
- Перенос стратегий масштабирования между разными приложениями
- Использование опыта работы с одной нагрузкой для другой

### Типы Transfer Learning

#### 1. Feature Extraction
```python
{
    "transfer_type": "feature_extraction",
    "frozen_layers": [0, 1, 2],  # Заморозить все слои кроме последнего
    "learning_rate": 1e-4
}
```

#### 2. Fine-tuning
```python
{
    "transfer_type": "fine_tuning",
    "frozen_layers": [],  # Не замораживать слои
    "learning_rate": 1e-4
}
```

#### 3. Layer-wise Training
```python
{
    "transfer_type": "layer_wise",
    "frozen_layers": [0, 1],  # Постепенно размораживать слои
    "learning_rate": 1e-4
}
```

### API Endpoints

#### Создание Transfer Learning задачи
```bash
POST /api/training/transfer-tasks
```

```json
{
    "name": "Transfer Learning Task",
    "description": "Адаптация PPO модели к новой среде",
    "source_task_id": "64f1b2a3c9d4e5f6a7b8c9d0",
    "target_task_id": "64f1b2a3c9d4e5f6a7b8c9d1",
    "transfer_type": "fine_tuning",
    "frozen_layers": [0, 1],
    "learning_rate": 1e-4,
    "total_episodes": 50
}
```

#### Запуск обучения
```bash
POST /api/training/transfer-tasks/{task_id}/start
```

#### Получение результатов
```bash
GET /api/training/transfer-tasks/{task_id}/results
```

### Метрики Transfer Learning

- **transfer_metric** - метрика переноса знаний
- **parameter_distance** - расстояние между параметрами моделей
- **adaptation_speed** - скорость адаптации

## 🧠 Meta-Learning (MAML/FOMAML)

### Концепция

Meta-Learning обучает модели "учиться учиться" - то есть быстро адаптироваться к новым задачам с минимальным количеством примеров.

### MAML (Model-Agnostic Meta-Learning)

MAML находит хорошие начальные параметры, которые можно быстро адаптировать к новым задачам за несколько шагов градиентного спуска.

#### Алгоритм MAML:
1. **Meta-training**: Обучение на множестве задач
2. **Inner loop**: Адаптация к каждой задаче
3. **Outer loop**: Обновление meta-параметров
4. **Meta-testing**: Быстрая адаптация к новым задачам

### FOMAML (First-Order MAML)

FOMAML - это вычислительно эффективная версия MAML, которая использует только первые производные, избегая дорогих вычислений вторых производных.

#### Преимущества FOMAML:
- ⚡ Быстрее MAML в 2-3 раза
- 💾 Меньше потребление памяти
- 🎯 Сопоставимая точность адаптации

### API Endpoints

#### Создание Meta-Learning задачи
```bash
POST /api/training/meta-tasks
```

```json
{
    "name": "MAML Meta-Learning",
    "description": "Обучение модели для быстрой адаптации",
    "meta_algorithm": "maml",  // или "fomaml", "implicit_fomaml"
    "inner_lr": 0.01,
    "outer_lr": 0.001,
    "adaptation_steps": 5,
    "meta_batch_size": 4,
    "task_distribution": {
        "kubernetes_scaling": {
            "max_pods": 20,
            "cpu_requirement": 1.0,
            "memory_requirement": 1000
        },
        "load_balancing": {
            "traffic_pattern": "uniform",
            "latency_threshold": 0.5
        }
    },
    "total_episodes": 100
}
```

#### Запуск Meta-Learning
```bash
POST /api/training/meta-tasks/{task_id}/start
```

#### Получение результатов
```bash
GET /api/training/meta-tasks/{task_id}/results
```

### Метрики Meta-Learning

- **meta_loss** - потеря meta-обучения
- **adaptation_accuracy** - точность адаптации
- **task_performance** - производительность на задачах
- **gradient_norm** - норма градиента (для FOMAML)
- **adaptation_speed** - скорость адаптации

## 🚀 Практические примеры

### Пример 1: Transfer Learning между кластерами

```python
# 1. Обучить модель на кластере A
source_task = {
    "name": "Cluster A Training",
    "model_type": "ppo",
    "env_config": {"cluster": "cluster-a"},
    "total_episodes": 100
}

# 2. Адаптировать к кластеру B
transfer_task = {
    "source_task_id": source_task_id,
    "target_task_id": target_task_id,
    "transfer_type": "fine_tuning",
    "learning_rate": 1e-4
}
```

### Пример 2: MAML для быстрой адаптации

```python
# 1. Meta-обучение на множестве задач
meta_task = {
    "meta_algorithm": "fomaml",
    "inner_lr": 0.01,
    "outer_lr": 0.001,
    "adaptation_steps": 5,
    "total_episodes": 100
}

# 2. Быстрая адаптация к новой задаче
# Модель адаптируется за 5 шагов вместо 100 эпизодов
```

## 📊 Сравнение алгоритмов

| Алгоритм | Скорость обучения | Точность адаптации | Вычислительная сложность | Применение |
|----------|-------------------|-------------------|-------------------------|------------|
| **Transfer Learning** | Быстро | Высокая | Низкая | Похожие задачи |
| **MAML** | Медленно | Очень высокая | Высокая | Новые задачи |
| **FOMAML** | Средне | Высокая | Средняя | Компромисс |
| **Implicit FOMAML** | Средне | Очень высокая | Средняя | Точность + скорость |

## 🎯 Рекомендации по использованию

### Когда использовать Transfer Learning:
- ✅ У вас есть обученная модель на похожей задаче
- ✅ Новая задача имеет схожие характеристики
- ✅ Ограниченные вычислительные ресурсы
- ✅ Нужна быстрая адаптация

### Когда использовать MAML/FOMAML:
- ✅ Много разнообразных задач
- ✅ Нужна быстрая адаптация к новым задачам
- ✅ Достаточно вычислительных ресурсов для meta-обучения
- ✅ Важна обобщающая способность

### Выбор алгоритма:
- **MAML**: Максимальная точность, много ресурсов
- **FOMAML**: Компромисс точности и скорости
- **Implicit FOMAML**: Лучший компромисс для продакшена

## 🔧 Конфигурация

### Переменные окружения

```bash
# Weights & Biases
WANDB_API_KEY=your_api_key
WANDB_PROJECT=lwmecps-gym
WANDB_ENTITY=your_entity

# MongoDB
MONGODB_URL=mongodb://localhost:27017
MONGODB_DATABASE=lwmecps_gym

# Kubernetes
KUBECONFIG=/path/to/kubeconfig
```

### Параметры производительности

```python
# Для Transfer Learning
TRANSFER_LEARNING_CONFIG = {
    "learning_rate": 1e-4,
    "frozen_layers": [0, 1],
    "total_episodes": 50
}

# Для MAML
MAML_CONFIG = {
    "inner_lr": 0.01,
    "outer_lr": 0.001,
    "adaptation_steps": 5,
    "meta_batch_size": 4
}

# Для FOMAML
FOMAML_CONFIG = {
    "inner_lr": 0.01,
    "outer_lr": 0.001,
    "adaptation_steps": 5,
    "meta_batch_size": 6,  # Больший batch size
    "use_implicit_gradients": True
}
```

## 📈 Мониторинг и отладка

### Weights & Biases метрики

```python
# Transfer Learning метрики
wandb.log({
    "transfer/episode_reward": reward,
    "transfer/transfer_metric": transfer_metric,
    "transfer/parameter_distance": distance
})

# MAML метрики
wandb.log({
    "maml/meta_loss": meta_loss,
    "maml/adaptation_accuracy": accuracy,
    "maml/task_performance": performance
})

# FOMAML метрики
wandb.log({
    "fomaml/meta_loss": meta_loss,
    "fomaml/gradient_norm": gradient_norm,
    "fomaml/adaptation_speed": speed
})
```

### Логирование

```python
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# В коде
logger.info(f"Transfer learning progress: {progress:.2f}%")
logger.info(f"MAML meta-episode {episode}: loss={loss:.4f}")
```

## 🚨 Устранение неполадок

### Частые проблемы

1. **Ошибка загрузки модели**
   ```
   ValueError: Model file not found
   ```
   **Решение**: Убедитесь, что исходная модель обучена и сохранена

2. **Медленная адаптация**
   ```
   Low adaptation accuracy
   ```
   **Решение**: Увеличьте количество adaptation_steps или inner_lr

3. **Высокий meta-loss**
   ```
   Meta-loss not decreasing
   ```
   **Решение**: Уменьшите outer_lr или увеличьте meta_batch_size

### Отладка

```python
# Проверка состояния задач
response = requests.get(f"{API_BASE_URL}/transfer-tasks/{task_id}")
task = response.json()
print(f"State: {task['state']}")
print(f"Progress: {task['progress']}%")
print(f"Error: {task.get('error_message', 'None')}")
```

## 📚 Дополнительные ресурсы

- [MAML Paper](https://arxiv.org/abs/1703.03400)
- [FOMAML Paper](https://arxiv.org/abs/1803.02999)
- [Transfer Learning Survey](https://arxiv.org/abs/1912.01703)
- [Weights & Biases Documentation](https://docs.wandb.ai/)

## 🤝 Вклад в проект

Мы приветствуем вклад в развитие Transfer Learning и Meta-Learning функциональности:

1. Новые алгоритмы meta-learning
2. Улучшения производительности
3. Дополнительные метрики
4. Примеры использования
5. Документация

Создавайте issues и pull requests для предложения улучшений!

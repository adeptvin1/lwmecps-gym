# Изменение размерности сети в мета-обучении

## Обзор

LWMECPS Gym поддерживает динамическое изменение размерности сети (количество нод) для мета-обученных моделей. Это позволяет адаптировать модели к изменяющимся размерам кластера без потери изученных знаний.

## Проблема

В динамических Kubernetes кластерах количество нод может изменяться:
- **Масштабирование вверх**: Добавление новых нод для увеличения производительности
- **Масштабирование вниз**: Удаление нод для экономии ресурсов
- **Автомасштабирование**: Автоматическое изменение размера кластера
- **Обслуживание**: Временное отключение нод для обслуживания

### Традиционные проблемы:
1. **Фиксированная архитектура**: Нейронные сети имеют фиксированный размер входного слоя
2. **Потеря знаний**: При изменении размерности теряются изученные паттерны
3. **Необходимость переобучения**: Требуется полное переобучение с нуля
4. **Время простоя**: Модель не может работать во время переобучения

## Решение

### Адаптивная архитектура

Созданы адаптивные нейронные сети, которые могут изменять свой размер:

```python
class AdaptiveActorCritic(AdaptiveNetwork):
    """
    Адаптивная Actor-Critic сеть для PPO.
    
    Может изменять количество нод, сохраняя изученные паттерны.
    """
    
    def adapt_to_nodes(self, num_nodes: int, strategy: str = 'weight_interpolation'):
        """Адаптирует сеть к новому количеству нод."""
        # Логика адаптации архитектуры
        pass
```

### Стратегии адаптации

#### 1. **Zero Padding** (Нулевое заполнение)
- **Сложность**: Низкая
- **Лучше всего для**: Малых изменений (1-2 ноды)
- **Принцип**: Добавляет нулевые веса для новых нод
- **Преимущества**: Быстро, просто, не теряет данные
- **Недостатки**: Может не сохранять изученные паттерны

```python
def _zero_padding_strategy(self, old_params, new_num_nodes):
    """Добавляет нулевые веса для новых нод."""
    if new_num_nodes > old_size:
        padding = torch.zeros(padding_size, param.shape[1])
        new_params[name] = torch.cat([param, padding], dim=0)
```

#### 2. **Weight Interpolation** (Интерполяция весов)
- **Сложность**: Средняя
- **Лучше всего для**: Средних изменений (3-5 нод)
- **Принцип**: Интерполирует веса между существующими нодами
- **Преимущества**: Сохраняет изученные паттерны, плавный переход
- **Недостатки**: Сложнее нулевого заполнения

```python
def _interpolate_weights_2d(self, weights, old_size, new_size):
    """Интерполирует 2D веса для новой размерности."""
    old_indices = torch.linspace(0, old_size - 1, old_size)
    new_indices = torch.linspace(0, old_size - 1, new_size)
    
    for i in range(weights.shape[1]):
        interpolated[:, i] = torch.interp(new_indices, old_indices, weights[:, i])
```

#### 3. **Knowledge Distillation** (Дистилляция знаний)
- **Сложность**: Высокая
- **Лучше всего для**: Больших изменений (5+ нод)
- **Принцип**: Использует дистилляцию знаний для передачи информации
- **Преимущества**: Сохраняет сложные паттерны, работает для больших изменений
- **Недостатки**: Вычислительно дорого, требует больше времени

```python
def _distill_weights_2d(self, weights, old_size, new_size):
    """Дистиллирует 2D веса для новой размерности."""
    # Копируем существующие веса
    new_weights[:copy_size] = weights[:copy_size]
    
    # Для дополнительных нод используем усредненные веса
    if new_size > old_size:
        avg_weights = weights.mean(dim=0, keepdim=True)
        for i in range(old_size, new_size):
            noise = torch.randn_like(avg_weights) * 0.1
            new_weights[i] = avg_weights.squeeze(0) + noise.squeeze(0)
```

#### 4. **Attention-Based** (На основе внимания)
- **Сложность**: Высокая
- **Лучше всего для**: Сложных изменений с различными характеристиками нод
- **Принцип**: Использует механизм внимания для адаптации
- **Преимущества**: Обрабатывает сложные отношения, адаптивен к различиям нод
- **Недостатки**: Наиболее сложный, требует значительных вычислений

```python
def _create_attention_weights_2d(self, weights, old_size, new_size):
    """Создает веса на основе механизма внимания."""
    attention_matrix = torch.softmax(
        torch.randn(new_size, old_size), dim=1
    )
    new_weights = torch.mm(attention_matrix, weights)
```

## API для изменения размерности

### Создание адаптивной задачи мета-обучения

```http
POST /meta-learning/meta-tasks
Content-Type: application/json

{
  "name": "Adaptive Meta-PPO Training",
  "model_type": "meta_ppo",
  "meta_method": "maml",
  "tasks": [...],
  "parameters": {
    "max_nodes": 20,
    "initial_nodes": 3,
    "hidden_size": 64
  }
}
```

### Адаптация к новому количеству нод

```http
POST /meta-learning/meta-tasks/{task_id}/adapt-nodes
Content-Type: application/json

{
  "new_num_nodes": 5,
  "strategy": "weight_interpolation",
  "node_info": {
    "node4": {"cpu": 8, "ram": 16000, "avg_latency": 25},
    "node5": {"cpu": 8, "ram": 16000, "avg_latency": 30}
  },
  "adaptation_episodes": 10
}
```

### Получение истории изменений архитектуры

```http
GET /meta-learning/meta-tasks/{task_id}/architecture-history
```

### Получение текущего количества нод

```http
GET /meta-learning/meta-tasks/{task_id}/current-nodes
```

### Получение поддерживаемых стратегий

```http
GET /meta-learning/supported-scaling-strategies
```

## Примеры использования

### 1. Базовое изменение размерности

```python
import requests

# Создаем адаптивную задачу мета-обучения
task_data = {
    "name": "Adaptive Meta-PPO Training",
    "model_type": "meta_ppo",
    "meta_method": "maml",
    "parameters": {
        "max_nodes": 20,
        "initial_nodes": 3
    },
    "tasks": [...]
}

# Создаем задачу
response = requests.post("http://localhost:8000/meta-learning/meta-tasks", json=task_data)
task_id = response.json()["id"]

# Запускаем мета-обучение
requests.post(f"http://localhost:8000/meta-learning/meta-tasks/{task_id}/start")

# Адаптируем к 5 нодам
scaling_config = {
    "new_num_nodes": 5,
    "strategy": "weight_interpolation",
    "node_info": {...}
}

response = requests.post(
    f"http://localhost:8000/meta-learning/meta-tasks/{task_id}/adapt-nodes",
    json=scaling_config
)
```

### 2. Последовательное изменение размерности

```python
# Начинаем с 3 нод
current_nodes = 3

# Масштабируем до 7 нод
for target_nodes in [5, 7]:
    scaling_config = {
        "new_num_nodes": target_nodes,
        "strategy": "weight_interpolation" if target_nodes - current_nodes <= 2 else "knowledge_distillation",
        "node_info": get_node_info(target_nodes)
    }
    
    response = requests.post(
        f"http://localhost:8000/meta-learning/meta-tasks/{task_id}/adapt-nodes",
        json=scaling_config
    )
    
    current_nodes = target_nodes
    print(f"Scaled to {target_nodes} nodes")
```

### 3. Выбор стратегии на основе изменения

```python
def choose_scaling_strategy(old_nodes, new_nodes):
    """Выбирает стратегию на основе размера изменения."""
    change = abs(new_nodes - old_nodes)
    
    if change <= 2:
        return "zero_padding"
    elif change <= 5:
        return "weight_interpolation"
    elif change <= 10:
        return "knowledge_distillation"
    else:
        return "attention_based"

# Используем автоматический выбор стратегии
strategy = choose_scaling_strategy(current_nodes, target_nodes)
```

## Мониторинг и отладка

### Метрики адаптации

- `adaptation_success`: Успешность адаптации
- `old_nodes`: Предыдущее количество нод
- `new_nodes`: Новое количество нод
- `strategy`: Использованная стратегия
- `adaptation_time`: Время адаптации
- `performance_retention`: Сохранение производительности

### Логирование

```python
# История изменений архитектуры
{
    "timestamp": 1640995200.0,
    "old_nodes": 3,
    "new_nodes": 5,
    "strategy": "weight_interpolation",
    "adaptation_info": {
        "success": True,
        "adaptation_time": 2.5,
        "performance_retention": 0.95
    }
}
```

## Рекомендации по использованию

### Выбор стратегии

| Изменение нод | Рекомендуемая стратегия | Обоснование |
|---------------|------------------------|-------------|
| 1-2 ноды | Zero Padding | Быстро и просто |
| 3-5 нод | Weight Interpolation | Хороший баланс скорости и качества |
| 5-10 нод | Knowledge Distillation | Сохраняет сложные паттерны |
| 10+ нод | Attention-Based | Максимальная адаптивность |

### Лучшие практики

1. **Планируйте заранее**: Устанавливайте `max_nodes` с запасом
2. **Мониторьте производительность**: Отслеживайте метрики после адаптации
3. **Тестируйте стратегии**: Выберите оптимальную стратегию для вашего случая
4. **Сохраняйте историю**: Ведите лог изменений архитектуры
5. **Градуальное масштабирование**: Избегайте резких изменений размерности

### Ограничения

1. **Максимальное количество нод**: Ограничено параметром `max_nodes`
2. **Вычислительная сложность**: Некоторые стратегии требуют больше ресурсов
3. **Качество адаптации**: Зависит от выбранной стратегии
4. **Время адаптации**: Может потребоваться время для адаптации

## Заключение

Изменение размерности сети в мета-обучении позволяет создавать гибкие и адаптивные системы, которые могут работать в динамических средах. Это особенно важно для Kubernetes кластеров, где количество нод может изменяться в зависимости от нагрузки и требований.

Ключевые преимущества:
- **Динамическая адаптация** без переобучения
- **Сохранение знаний** при изменении размерности
- **Множественные стратегии** для разных сценариев
- **Производственная готовность** для реальных сред

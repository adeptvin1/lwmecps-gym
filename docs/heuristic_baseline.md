# Heuristic Baseline Algorithms

Простая эвристика (baseline) для сравнения с RL-алгоритмами в задаче размещения подов в MEC-инфраструктуре.

## Доступные эвристики

### 1. Uniform (равномерное распределение)
Распределяет реплики равномерно по всем развертываниям.

**Параметры:**
- `heuristic_type`: `"uniform"`
- `max_replicas`: максимальное количество реплик

### 2. Static (статическая конфигурация)
Использует фиксированное количество реплик для всех развертываний.

**Параметры:**
- `heuristic_type`: `"static"`
- `static_replicas`: фиксированное количество реплик (по умолчанию: `max_replicas // 2`)

### 3. Greedy Latency (жадный по задержке)
Размещает реплики на узлах с минимальной задержкой.

**Параметры:**
- `heuristic_type`: `"greedy_latency"`
- `static_replicas`: количество реплик для размещения

### 4. Greedy Load (жадный по загрузке)
Размещает реплики на узлах с минимальной загрузкой ресурсов.

**Параметры:**
- `heuristic_type`: `"greedy_load"`
- `static_replicas`: количество реплик для размещения

## Использование через API

### Пример: Uniform heuristic

```bash
curl -X POST "http://localhost:8010/api/v1/training/tasks" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "Uniform Baseline",
       "description": "Uniform distribution heuristic baseline",
       "model_type": "heuristic",
       "parameters": {
         "heuristic_type": "uniform",
         "static_replicas": 5
       },
       "total_episodes": 100
     }'
```

### Пример: Static heuristic

```bash
curl -X POST "http://localhost:8010/api/v1/training/tasks" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "Static Baseline",
       "description": "Static replica count heuristic baseline",
       "model_type": "heuristic",
       "parameters": {
         "heuristic_type": "static",
         "static_replicas": 3
       },
       "total_episodes": 100
     }'
```

### Пример: Greedy Latency heuristic

```bash
curl -X POST "http://localhost:8010/api/v1/training/tasks" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "Greedy Latency Baseline",
       "description": "Greedy latency-based placement heuristic",
       "model_type": "heuristic",
       "parameters": {
         "heuristic_type": "greedy_latency",
         "static_replicas": 4
       },
       "total_episodes": 100
     }'
```

### Пример: Greedy Load heuristic

```bash
curl -X POST "http://localhost:8010/api/v1/training/tasks" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "Greedy Load Baseline",
       "description": "Greedy load-based placement heuristic",
       "model_type": "heuristic",
       "parameters": {
         "heuristic_type": "greedy_load",
         "static_replicas": 4
       },
       "total_episodes": 100
     }'
```

## Использование в коде

```python
from lwmecps_gym.ml.models.heuristic_baseline import HeuristicBaseline
import gymnasium as gym

# Создать окружение
env = gym.make("lwmecps-v3", ...)

# Создать эвристику
heuristic = HeuristicBaseline(
    heuristic_type="uniform",
    num_deployments=4,
    max_replicas=10,
    static_replicas=5
)

# Запустить оценку
results = heuristic.train(env, total_episodes=100)
```

## Метрики

Эвристика возвращает те же метрики, что и RL-алгоритмы:
- `episode_rewards`: награды за каждый эпизод
- `episode_lengths`: длина каждого эпизода
- `episode_latencies`: средняя задержка за эпизод
- `mean_rewards`: скользящее среднее наград
- `mean_lengths`: скользящее среднее длин эпизодов

## Примечания

- Эвристики не требуют обучения, поэтому метод `train()` просто выполняет оценку
- Результаты сохраняются в том же формате, что и для RL-алгоритмов, для удобства сравнения
- Эвристики можно использовать как baseline для количественной оценки улучшений RL-методов


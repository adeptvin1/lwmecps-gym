# Experiments Configuration

Эта папка содержит структурированные конфигурации для запуска экспериментов с различными алгоритмами обучения с подкреплением и эвристическими baseline-методами.

## Структура папок

```
experiments/
├── training/              # Конфигурации для обучения
│   ├── ppo/               # PPO (3 запуска)
│   ├── sac/               # SAC (3 запуска)
│   ├── td3/               # TD3 (3 запуска)
│   └── baseline/          # Baseline эвристики
│       ├── static/        # Static (3 запуска)
│       └── greedy_latency/ # Greedy latency (3 запуска)
└── reconciliation/        # Шаблоны для reconciliation
    ├── profile_a/         # Профиль A (стабильность)
    └── profile_b/         # Профиль B (обобщение)
```

## Параметры экспериментов

### Обучение

**Реальное время:** 24 часа на запуск (фиксировано)

**Количество шагов (зависит от скорости алгоритма):**
- **SAC:** 1600 шагов за 24 часа (320 эпизодов)
- **TD3/PPO:** 960 шагов за 24 часа (192 эпизода)

**Профиль нагрузки:** Профиль A (полный суточный профиль)

**Количество запусков:** 3 независимых запуска

### Reconciliation

**Реальное время:** 24 часа на эксперимент (фиксировано)

**Количество шагов:**
- **SAC:** 1600 шагов
- **TD3/PPO:** 960 шагов

**Профили:**
- **Профиль A:** Проверка стабильности (тот же, что использовался для обучения)
- **Профиль B:** Проверка обобщения (альтернативный профиль)

## Использование

### Обучение

```bash
# Пример: запуск PPO run 1
curl -X POST "http://localhost:8010/api/v1/training/tasks" \
     -H "Content-Type: application/json" \
     -d @training/ppo/run1.json
```

### Reconciliation

```bash
# Создать reconciliation задачу для обученной модели
curl -X POST "http://localhost:8010/api/v1/training/tasks/{training_task_id}/reconcile" \
     -H "Content-Type: application/json" \
     -d '{
       "sample_size": 960,
       "group_id": "reconciliation-ppo-run1-profilea"
     }'

# Для SAC используйте sample_size: 1600
# Для TD3/PPO используйте sample_size: 960
```

## План выполнения

См. `EXPERIMENT_PLAN.md` для детального плана экспериментов.

**Общее время с 3 VM:** ~9 дней
- Обучение: ~3 дня
- Reconciliation: ~6 дней

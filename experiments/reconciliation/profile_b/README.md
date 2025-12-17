# Reconciliation на Профиле B (Обобщение)

Эта папка содержит шаблоны для reconciliation экспериментов на Профиле B (проверка обобщения).

## Использование

Reconciliation создается через API после завершения обучения:

```bash
# Создать reconciliation задачу
curl -X POST "http://localhost:8010/api/v1/training/tasks/{training_task_id}/reconcile" \
     -H "Content-Type: application/json" \
     -d '{
       "sample_size": 960,
       "group_id": "reconciliation-ppo-run1-profileb"
     }'

# Для SAC используйте sample_size: 1600
# Для TD3/PPO используйте sample_size: 960
```

## Параметры

- **Профиль:** Профиль B (альтернативный профиль для проверки обобщения)
- **Реальное время:** 24 часа
- **Количество шагов:**
  - SAC: 1600 шагов
  - TD3/PPO: 960 шагов

## Структура

Для каждой обученной модели создается отдельная reconciliation задача.


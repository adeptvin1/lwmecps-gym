# Обработка лог-файлов экспериментов

## Описание

Скрипт `parse_output_log.py` обрабатывает лог-файлы экспериментов и создает CSV файлы с разбивкой по количеству подов на нодах.

## Команда для обработки всех файлов

Выполните следующую команду из директории `source_code/lwmecps-gym`:

```bash
cd source_code/lwmecps-gym && find result_of_experiments -name "*.log" -type f -print0 | while IFS= read -r -d '' log_file; do
  csv_file="${log_file%.log}.csv"
  echo "Обработка: $log_file -> $csv_file"
  python3 parse_output_log.py "$log_file" "$csv_file"
  echo "---"
done
```

## Формат выходных CSV файлов

Каждый CSV файл содержит:
- **Relative Time (Process)** - относительное время процесса
- Колонки для каждой ноды с именем процесса: `Process Name - node_name`
  - Например: `TD3 Training - Run 1 (Profile A) 695c6acb8c72832fd1fbcc95 - minikube`
  - Например: `TD3 Training - Run 1 (Profile A) 695c6acb8c72832fd1fbcc95 - minikube-m02`
  - И так далее для всех 4 нод

## Структура папок

```
result_of_experiments/
├── Static/
│   └── Profile_A/
│       ├── Static Heuristic - Run 1 (Profile A).log
│       ├── Static Heuristic - Run 1 (Profile A).csv
│       └── ...
├── Greedy/
│   └── Profile_A/
│       └── ...
├── PPO/
│   └── Profile_A/
│       └── ...
├── SAC/
│   └── Profile_A/
│       └── ...
└── TD3/
    └── Profile_A/
        └── ...
```

## Примечания

- Скрипт автоматически определяет тип эксперимента из лога
- CSV файлы создаются рядом с исходными лог-файлами
- Разделитель в CSV - табуляция (как в CPU.txt)
- Скрипт отслеживает только deployment'ы эксперимента (`lwmecps-testapp-server-bs*`)


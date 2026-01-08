#!/usr/bin/env python3
"""
Парсер для output.log файла из wandb, создающий CSV файл с разбивкой по количеству подов на нодах
"""

import re
import csv
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import sys


def parse_log_file(log_path: str) -> Dict[str, List[Tuple[float, Dict]]]:
    """
    Парсит лог файл и извлекает информацию о количестве подов на нодах для каждой группы экспериментов.
    
    Args:
        log_path: Путь к файлу лога
        
    Returns:
        Словарь с данными: {group_id: [(relative_time, metrics_dict), ...]}
    """
    data = defaultdict(list)
    start_times = {}  # Время начала для каждой группы
    current_time = 0.0
    line_number = 0
    
    # Паттерны для поиска
    group_start_pattern = re.compile(r'Starting experiment group (\w+)')
    metrics_retrieved_pattern = re.compile(r'Retrieved metrics for group (\w+)')
    nodes_found_pattern = re.compile(r'Found (\d+) nodes')
    node_names_pattern = re.compile(r"Successfully collected state for nodes: \[(.*?)\]")
    replicas_pattern = re.compile(r'Updated deployment (lwmecps-testapp-server-\w+) to (\d+) replicas')
    setting_replicas_pattern = re.compile(r'Setting (?:initial )?replicas to (\d+) for deployment (lwmecps-testapp-server-\w+)')
    completed_pattern = re.compile(r'Experiment group COMPLETED|Terminating episode')
    
    # Для отслеживания текущей группы и состояния подов
    current_group_id = None
    current_nodes = {}  # {group_id: список нод}
    current_replicas = defaultdict(dict)  # {group_id: {deployment: replicas}}
    experiment_duration = 86400.0  # Длительность эксперимента в секундах (24 часа)
    group_end_times = {}  # {group_id: время окончания эксперимента}
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_number += 1
                line_stripped = line.strip()
                
                # Ищем начало группы экспериментов
                group_match = group_start_pattern.search(line_stripped)
                if group_match:
                    group_id = group_match.group(1)
                    if group_id not in start_times:
                        start_times[group_id] = current_time
                    current_group_id = group_id
                
                # Ищем список нод
                nodes_match = node_names_pattern.search(line_stripped)
                if nodes_match and current_group_id:
                    nodes_str = nodes_match.group(1)
                    nodes = [n.strip().strip("'\"") for n in nodes_str.split(',')]
                    current_nodes[current_group_id] = nodes
                
                # Ищем установку replicas (более раннее событие - "Setting replicas to X for deployment Y")
                setting_match = setting_replicas_pattern.search(line_stripped)
                if setting_match and current_group_id:
                    replicas = int(setting_match.group(1))
                    deployment = setting_match.group(2)
                    # Обновляем только если это deployment эксперимента
                    if 'lwmecps-testapp-server' in deployment:
                        current_replicas[current_group_id][deployment] = replicas
                
                # Ищем подтверждение обновления replicas (более позднее событие - "Updated deployment X to Y replicas")
                replicas_match = replicas_pattern.search(line_stripped)
                if replicas_match and current_group_id:
                    deployment = replicas_match.group(1)
                    replicas = int(replicas_match.group(2))
                    # Обновляем только если это deployment эксперимента
                    if 'lwmecps-testapp-server' in deployment:
                        current_replicas[current_group_id][deployment] = replicas
                
                # Ищем завершение эксперимента
                completed_match = completed_pattern.search(line_stripped)
                if completed_match and current_group_id:
                    # Эксперимент завершился в 86400 секунд от начала
                    if current_group_id in start_times:
                        group_end_times[current_group_id] = start_times[current_group_id] + experiment_duration
                
                # Ищем получение метрик - это момент, когда нужно записать состояние
                metrics_match = metrics_retrieved_pattern.search(line_stripped)
                if metrics_match:
                    group_id = metrics_match.group(1)
                    if group_id not in start_times:
                        start_times[group_id] = current_time
                    
                    # Вычисляем относительное время
                    # Если известна длительность эксперимента, нормализуем время
                    relative_time = current_time - start_times[group_id]
                    
                    # Если эксперимент завершился, используем нормализованное время до 86400
                    if group_id in group_end_times:
                        # Линейная интерполяция: текущее время относительно начала и конца
                        elapsed = current_time - start_times[group_id]
                        total_duration = group_end_times[group_id] - start_times[group_id]
                        if total_duration > 0:
                            # Нормализуем к 86400 секундам
                            relative_time = (elapsed / total_duration) * experiment_duration
                            relative_time = min(relative_time, experiment_duration)
                    
                    # Вычисляем количество подов на каждой ноде на основе replicas deployment'ов
                    pods_per_node = {}
                    nodes = current_nodes.get(group_id, [])
                    replicas_dict = current_replicas.get(group_id, {})
                    
                    # Суммируем общее количество подов из всех deployment'ов эксперимента
                    total_pods = sum(replicas_dict.values())
                    
                    if nodes and total_pods > 0:
                        # Равномерно распределяем поды по нодам
                        pods_per_node_count = total_pods // len(nodes) if len(nodes) > 0 else 0
                        remainder = total_pods % len(nodes)
                        
                        for i, node in enumerate(nodes):
                            # Распределяем остаток по первым нодам
                            pods_count = pods_per_node_count + (1 if i < remainder else 0)
                            pods_per_node[node] = pods_count
                    elif nodes:
                        # Если нет подов, устанавливаем 0 для всех нод
                        for node in nodes:
                            pods_per_node[node] = 0
                    
                    metrics_data = {
                        'pods_per_node': pods_per_node,
                        'total_pods': total_pods,
                        'nodes': nodes
                    }
                    
                    data[group_id].append((relative_time, metrics_data))
                    current_group_id = group_id
                
                # Увеличиваем текущее время
                # Используем приблизительное время: каждая строка ~0.01 секунды
                # Но учитываем "Waiting 10 seconds" как реальную задержку
                if 'Waiting' in line_stripped and 'seconds' in line_stripped:
                    wait_match = re.search(r'Waiting\s+(\d+)\s+seconds', line_stripped)
                    if wait_match:
                        wait_time = float(wait_match.group(1))
                        current_time += wait_time
                    else:
                        current_time += 0.01
                else:
                    current_time += 0.01  # Примерно 10мс на строку
                
    except Exception as e:
        print(f"Ошибка при чтении файла на строке {line_number}: {e}", file=sys.stderr)
        raise
    
    # Нормализуем время для каждой группы: эксперимент всегда длится 86400 секунд,
    # распределяем события равномерно по этому времени
    experiment_duration = 86400.0  # 24 часа в секундах
    normalized_data = {}
    
    for group_id, group_data in data.items():
        if not group_data:
            continue
        
        num_events = len(group_data)
        if num_events == 0:
            continue
        
        # Распределяем события равномерно от 0 до 86400 секунд
        normalized_group_data = []
        for i, (time_val, metrics) in enumerate(group_data):
            if num_events > 1:
                # Равномерное распределение: первое событие = 0, последнее = 86400
                normalized_time = (i / (num_events - 1)) * experiment_duration
            else:
                normalized_time = 0.0
            
            normalized_time = min(normalized_time, experiment_duration)
            normalized_group_data.append((normalized_time, metrics))
        
        normalized_data[group_id] = normalized_group_data
    
    return normalized_data


def extract_experiment_info(log_path: str) -> Dict[str, Dict]:
    """
    Извлекает информацию о типах экспериментов и их названиях из лога.
    
    Args:
        log_path: Путь к файлу лога
        
    Returns:
        Словарь с информацией об экспериментах: {group_id: {'type': ..., 'run': ..., 'profile': ...}}
    """
    experiments_info = {}
    
    # Паттерны для поиска информации об экспериментах
    heuristic_pattern = re.compile(r'(Static|Greedy Latency|Optimized Balance)\s+Heuristic')
    run_pattern = re.compile(r'Run\s+(\d+)')
    profile_pattern = re.compile(r'Profile\s+([A-Z])')
    
    # Также ищем упоминания типов агентов
    agent_pattern = re.compile(r'Creating\s+(\w+)\s+agent|Initializing\s+agent\s+of\s+type\s+(\w+)')
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            current_heuristic_type = None
            current_run = None
            current_profile = None
            
            for line in f:
                # Ищем упоминания типов эвристик
                heuristic_match = heuristic_pattern.search(line)
                if heuristic_match:
                    current_heuristic_type = heuristic_match.group(1)
                    run_match = run_pattern.search(line)
                    profile_match = profile_pattern.search(line)
                    if run_match:
                        current_run = int(run_match.group(1))
                    if profile_match:
                        current_profile = profile_match.group(1)
                
                # Ищем ID группы и связываем с текущей информацией
                group_id_match = re.search(r'(\w{24})', line)
                if group_id_match:
                    group_id = group_id_match.group(1)
                    if group_id not in experiments_info:
                        experiments_info[group_id] = {
                            'type': current_heuristic_type or 'Unknown',
                            'run': current_run,
                            'profile': current_profile or '?'
                        }
                
                # Ищем тип агента из строк типа "Creating HeuristicBaseline agent with type=static"
                agent_match = re.search(r'type=(\w+)', line)
                if agent_match and not current_heuristic_type:
                    agent_type = agent_match.group(1)
                    if agent_type == 'static':
                        current_heuristic_type = 'Static'
                    elif 'greedy' in agent_type.lower() or 'latency' in agent_type.lower():
                        current_heuristic_type = 'Greedy Latency'
                    elif 'balance' in agent_type.lower() or 'optimized' in agent_type.lower():
                        current_heuristic_type = 'Optimized Balance'
                        
    except Exception as e:
        print(f"Ошибка при извлечении информации об экспериментах: {e}", file=sys.stderr)
    
    return experiments_info


def create_csv_output(data: Dict[str, List[Tuple[float, Dict]]], 
                     experiments_info: Dict[str, Dict],
                     output_path: str):
    """
    Создает CSV файл с разбивкой по количеству подов на нодах
    
    Args:
        data: Данные из лога
        experiments_info: Информация об экспериментах
        output_path: Путь к выходному CSV файлу
    """
    # Собираем все уникальные группы экспериментов
    all_groups = sorted(data.keys())
    
    if not all_groups:
        print("Не найдено данных для экспорта", file=sys.stderr)
        return
    
    # Собираем все уникальные ноды из всех групп
    all_nodes = set()
    for group_data in data.values():
        for _, metrics in group_data:
            nodes = metrics.get('nodes', [])
            all_nodes.update(nodes)
    all_nodes = sorted(all_nodes)
    
    # Создаем заголовки: время и ноды с именем процесса
    headers = ['Relative Time (Process)']
    
    # Для каждой группы создаем колонки для каждой ноды с именем процесса
    for group_id in all_groups:
        exp_info = experiments_info.get(group_id, {})
        heuristic_type = exp_info.get('type', 'Unknown')
        run_num = exp_info.get('run', '?')
        profile = exp_info.get('profile', '?')
        process_name = f"{heuristic_type} Heuristic - Run {run_num} (Profile {profile}) {group_id}"
        
        # Добавляем колонки для каждой ноды с именем процесса
        for node in all_nodes:
            headers.append(f"{process_name} - {node}")
    
    # Собираем все временные метки и создаем строки данных
    all_times = set()
    for group_data in data.values():
        for time_val, _ in group_data:
            all_times.add(time_val)
    
    sorted_times = sorted(all_times)
    
    # Создаем словарь для быстрого доступа к данным по времени и группе
    time_data = defaultdict(dict)
    for group_id, group_data in data.items():
        for time_val, metrics in group_data:
            time_data[time_val][group_id] = metrics
    
    # Записываем CSV файл с табуляцией в качестве разделителя (как в CPU.txt)
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        
        # Записываем заголовки
        writer.writerow(headers)
        
        # Записываем данные
        for time_val in sorted_times:
            # Создаем одну строку для каждого времени со всеми группами
            row = [time_val]
            
            # Для каждой группы добавляем данные по всем нодам
            for group_id in all_groups:
                metrics = time_data[time_val].get(group_id, {})
                pods_per_node = metrics.get('pods_per_node', {})
                
                # Добавляем количество подов для каждой ноды этой группы
                for node in all_nodes:
                    pods_count = pods_per_node.get(node, 0)
                    row.append(pods_count)
            
            writer.writerow(row)
    
    print(f"CSV файл создан: {output_path}")
    print(f"Обработано групп экспериментов: {len(all_groups)}")
    print(f"Обработано нод: {len(all_nodes)}")
    print(f"Создано строк данных: {len(sorted_times)}")


def main():
    """Основная функция"""
    log_file = 'source_code/lwmecps-gym/output.log'
    output_file = 'output_metrics.csv'
    
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print(f"Парсинг файла: {log_file}")
    
    # Извлекаем информацию об экспериментах
    print("Извлечение информации об экспериментах...")
    experiments_info = extract_experiment_info(log_file)
    print(f"Найдено экспериментов: {len(experiments_info)}")
    
    # Парсим лог файл
    print("Парсинг лог файла...")
    data = parse_log_file(log_file)
    print(f"Найдено групп с данными: {len(data)}")
    
    # Создаем CSV файл
    print("Создание CSV файла...")
    create_csv_output(data, experiments_info, output_file)
    
    print("Готово!")


if __name__ == '__main__':
    main()


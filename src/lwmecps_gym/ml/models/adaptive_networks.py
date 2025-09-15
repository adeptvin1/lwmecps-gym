"""
Адаптивные нейронные сети для работы с изменяющейся размерностью.

Этот модуль содержит реализации нейронных сетей, которые могут
адаптироваться к изменению количества нод в кластере.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AdaptiveNetwork(ABC):
    """
    Абстрактный базовый класс для адаптивных нейронных сетей.
    
    Адаптивные сети могут изменять свою архитектуру в зависимости
    от количества нод в кластере, сохраняя при этом изученные знания.
    """
    
    def __init__(self, base_obs_dim: int, base_act_dim: int, 
                 hidden_size: int = 64, max_nodes: int = 20):
        """
        Инициализация адаптивной сети.
        
        Args:
            base_obs_dim: Базовая размерность наблюдений (не зависящая от нод)
            base_act_dim: Размерность действий
            hidden_size: Размер скрытых слоев
            max_nodes: Максимальное количество нод, которое может поддерживать сеть
        """
        self.base_obs_dim = base_obs_dim
        self.base_act_dim = base_act_dim
        self.hidden_size = hidden_size
        self.max_nodes = max_nodes
        self.current_nodes = 0
        self.architecture_history = []
        
        # Стратегии адаптации архитектуры
        self.scaling_strategies = {
            'zero_padding': self._zero_padding_strategy,
            'weight_interpolation': self._weight_interpolation_strategy,
            'knowledge_distillation': self._knowledge_distillation_strategy,
            'attention_based': self._attention_based_strategy
        }
    
    @abstractmethod
    def adapt_to_nodes(self, num_nodes: int, strategy: str = 'weight_interpolation') -> Dict[str, Any]:
        """
        Адаптирует сеть к новому количеству нод.
        
        Args:
            num_nodes: Новое количество нод
            strategy: Стратегия адаптации
            
        Returns:
            Словарь с информацией об адаптации
        """
        pass
    
    @abstractmethod
    def get_node_observation_dim(self) -> int:
        """Возвращает размерность наблюдения для одной ноды."""
        pass
    
    def _zero_padding_strategy(self, old_params: Dict[str, torch.Tensor], 
                              new_num_nodes: int) -> Dict[str, torch.Tensor]:
        """
        Стратегия нулевого заполнения.
        
        Простая стратегия: добавляет нулевые веса для новых нод.
        Подходит для небольших изменений размерности.
        """
        new_params = {}
        
        for name, param in old_params.items():
            if 'node_encoder' in name:
                # Для энкодеров нод добавляем нулевые веса
                old_size = param.shape[0]
                if new_num_nodes > old_size:
                    # Добавляем нулевые веса для новых нод
                    padding_size = new_num_nodes - old_size
                    if len(param.shape) == 2:
                        padding = torch.zeros(padding_size, param.shape[1], 
                                            device=param.device, dtype=param.dtype)
                        new_params[name] = torch.cat([param, padding], dim=0)
                    else:
                        new_params[name] = param
                else:
                    # Удаляем лишние ноды
                    new_params[name] = param[:new_num_nodes]
            else:
                new_params[name] = param
        
        return new_params
    
    def _weight_interpolation_strategy(self, old_params: Dict[str, torch.Tensor], 
                                     new_num_nodes: int) -> Dict[str, torch.Tensor]:
        """
        Стратегия интерполяции весов.
        
        Интерполирует веса между существующими нодами для создания
        весов для новых нод. Подходит для средних изменений размерности.
        """
        new_params = {}
        
        for name, param in old_params.items():
            if 'node_encoder' in name:
                old_size = param.shape[0]
                if new_num_nodes > old_size:
                    # Создаем интерполированные веса
                    if len(param.shape) == 2:
                        # Линейная интерполация весов
                        interpolated_weights = self._interpolate_weights_2d(
                            param, old_size, new_num_nodes
                        )
                        new_params[name] = interpolated_weights
                    else:
                        new_params[name] = param
                else:
                    new_params[name] = param[:new_num_nodes]
            else:
                new_params[name] = param
        
        return new_params
    
    def _interpolate_weights_2d(self, weights: torch.Tensor, old_size: int, 
                               new_size: int) -> torch.Tensor:
        """Интерполирует 2D веса для новой размерности."""
        if old_size == 0:
            return torch.zeros(new_size, weights.shape[1], 
                             device=weights.device, dtype=weights.dtype)
        
        # Создаем индексы для интерполяции
        old_indices = torch.linspace(0, old_size - 1, old_size, device=weights.device)
        new_indices = torch.linspace(0, old_size - 1, new_size, device=weights.device)
        
        # Интерполируем каждый столбец отдельно
        interpolated = torch.zeros(new_size, weights.shape[1], 
                                 device=weights.device, dtype=weights.dtype)
        
        for i in range(weights.shape[1]):
            interpolated[:, i] = torch.interp(new_indices, old_indices, weights[:, i])
        
        return interpolated
    
    def _knowledge_distillation_strategy(self, old_params: Dict[str, torch.Tensor], 
                                       new_num_nodes: int) -> Dict[str, torch.Tensor]:
        """
        Стратегия дистилляции знаний.
        
        Использует дистилляцию знаний для передачи информации
        от старой архитектуры к новой. Подходит для больших изменений размерности.
        """
        new_params = {}
        
        for name, param in old_params.items():
            if 'node_encoder' in name:
                old_size = param.shape[0]
                if new_num_nodes > old_size:
                    # Создаем новые веса с помощью дистилляции
                    if len(param.shape) == 2:
                        distilled_weights = self._distill_weights_2d(
                            param, old_size, new_num_nodes
                        )
                        new_params[name] = distilled_weights
                    else:
                        new_params[name] = param
                else:
                    new_params[name] = param[:new_num_nodes]
            else:
                new_params[name] = param
        
        return new_params
    
    def _distill_weights_2d(self, weights: torch.Tensor, old_size: int, 
                           new_size: int) -> torch.Tensor:
        """Дистиллирует 2D веса для новой размерности."""
        if old_size == 0:
            return torch.zeros(new_size, weights.shape[1], 
                             device=weights.device, dtype=weights.dtype)
        
        # Создаем новые веса, инициализированные из старых
        new_weights = torch.zeros(new_size, weights.shape[1], 
                                device=weights.device, dtype=weights.dtype)
        
        # Копируем существующие веса
        copy_size = min(old_size, new_size)
        new_weights[:copy_size] = weights[:copy_size]
        
        # Для дополнительных нод используем усредненные веса
        if new_size > old_size:
            avg_weights = weights.mean(dim=0, keepdim=True)
            for i in range(old_size, new_size):
                # Добавляем небольшой шум для разнообразия
                noise = torch.randn_like(avg_weights) * 0.1
                new_weights[i] = avg_weights.squeeze(0) + noise.squeeze(0)
        
        return new_weights
    
    def _attention_based_strategy(self, old_params: Dict[str, torch.Tensor], 
                                 new_num_nodes: int) -> Dict[str, torch.Tensor]:
        """
        Стратегия на основе внимания.
        
        Использует механизм внимания для адаптации к новой размерности.
        Подходит для сложных изменений размерности.
        """
        new_params = {}
        
        for name, param in old_params.items():
            if 'node_encoder' in name:
                old_size = param.shape[0]
                if new_num_nodes > old_size:
                    # Создаем веса на основе внимания
                    if len(param.shape) == 2:
                        attention_weights = self._create_attention_weights_2d(
                            param, old_size, new_num_nodes
                        )
                        new_params[name] = attention_weights
                    else:
                        new_params[name] = param
                else:
                    new_params[name] = param[:new_num_nodes]
            else:
                new_params[name] = param
        
        return new_params
    
    def _create_attention_weights_2d(self, weights: torch.Tensor, old_size: int, 
                                    new_size: int) -> torch.Tensor:
        """Создает веса на основе механизма внимания."""
        if old_size == 0:
            return torch.zeros(new_size, weights.shape[1], 
                             device=weights.device, dtype=weights.dtype)
        
        # Создаем матрицу внимания
        attention_matrix = torch.softmax(
            torch.randn(new_size, old_size, device=weights.device), dim=1
        )
        
        # Применяем внимание к существующим весам
        new_weights = torch.mm(attention_matrix, weights)
        
        return new_weights
    
    def log_architecture_change(self, old_nodes: int, new_nodes: int, 
                              strategy: str, adaptation_info: Dict[str, Any]):
        """Логирует изменение архитектуры."""
        change_record = {
            'timestamp': time.time(),
            'old_nodes': old_nodes,
            'new_nodes': new_nodes,
            'strategy': strategy,
            'adaptation_info': adaptation_info
        }
        self.architecture_history.append(change_record)
        logger.info(f"Architecture changed from {old_nodes} to {new_nodes} nodes using {strategy}")


class AdaptiveActorCritic(AdaptiveNetwork):
    """
    Адаптивная Actor-Critic сеть для PPO.
    
    Может изменять количество нод, сохраняя изученные паттерны.
    """
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 64, 
                 max_nodes: int = 20, max_replicas: int = 10):
        super().__init__(obs_dim, act_dim, hidden_size, max_nodes)
        self.max_replicas = max_replicas
        
        # Базовые слои (не зависят от количества нод)
        self.base_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Адаптивные энкодеры для нод
        self.node_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.get_node_observation_dim(), hidden_size),
                nn.ReLU()
            ) for _ in range(max_nodes)
        ])
        
        # Механизм внимания для объединения нод
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        
        # Выходные слои
        self.actor_head = nn.Linear(hidden_size, act_dim * (max_replicas + 1))
        self.critic_head = nn.Linear(hidden_size, 1)
        
        # Инициализация
        self.current_nodes = 0
    
    def get_node_observation_dim(self) -> int:
        """Возвращает размерность наблюдения для одной ноды."""
        return 7  # CPU, RAM, TX, RX, Read, Write, Latency
    
    def adapt_to_nodes(self, num_nodes: int, strategy: str = 'weight_interpolation') -> Dict[str, Any]:
        """Адаптирует сеть к новому количеству нод."""
        if num_nodes > self.max_nodes:
            raise ValueError(f"Number of nodes {num_nodes} exceeds maximum {self.max_nodes}")
        
        old_nodes = self.current_nodes
        self.current_nodes = num_nodes
        
        # Получаем текущие параметры
        current_params = {name: param.clone() for name, param in self.named_parameters()}
        
        # Применяем стратегию адаптации
        if strategy in self.scaling_strategies:
            adapted_params = self.scaling_strategies[strategy](current_params, num_nodes)
            
            # Обновляем параметры сети
            with torch.no_grad():
                for name, param in self.named_parameters():
                    if name in adapted_params:
                        param.data = adapted_params[name].data
            
            adaptation_info = {
                'strategy': strategy,
                'old_nodes': old_nodes,
                'new_nodes': num_nodes,
                'success': True
            }
        else:
            raise ValueError(f"Unknown scaling strategy: {strategy}")
        
        # Логируем изменение
        self.log_architecture_change(old_nodes, num_nodes, strategy, adaptation_info)
        
        return adaptation_info
    
    def forward(self, x):
        """Прямой проход через адаптивную сеть."""
        batch_size = x.shape[0]
        
        # Разделяем наблюдения по нодам
        node_observations = self._split_observations(x)
        
        # Кодируем каждую ноду
        node_encodings = []
        for i in range(self.current_nodes):
            if i < len(node_observations):
                encoded = self.node_encoders[i](node_observations[i])
                node_encodings.append(encoded)
        
        if not node_encodings:
            # Если нет нод, используем нулевые кодировки
            node_encodings = [torch.zeros(batch_size, self.hidden_size, device=x.device)]
        
        # Объединяем кодировки нод
        node_tensor = torch.stack(node_encodings, dim=1)  # (batch_size, num_nodes, hidden_size)
        
        # Применяем механизм внимания
        attended, _ = self.attention(node_tensor, node_tensor, node_tensor)
        
        # Глобальное усреднение
        global_features = attended.mean(dim=1)  # (batch_size, hidden_size)
        
        # Применяем базовый энкодер
        features = self.base_encoder(global_features)
        
        # Выходные головы
        actor_output = self.actor_head(features)
        critic_output = self.critic_head(features)
        
        # Формируем выход для PPO
        actor_output = actor_output.view(batch_size, self.base_act_dim, self.max_replicas + 1)
        
        return actor_output, critic_output
    
    def _split_observations(self, x):
        """Разделяет наблюдения на части, соответствующие нодам."""
        # Предполагаем, что наблюдения структурированы как:
        # [global_features, node1_features, node2_features, ...]
        
        batch_size = x.shape[0]
        node_dim = self.get_node_observation_dim()
        
        # Вычисляем, сколько глобальных признаков
        global_dim = self.base_obs_dim - (self.current_nodes * node_dim)
        
        # Извлекаем глобальные признаки
        global_features = x[:, :global_dim] if global_dim > 0 else torch.zeros(batch_size, 0, device=x.device)
        
        # Извлекаем признаки нод
        node_observations = []
        start_idx = global_dim
        
        for i in range(self.current_nodes):
            end_idx = start_idx + node_dim
            if end_idx <= x.shape[1]:
                node_obs = x[:, start_idx:end_idx]
                node_observations.append(node_obs)
                start_idx = end_idx
            else:
                # Если не хватает данных, используем нули
                node_obs = torch.zeros(batch_size, node_dim, device=x.device)
                node_observations.append(node_obs)
        
        return node_observations


class AdaptiveDQN(AdaptiveNetwork):
    """
    Адаптивная DQN сеть.
    
    Может изменять количество нод, сохраняя изученные Q-значения.
    """
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 64, 
                 max_nodes: int = 20):
        super().__init__(obs_dim, act_dim, hidden_size, max_nodes)
        
        # Базовые слои
        self.base_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Адаптивные энкодеры для нод
        self.node_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.get_node_observation_dim(), hidden_size),
                nn.ReLU()
            ) for _ in range(max_nodes)
        ])
        
        # Выходной слой
        self.q_head = nn.Linear(hidden_size, act_dim)
        
        self.current_nodes = 0
    
    def get_node_observation_dim(self) -> int:
        """Возвращает размерность наблюдения для одной ноды."""
        return 7  # CPU, RAM, TX, RX, Read, Write, Latency
    
    def adapt_to_nodes(self, num_nodes: int, strategy: str = 'weight_interpolation') -> Dict[str, Any]:
        """Адаптирует сеть к новому количеству нод."""
        if num_nodes > self.max_nodes:
            raise ValueError(f"Number of nodes {num_nodes} exceeds maximum {self.max_nodes}")
        
        old_nodes = self.current_nodes
        self.current_nodes = num_nodes
        
        # Получаем текущие параметры
        current_params = {name: param.clone() for name, param in self.named_parameters()}
        
        # Применяем стратегию адаптации
        if strategy in self.scaling_strategies:
            adapted_params = self.scaling_strategies[strategy](current_params, num_nodes)
            
            # Обновляем параметры сети
            with torch.no_grad():
                for name, param in self.named_parameters():
                    if name in adapted_params:
                        param.data = adapted_params[name].data
            
            adaptation_info = {
                'strategy': strategy,
                'old_nodes': old_nodes,
                'new_nodes': num_nodes,
                'success': True
            }
        else:
            raise ValueError(f"Unknown scaling strategy: {strategy}")
        
        # Логируем изменение
        self.log_architecture_change(old_nodes, num_nodes, strategy, adaptation_info)
        
        return adaptation_info
    
    def forward(self, x):
        """Прямой проход через адаптивную DQN сеть."""
        batch_size = x.shape[0]
        
        # Разделяем наблюдения по нодам
        node_observations = self._split_observations(x)
        
        # Кодируем каждую ноду
        node_encodings = []
        for i in range(self.current_nodes):
            if i < len(node_observations):
                encoded = self.node_encoders[i](node_observations[i])
                node_encodings.append(encoded)
        
        if not node_encodings:
            # Если нет нод, используем нулевые кодировки
            node_encodings = [torch.zeros(batch_size, self.hidden_size, device=x.device)]
        
        # Объединяем кодировки нод
        node_tensor = torch.stack(node_encodings, dim=1)  # (batch_size, num_nodes, hidden_size)
        
        # Глобальное усреднение
        global_features = node_tensor.mean(dim=1)  # (batch_size, hidden_size)
        
        # Применяем базовый энкодер
        features = self.base_encoder(global_features)
        
        # Q-значения
        q_values = self.q_head(features)
        
        return q_values
    
    def _split_observations(self, x):
        """Разделяет наблюдения на части, соответствующие нодам."""
        # Аналогично AdaptiveActorCritic
        batch_size = x.shape[0]
        node_dim = self.get_node_observation_dim()
        
        global_dim = self.base_obs_dim - (self.current_nodes * node_dim)
        node_observations = []
        start_idx = global_dim
        
        for i in range(self.current_nodes):
            end_idx = start_idx + node_dim
            if end_idx <= x.shape[1]:
                node_obs = x[:, start_idx:end_idx]
                node_observations.append(node_obs)
                start_idx = end_idx
            else:
                node_obs = torch.zeros(batch_size, node_dim, device=x.device)
                node_observations.append(node_obs)
        
        return node_observations

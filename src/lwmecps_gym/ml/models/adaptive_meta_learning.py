"""
Адаптивные версии мета-алгоритмов с поддержкой изменения размерности сети.

Этот модуль расширяет существующие мета-алгоритмы функциональностью
адаптации к изменению количества нод в кластере.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import time

from .meta_learning import MAML, FOMAML, MetaLearningWrapper
from .adaptive_networks import AdaptiveActorCritic, AdaptiveDQN
from .ppo_learning import PPO
from .sac_learning import SAC
from .td3_learning import TD3
from .dq_learning import DQNAgent

logger = logging.getLogger(__name__)


class AdaptiveMetaLearningWrapper(MetaLearningWrapper):
    """
    Адаптивная версия MetaLearningWrapper с поддержкой изменения размерности.
    
    Расширяет базовый MetaLearningWrapper функциональностью адаптации
    к изменению количества нод в кластере.
    """
    
    def __init__(self, base_algorithm: Any, meta_method: str = "maml", **kwargs):
        super().__init__(base_algorithm, meta_method, **kwargs)
        
        # История изменений архитектуры
        self.architecture_history = []
        
        # Стратегии адаптации размерности
        self.node_scaling_strategies = {
            'zero_padding': self._zero_padding_strategy,
            'weight_interpolation': self._weight_interpolation_strategy,
            'knowledge_distillation': self._knowledge_distillation_strategy,
            'attention_based': self._attention_based_strategy
        }
    
    def adapt_to_new_node_count(self, new_num_nodes: int, strategy: str = 'weight_interpolation') -> Dict[str, Any]:
        """
        Адаптирует мета-обученную модель к новому количеству нод.
        
        Args:
            new_num_nodes: Новое количество нод
            strategy: Стратегия адаптации архитектуры
            
        Returns:
            Словарь с результатами адаптации
        """
        logger.info(f"Adapting meta-learning model to {new_num_nodes} nodes using {strategy}")
        
        # 1. Адаптируем базовую архитектуру
        base_adaptation = self._adapt_base_architecture(new_num_nodes, strategy)
        
        # 2. Адаптируем мета-параметры
        meta_adaptation = self._adapt_meta_parameters(new_num_nodes, strategy)
        
        # 3. Обновляем мета-обучатель
        self._update_meta_learner(new_num_nodes, strategy)
        
        # 4. Сохраняем историю изменений
        adaptation_record = {
            'timestamp': time.time(),
            'old_nodes': getattr(self.base_algorithm, 'current_nodes', 0),
            'new_nodes': new_num_nodes,
            'strategy': strategy,
            'base_adaptation': base_adaptation,
            'meta_adaptation': meta_adaptation
        }
        self.architecture_history.append(adaptation_record)
        
        logger.info(f"Successfully adapted to {new_num_nodes} nodes")
        
        return {
            'success': True,
            'new_num_nodes': new_num_nodes,
            'strategy': strategy,
            'base_adaptation': base_adaptation,
            'meta_adaptation': meta_adaptation,
            'adaptation_timestamp': time.time()
        }
    
    def _adapt_base_architecture(self, new_num_nodes: int, strategy: str) -> Dict[str, Any]:
        """Адаптирует базовую архитектуру алгоритма."""
        if hasattr(self.base_algorithm, 'adapt_to_nodes'):
            # Если базовый алгоритм поддерживает адаптацию
            return self.base_algorithm.adapt_to_nodes(new_num_nodes, strategy)
        else:
            # Создаем новую адаптивную архитектуру
            return self._create_adaptive_architecture(new_num_nodes, strategy)
    
    def _create_adaptive_architecture(self, new_num_nodes: int, strategy: str) -> Dict[str, Any]:
        """Создает новую адаптивную архитектуру."""
        # Это зависит от конкретного типа базового алгоритма
        # В реальной реализации здесь будет логика создания адаптивной архитектуры
        return {
            'strategy': strategy,
            'new_num_nodes': new_num_nodes,
            'success': True
        }
    
    def _adapt_meta_parameters(self, new_num_nodes: int, strategy: str) -> Dict[str, Any]:
        """Адаптирует мета-параметры к новой размерности."""
        if not hasattr(self.meta_learner, 'meta_params'):
            return {'success': False, 'reason': 'No meta parameters found'}
        
        # Получаем текущие мета-параметры
        current_meta_params = self.meta_learner.meta_params
        
        # Применяем стратегию адаптации
        if strategy in self.node_scaling_strategies:
            adapted_params = self.node_scaling_strategies[strategy](
                current_meta_params, new_num_nodes
            )
            
            # Обновляем мета-параметры
            self.meta_learner.meta_params = adapted_params
            
            return {
                'success': True,
                'strategy': strategy,
                'old_params_count': len(current_meta_params),
                'new_params_count': len(adapted_params)
            }
        else:
            return {
                'success': False,
                'reason': f'Unknown strategy: {strategy}'
            }
    
    def _update_meta_learner(self, new_num_nodes: int, strategy: str):
        """Обновляет мета-обучатель с новой размерностью."""
        # Обновляем параметры мета-обучателя
        if hasattr(self.meta_learner, 'current_nodes'):
            self.meta_learner.current_nodes = new_num_nodes
        
        # Обновляем базовый алгоритм в мета-обучателе
        if hasattr(self.meta_learner, 'base_algorithm'):
            self.meta_learner.base_algorithm = self.base_algorithm
    
    def _zero_padding_strategy(self, meta_params: List[torch.Tensor], new_num_nodes: int) -> List[torch.Tensor]:
        """Стратегия нулевого заполнения для мета-параметров."""
        adapted_params = []
        
        for param in meta_params:
            if len(param.shape) >= 2 and param.shape[0] > 0:
                # Если параметр может быть адаптирован
                old_size = param.shape[0]
                if new_num_nodes > old_size:
                    # Добавляем нулевые веса
                    padding_size = new_num_nodes - old_size
                    if len(param.shape) == 2:
                        padding = torch.zeros(padding_size, param.shape[1], 
                                            device=param.device, dtype=param.dtype)
                        adapted_param = torch.cat([param, padding], dim=0)
                    else:
                        adapted_param = param
                else:
                    # Удаляем лишние веса
                    adapted_param = param[:new_num_nodes]
                else:
                    adapted_param = param
            else:
                adapted_param = param
            
            adapted_params.append(adapted_param)
        
        return adapted_params
    
    def _weight_interpolation_strategy(self, meta_params: List[torch.Tensor], new_num_nodes: int) -> List[torch.Tensor]:
        """Стратегия интерполяции весов для мета-параметров."""
        adapted_params = []
        
        for param in meta_params:
            if len(param.shape) >= 2 and param.shape[0] > 0:
                old_size = param.shape[0]
                if new_num_nodes > old_size:
                    # Интерполируем веса
                    if len(param.shape) == 2:
                        adapted_param = self._interpolate_weights_2d(param, old_size, new_num_nodes)
                    else:
                        adapted_param = param
                else:
                    adapted_param = param[:new_num_nodes]
                else:
                    adapted_param = param
            else:
                adapted_param = param
            
            adapted_params.append(adapted_param)
        
        return adapted_params
    
    def _interpolate_weights_2d(self, weights: torch.Tensor, old_size: int, new_size: int) -> torch.Tensor:
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
    
    def _knowledge_distillation_strategy(self, meta_params: List[torch.Tensor], new_num_nodes: int) -> List[torch.Tensor]:
        """Стратегия дистилляции знаний для мета-параметров."""
        adapted_params = []
        
        for param in meta_params:
            if len(param.shape) >= 2 and param.shape[0] > 0:
                old_size = param.shape[0]
                if new_num_nodes > old_size:
                    # Дистиллируем веса
                    if len(param.shape) == 2:
                        adapted_param = self._distill_weights_2d(param, old_size, new_num_nodes)
                    else:
                        adapted_param = param
                else:
                    adapted_param = param[:new_num_nodes]
                else:
                    adapted_param = param
            else:
                adapted_param = param
            
            adapted_params.append(adapted_param)
        
        return adapted_params
    
    def _distill_weights_2d(self, weights: torch.Tensor, old_size: int, new_size: int) -> torch.Tensor:
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
    
    def _attention_based_strategy(self, meta_params: List[torch.Tensor], new_num_nodes: int) -> List[torch.Tensor]:
        """Стратегия на основе внимания для мета-параметров."""
        adapted_params = []
        
        for param in meta_params:
            if len(param.shape) >= 2 and param.shape[0] > 0:
                old_size = param.shape[0]
                if new_num_nodes > old_size:
                    # Создаем веса на основе внимания
                    if len(param.shape) == 2:
                        adapted_param = self._create_attention_weights_2d(param, old_size, new_num_nodes)
                    else:
                        adapted_param = param
                else:
                    adapted_param = param[:new_num_nodes]
                else:
                    adapted_param = param
            else:
                adapted_param = param
            
            adapted_params.append(adapted_param)
        
        return adapted_params
    
    def _create_attention_weights_2d(self, weights: torch.Tensor, old_size: int, new_size: int) -> torch.Tensor:
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
    
    def get_architecture_history(self) -> List[Dict[str, Any]]:
        """Возвращает историю изменений архитектуры."""
        return self.architecture_history.copy()
    
    def get_current_node_count(self) -> int:
        """Возвращает текущее количество нод."""
        if hasattr(self.base_algorithm, 'current_nodes'):
            return self.base_algorithm.current_nodes
        return 0


class AdaptiveMetaPPO(AdaptiveMetaLearningWrapper):
    """
    Адаптивная версия MetaPPO с поддержкой изменения размерности.
    """
    
    def __init__(self, obs_dim: int, act_dim: int, meta_method: str = "maml", **kwargs):
        # Создаем адаптивную базовую архитектуру
        adaptive_network = AdaptiveActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_size=kwargs.get('hidden_size', 64),
            max_nodes=kwargs.get('max_nodes', 20),
            max_replicas=kwargs.get('max_replicas', 10)
        )
        
        # Создаем PPO с адаптивной архитектурой
        base_ppo = PPO(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_size=kwargs.get('hidden_size', 64),
            lr=kwargs.get('lr', 3e-4),
            gamma=kwargs.get('gamma', 0.99),
            lam=kwargs.get('lam', 0.95),
            clip_eps=kwargs.get('clip_eps', 0.2),
            ent_coef=kwargs.get('ent_coef', 0.0),
            vf_coef=kwargs.get('vf_coef', 0.5),
            n_steps=kwargs.get('n_steps', 2048),
            batch_size=kwargs.get('batch_size', 64),
            n_epochs=kwargs.get('n_epochs', 10),
            device=kwargs.get('device', 'cpu'),
            deployments=kwargs.get('deployments', []),
            max_replicas=kwargs.get('max_replicas', 10)
        )
        
        # Заменяем сеть на адаптивную
        base_ppo.model = adaptive_network
        
        # Инициализируем адаптивную обертку
        super().__init__(
            base_algorithm=base_ppo,
            meta_method=meta_method,
            meta_lr=kwargs.get('meta_lr', 0.01),
            inner_lr=kwargs.get('inner_lr', 0.01),
            num_inner_steps=kwargs.get('num_inner_steps', 1),
            device=kwargs.get('device', 'cpu')
        )
        
        # Инициализируем адаптивную сеть
        self.base_algorithm.model.adapt_to_nodes(
            kwargs.get('initial_nodes', 3), 
            'zero_padding'
        )


class AdaptiveMetaSAC(AdaptiveMetaLearningWrapper):
    """
    Адаптивная версия MetaSAC с поддержкой изменения размерности.
    """
    
    def __init__(self, obs_dim: int, act_dim: int, meta_method: str = "maml", **kwargs):
        # Создаем SAC с адаптивной архитектурой
        base_sac = SAC(
            obs_dim=obs_dim,
            act_dim=act_dim,
            num_actions_per_dim=kwargs.get('num_actions_per_dim', 11),
            hidden_size=kwargs.get('hidden_size', 256),
            lr=kwargs.get('lr', 3e-4),
            gamma=kwargs.get('gamma', 0.99),
            tau=kwargs.get('tau', 0.005),
            alpha=kwargs.get('alpha', 0.2),
            auto_entropy=kwargs.get('auto_entropy', True),
            target_entropy=kwargs.get('target_entropy', -1.0),
            batch_size=kwargs.get('batch_size', 256),
            device=kwargs.get('device', 'cpu'),
            deployments=kwargs.get('deployments', []),
            max_replicas=kwargs.get('max_replicas', 10)
        )
        
        # Инициализируем адаптивную обертку
        super().__init__(
            base_algorithm=base_sac,
            meta_method=meta_method,
            meta_lr=kwargs.get('meta_lr', 0.01),
            inner_lr=kwargs.get('inner_lr', 0.01),
            num_inner_steps=kwargs.get('num_inner_steps', 1),
            device=kwargs.get('device', 'cpu')
        )


class AdaptiveMetaTD3(AdaptiveMetaLearningWrapper):
    """
    Адаптивная версия MetaTD3 с поддержкой изменения размерности.
    """
    
    def __init__(self, obs_dim: int, act_dim: int, meta_method: str = "maml", **kwargs):
        # Создаем TD3 с адаптивной архитектурой
        base_td3 = TD3(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_size=kwargs.get('hidden_size', 256),
            lr=kwargs.get('lr', 3e-4),
            gamma=kwargs.get('gamma', 0.99),
            tau=kwargs.get('tau', 0.005),
            policy_delay=kwargs.get('policy_delay', 2),
            noise_clip=kwargs.get('noise_clip', 0.5),
            noise=kwargs.get('noise', 0.2),
            batch_size=kwargs.get('batch_size', 256),
            device=kwargs.get('device', 'cpu'),
            deployments=kwargs.get('deployments', []),
            max_replicas=kwargs.get('max_replicas', 10)
        )
        
        # Инициализируем адаптивную обертку
        super().__init__(
            base_algorithm=base_td3,
            meta_method=meta_method,
            meta_lr=kwargs.get('meta_lr', 0.01),
            inner_lr=kwargs.get('inner_lr', 0.01),
            num_inner_steps=kwargs.get('num_inner_steps', 1),
            device=kwargs.get('device', 'cpu')
        )


class AdaptiveMetaDQN(AdaptiveMetaLearningWrapper):
    """
    Адаптивная версия MetaDQN с поддержкой изменения размерности.
    """
    
    def __init__(self, env, meta_method: str = "maml", **kwargs):
        # Создаем DQN с адаптивной архитектурой
        from .dq_learning import ReplayBuffer
        
        replay_buffer = ReplayBuffer(kwargs.get('memory_size', 10000))
        base_dqn = DQNAgent(
            env=env,
            replay_buffer=replay_buffer,
            learning_rate=kwargs.get('learning_rate', 0.001),
            discount_factor=kwargs.get('discount_factor', 0.99),
            epsilon=kwargs.get('epsilon', 0.1),
            memory_size=kwargs.get('memory_size', 10000),
            batch_size=kwargs.get('batch_size', 32)
        )
        
        # Заменяем сеть на адаптивную
        base_dqn.model = AdaptiveDQN(
            obs_dim=base_dqn.obs_dim,
            act_dim=base_dqn.act_dim,
            hidden_size=kwargs.get('hidden_size', 64),
            max_nodes=kwargs.get('max_nodes', 20)
        )
        
        # Инициализируем адаптивную обертку
        super().__init__(
            base_algorithm=base_dqn,
            meta_method=meta_method,
            meta_lr=kwargs.get('meta_lr', 0.01),
            inner_lr=kwargs.get('inner_lr', 0.01),
            num_inner_steps=kwargs.get('num_inner_steps', 1),
            device=kwargs.get('device', 'cpu')
        )
        
        # Инициализируем адаптивную сеть
        self.base_algorithm.model.adapt_to_nodes(
            kwargs.get('initial_nodes', 3), 
            'zero_padding'
        )

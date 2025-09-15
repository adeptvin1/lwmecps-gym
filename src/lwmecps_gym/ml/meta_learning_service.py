"""
Meta-learning service for managing meta-learning training tasks.

This service provides functionality for training meta-learning algorithms
(MAML and FOMAML) on multiple tasks and adapting to new tasks.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import gymnasium as gym
from datetime import datetime
import asyncio
import threading
import os

from ..core.database import Database
from ..core.models import TrainingTask, TrainingResult, TrainingState, ModelType
from ..core.wandb_config import init_wandb, log_metrics, finish_wandb, WandbConfig
from .models.meta_ppo import MetaPPO
from .models.meta_sac import MetaSAC
from .models.meta_td3 import MetaTD3
from .models.meta_dqn import MetaDQN

logger = logging.getLogger(__name__)


class MetaLearningService:
    """
    Service class for managing meta-learning training tasks.
    
    This service handles meta-learning training on multiple tasks,
    adaptation to new tasks, and integration with the existing
    training infrastructure.
    """
    
    def __init__(self, db: Database, wandb_config: WandbConfig):
        """
        Initialize the meta-learning service.
        
        Args:
            db: Database instance for storing training tasks and results
            wandb_config: Configuration for Weights & Biases integration
        """
        self.db = db
        self.wandb_config = wandb_config
        self.active_tasks: Dict[str, bool] = {}  # Track running tasks
        
        # Meta-learning specific configurations
        self.supported_algorithms = {
            ModelType.META_PPO: MetaPPO,
            ModelType.META_SAC: MetaSAC,
            ModelType.META_TD3: MetaTD3,
            ModelType.META_DQN: MetaDQN
        }
    
    async def create_meta_training_task(self, task_data: Dict[str, Any]) -> TrainingTask:
        """
        Create a new meta-learning training task.
        
        Args:
            task_data: Dictionary containing task parameters and configuration
            
        Returns:
            Created TrainingTask instance
        """
        # Validate meta-learning specific parameters
        if "meta_method" not in task_data:
            task_data["meta_method"] = "maml"  # Default to MAML
        
        if task_data["meta_method"] not in ["maml", "fomaml"]:
            raise ValueError(f"Unsupported meta-learning method: {task_data['meta_method']}")
        
        if "tasks" not in task_data:
            raise ValueError("Meta-learning requires 'tasks' parameter with list of task environments")
        
        # Set default meta-learning parameters
        meta_params = task_data.get("meta_parameters", {})
        task_data["meta_parameters"] = {
            "meta_lr": meta_params.get("meta_lr", 0.01),
            "inner_lr": meta_params.get("inner_lr", 0.01),
            "num_inner_steps": meta_params.get("num_inner_steps", 1),
            "num_meta_epochs": meta_params.get("num_meta_epochs", 100),
            **meta_params
        }
        
        # Create training task
        task = TrainingTask(
            name=task_data.get("name", f"Meta-{task_data['meta_method'].upper()} Training"),
            description=task_data.get("description", f"Meta-learning training using {task_data['meta_method'].upper()}"),
            model_type=task_data["model_type"],
            parameters=task_data.get("parameters", {}),
            env_config=task_data.get("env_config", {}),
            model_params=task_data.get("model_params", {}),
            state=task_data.get("state", TrainingState.PENDING),
            total_episodes=task_data.get("total_episodes", 100),
            group_id=task_data.get("group_id", f"meta-{task_data['meta_method']}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            namespace=task_data.get("namespace", "lwmecps-testapp"),
            max_pods=task_data.get("max_pods", 50),
            base_url=task_data.get("base_url", "http://34.51.217.76:8001"),
            stabilization_time=task_data.get("stabilization_time", 10)
        )
        
        return await self.db.create_training_task(task)
    
    async def start_meta_training(self, task_id: str) -> Optional[TrainingTask]:
        """
        Start a meta-learning training task.
        
        Args:
            task_id: Unique identifier of the training task
            
        Returns:
            Updated TrainingTask instance if successful, None otherwise
        """
        task = await self.db.get_training_task(task_id)
        if not task:
            return None

        if task.model_type not in self.supported_algorithms:
            raise ValueError(f"Unsupported model type for meta-learning: {task.model_type}")

        task.state = TrainingState.RUNNING
        await self.db.update_training_task(task_id, task.model_dump())

        # Start training in a separate thread
        loop = asyncio.get_running_loop()
        thread = threading.Thread(target=self._run_meta_training, args=(task_id, task, self, loop))
        thread.start()

        return task
    
    def _run_meta_training(self, task_id: str, task: TrainingTask, service_instance, loop):
        """
        The actual meta-learning training process.
        
        Args:
            task_id: Unique identifier of the training task
            task: The TrainingTask instance
            service_instance: The instance of MetaLearningService
            loop: The asyncio event loop from the main thread
        """
        db_thread = None
        try:
            # Create a new DB connection for this thread
            db_thread = Database()

            # Initialize wandb
            init_wandb(self.wandb_config, task.name)
            task.wandb_run_id = wandb.run.id
            
            future = asyncio.run_coroutine_threadsafe(
                db_thread.update_training_task(task_id, task.model_dump()), 
                loop
            )
            future.result()

            logger.info(f"Starting meta-learning training for task {task_id}")
            
            # Get task parameters
            meta_method = task.parameters.get("meta_method", "maml")
            tasks_data = task.parameters.get("tasks", [])
            meta_params = task.parameters.get("meta_parameters", {})
            
            if not tasks_data:
                raise ValueError("No tasks provided for meta-learning")
            
            # Create environments for each task
            task_environments = []
            for i, task_config in enumerate(tasks_data):
                env = self._create_task_environment(task_config, task)
                task_environments.append({
                    'env': env,
                    'episodes': task_config.get('episodes', 50),
                    'task_id': f"task_{i}"
                })
            
            # Initialize meta-learning algorithm
            meta_algorithm = self._create_meta_algorithm(
                task.model_type, 
                task_environments[0]['env'], 
                meta_method,
                task.parameters,
                meta_params
            )
            
            # Train meta-learning algorithm
            num_meta_epochs = meta_params.get("num_meta_epochs", 100)
            training_metrics = meta_algorithm.train_meta_on_tasks(task_environments, num_meta_epochs)
            
            # Save meta-learned model
            os.makedirs("./models", exist_ok=True)
            model_path = f"./models/meta_{task.model_type.value}_{task_id}.pth"
            meta_algorithm.save_model(model_path)
            
            # Save model to wandb
            if task.wandb_run_id:
                metadata = {
                    'model_type': task.model_type.value,
                    'meta_method': meta_method,
                    'num_tasks': len(task_environments),
                    'num_meta_epochs': num_meta_epochs,
                    'meta_parameters': meta_params
                }
                
                artifact = wandb.Artifact(
                    name=f'meta_{task.model_type.value}_model_{task_id}',
                    type='meta_model',
                    description=f'Meta-learned model using {meta_method.upper()}',
                    metadata=metadata
                )
                artifact.add_file(model_path)
                wandb.log_artifact(artifact)
            
            # Update task state
            task.state = TrainingState.COMPLETED
            task.model_path = model_path
            task.metrics = training_metrics
            
            future = asyncio.run_coroutine_threadsafe(
                db_thread.update_training_task(task_id, task.model_dump()),
                loop
            )
            future.result()
            
            logger.info(f"Meta-learning training completed for task {task_id}")
            
        except Exception as e:
            logger.error(f"Error in meta-learning training for task {task_id}: {str(e)}")
            task.state = TrainingState.FAILED
            task.error_message = str(e)
            if db_thread:
                future = asyncio.run_coroutine_threadsafe(
                    db_thread.update_training_task(task_id, task.model_dump()),
                    loop
                )
                future.result()
        finally:
            # Cleanup
            self.active_tasks[task_id] = False
            if db_thread:
                future = asyncio.run_coroutine_threadsafe(db_thread.close(), loop)
                future.result()
            finish_wandb()
    
    def _create_task_environment(self, task_config: Dict[str, Any], training_task: TrainingTask) -> gym.Env:
        """
        Create environment for a specific task.
        
        Args:
            task_config: Task configuration
            training_task: Training task instance
            
        Returns:
            Environment instance
        """
        # This is a simplified version - in practice, you'd want to create
        # different environments based on task_config
        from gymnasium.envs.registration import register
        from ..envs import LWMECPSEnv3
        
        # Register environment if not already registered
        if not gym.envs.registry.get("lwmecps-v3"):
            register(
                id="lwmecps-v3",
                entry_point="lwmecps_gym.envs:LWMECPSEnv3",
                max_episode_steps=5,
            )
        
        # Create environment with task-specific parameters
        env = gym.make(
            "lwmecps-v3",
            node_name=task_config.get("node_name", ["node1", "node2", "node3"]),
            max_hardware=task_config.get("max_hardware", {
                "cpu": 8, "ram": 16000, "tx_bandwidth": 1000, "rx_bandwidth": 1000,
                "read_disks_bandwidth": 500, "write_disks_bandwidth": 500, "avg_latency": 300
            }),
            pod_usage=task_config.get("pod_usage", {
                "cpu": 2, "ram": 2000, "tx_bandwidth": 20, "rx_bandwidth": 20,
                "read_disks_bandwidth": 100, "write_disks_bandwidth": 100
            }),
            node_info=task_config.get("node_info", {}),
            num_nodes=task_config.get("num_nodes", 3),
            namespace=training_task.namespace,
            deployments=task_config.get("deployments", [
                "lwmecps-testapp-server-bs1", "lwmecps-testapp-server-bs2",
                "lwmecps-testapp-server-bs3", "lwmecps-testapp-server-bs4"
            ]),
            max_pods=training_task.max_pods,
            group_id=training_task.group_id,
            env_config={
                "base_url": training_task.base_url,
                "stabilization_time": training_task.stabilization_time
            }
        )
        
        return env
    
    def _create_meta_algorithm(self, model_type: ModelType, env: gym.Env, meta_method: str, 
                             parameters: Dict[str, Any], meta_params: Dict[str, Any]) -> Any:
        """
        Create meta-learning algorithm instance.
        
        Args:
            model_type: Type of base algorithm
            env: Environment instance
            meta_method: Meta-learning method
            parameters: Algorithm parameters
            meta_params: Meta-learning parameters
            
        Returns:
            Meta-learning algorithm instance
        """
        # Calculate dimensions
        obs_dim = self._calculate_obs_dim(env)
        act_dim = env.action_space.shape[0]
        
        # Get base algorithm parameters
        base_params = {
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "meta_method": meta_method,
            "device": parameters.get("device", "cpu"),
            "deployments": parameters.get("deployments", [
                "lwmecps-testapp-server-bs1", "lwmecps-testapp-server-bs2",
                "lwmecps-testapp-server-bs3", "lwmecps-testapp-server-bs4"
            ]),
            "max_replicas": parameters.get("max_replicas", 10)
        }
        
        # Add algorithm-specific parameters
        if model_type == ModelType.META_PPO:
            base_params.update({
                "hidden_size": parameters.get("hidden_size", 64),
                "lr": parameters.get("learning_rate", 3e-4),
                "gamma": parameters.get("discount_factor", 0.99),
                "lam": parameters.get("lambda", 0.95),
                "clip_eps": parameters.get("clip_epsilon", 0.2),
                "ent_coef": parameters.get("entropy_coef", 0.0),
                "vf_coef": parameters.get("value_function_coef", 0.5),
                "n_steps": parameters.get("n_steps", 2048),
                "batch_size": parameters.get("batch_size", 64),
                "n_epochs": parameters.get("n_epochs", 10)
            })
        elif model_type == ModelType.META_SAC:
            base_params.update({
                "num_actions_per_dim": parameters.get("max_replicas", 10) + 1,
                "hidden_size": parameters.get("hidden_size", 256),
                "lr": parameters.get("learning_rate", 3e-4),
                "gamma": parameters.get("discount_factor", 0.99),
                "tau": parameters.get("tau", 0.005),
                "alpha": parameters.get("alpha", 0.2),
                "auto_entropy": parameters.get("auto_entropy", True),
                "target_entropy": parameters.get("target_entropy", -1.0),
                "batch_size": parameters.get("batch_size", 256)
            })
        elif model_type == ModelType.META_TD3:
            base_params.update({
                "hidden_size": parameters.get("hidden_size", 256),
                "lr": parameters.get("learning_rate", 3e-4),
                "gamma": parameters.get("discount_factor", 0.99),
                "tau": parameters.get("tau", 0.005),
                "policy_delay": parameters.get("policy_delay", 2),
                "noise_clip": parameters.get("noise_clip", 0.5),
                "noise": parameters.get("noise", 0.2),
                "batch_size": parameters.get("batch_size", 256)
            })
        elif model_type == ModelType.META_DQN:
            base_params.update({
                "env": env,
                "learning_rate": parameters.get("learning_rate", 0.001),
                "discount_factor": parameters.get("discount_factor", 0.99),
                "epsilon": parameters.get("epsilon", 0.1),
                "memory_size": parameters.get("memory_size", 10000),
                "batch_size": parameters.get("batch_size", 32)
            })
        
        # Add meta-learning parameters
        base_params.update({
            "meta_lr": meta_params.get("meta_lr", 0.01),
            "inner_lr": meta_params.get("inner_lr", 0.01),
            "num_inner_steps": meta_params.get("num_inner_steps", 1)
        })
        
        # Create algorithm instance
        algorithm_class = self.supported_algorithms[model_type]
        return algorithm_class(**base_params)
    
    def _calculate_obs_dim(self, env: gym.Env) -> int:
        """
        Calculate observation dimension for environment.
        
        Args:
            env: Environment instance
            
        Returns:
            Observation dimension
        """
        # This is a simplified calculation - in practice, you'd want to
        # properly calculate based on the environment's observation space
        if hasattr(env, 'node_name') and hasattr(env, 'deployments'):
            obs_dim = 0
            for node in env.node_name:
                obs_dim += 4  # CPU, RAM, TX, RX
                for deployment in env.deployments:
                    obs_dim += 5  # CPU_usage, RAM_usage, TX_usage, RX_usage, Replicas
                obs_dim += 1  # avg_latency
            return obs_dim
        else:
            return 100  # Default fallback
    
    async def adapt_to_new_task(self, task_id: str, new_task_config: Dict[str, Any]) -> Dict[str, float]:
        """
        Adapt a trained meta-learning model to a new task.
        
        Args:
            task_id: ID of the trained meta-learning task
            new_task_config: Configuration for the new task
            
        Returns:
            Dictionary of adaptation metrics
        """
        task = await self.db.get_training_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        if task.state != TrainingState.COMPLETED:
            raise ValueError(f"Task {task_id} is not completed")
        
        if task.model_type not in self.supported_algorithms:
            raise ValueError(f"Unsupported model type for adaptation: {task.model_type}")
        
        # Load meta-learned model
        if not task.model_path or not os.path.exists(task.model_path):
            raise ValueError(f"Model file not found: {task.model_path}")
        
        # Create new task environment
        new_env = self._create_task_environment(new_task_config, task)
        
        # Create meta-learning algorithm
        meta_algorithm = self._create_meta_algorithm(
            task.model_type,
            new_env,
            task.parameters.get("meta_method", "maml"),
            task.parameters,
            task.parameters.get("meta_parameters", {})
        )
        
        # Load meta-learned parameters
        meta_algorithm.load_model(task.model_path)
        
        # Adapt to new task
        num_adaptation_episodes = new_task_config.get("adaptation_episodes", 10)
        adaptation_metrics = meta_algorithm.adapt_to_new_task(new_env, num_adaptation_episodes)
        
        return adaptation_metrics
    
    async def adapt_to_new_node_count(self, task_id: str, node_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt a trained meta-learning model to a new number of nodes.
        
        Args:
            task_id: ID of the trained meta-learning task
            node_config: New node configuration
            
        Returns:
            Result of adaptation
        """
        task = await self.db.get_training_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        if task.state != TrainingState.COMPLETED:
            raise ValueError(f"Task {task_id} is not completed")
        
        # Load meta-learned model
        if not task.model_path or not os.path.exists(task.model_path):
            raise ValueError(f"Model file not found: {task.model_path}")
        
        # Create adaptive version of algorithm
        adaptive_algorithm = self._create_adaptive_algorithm(task, node_config)
        
        # Load meta-learned parameters
        adaptive_algorithm.load_model(task.model_path)
        
        # Adapt to new number of nodes
        new_num_nodes = node_config['new_num_nodes']
        strategy = node_config.get('strategy', 'weight_interpolation')
        
        adaptation_result = adaptive_algorithm.adapt_to_new_node_count(new_num_nodes, strategy)
        
        # Save adapted model
        adapted_model_path = f"./models/adapted_{task.model_type.value}_{task_id}_{new_num_nodes}nodes.pth"
        adaptive_algorithm.save_model(adapted_model_path)
        
        # Update task
        task.model_path = adapted_model_path
        task.parameters['adapted_nodes'] = new_num_nodes
        task.parameters['adaptation_strategy'] = strategy
        
        await self.db.update_training_task(task_id, task.model_dump())
        
        return {
            "adaptation_result": adaptation_result,
            "new_model_path": adapted_model_path,
            "new_num_nodes": new_num_nodes,
            "strategy": strategy
        }
    
    def _create_adaptive_algorithm(self, task: TrainingTask, node_config: Dict[str, Any]) -> Any:
        """Create adaptive version of algorithm."""
        # This would create the appropriate adaptive algorithm based on task type
        # For now, return a placeholder
        return None
    
    async def get_architecture_history(self, task_id: str) -> List[Dict[str, Any]]:
        """
        Get architecture change history for a meta-learning task.
        
        Args:
            task_id: ID of the meta-learning task
            
        Returns:
            Architecture change history
        """
        task = await self.db.get_training_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        # Return architecture history from task parameters
        return task.parameters.get('architecture_history', [])
    
    async def get_current_node_count(self, task_id: str) -> Dict[str, Any]:
        """
        Get current number of nodes for a meta-learning task.
        
        Args:
            task_id: ID of the meta-learning task
            
        Returns:
            Current node count information
        """
        task = await self.db.get_training_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        return {
            "current_nodes": task.parameters.get('adapted_nodes', 3),
            "max_nodes": task.parameters.get('max_nodes', 20),
            "adaptation_capable": task.model_type in self.supported_algorithms
        }
    
    async def get_meta_training_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current progress of a meta-learning training task.
        
        Args:
            task_id: Unique identifier of the training task
            
        Returns:
            Dictionary containing task progress information
        """
        task = await self.db.get_training_task(task_id)
        if not task:
            return None
        
        return {
            "state": task.state,
            "metrics": task.metrics,
            "wandb_run_id": task.wandb_run_id,
            "meta_method": task.parameters.get("meta_method", "maml"),
            "num_tasks": len(task.parameters.get("tasks", [])),
            "meta_parameters": task.parameters.get("meta_parameters", {}),
            "current_nodes": task.parameters.get('adapted_nodes', 3),
            "adaptation_capable": task.model_type in self.supported_algorithms
        }

from typing import Dict, Any, Optional
import wandb
from datetime import datetime
import logging
from ..core.database import Database
from ..core.models import TrainingTask, TrainingResult, ReconciliationResult, TrainingState, ModelType
from ..core.wandb_config import init_wandb, log_metrics, log_model, finish_wandb
from ..core.wandb_config import WandbConfig
import gymnasium as gym
from kubernetes import client, config
from kubernetes.client import CoreV1Api
from lwmecps_gym.ml.models.q_learn import QLearningAgent
from .models.dq_learning import DQNAgent
from .models.ppo_learning import PPO
from gymnasium.envs.registration import register
from lwmecps_gym.envs import LWMECPSEnv3
import numpy as np
from gymnasium import spaces
import torch
import re
import bitmath
from lwmecps_gym.envs.kubernetes_api import k8s
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Register the environment
register(
    id="lwmecps-v3",
    entry_point="lwmecps_gym.envs:LWMECPSEnv3",
)

class TrainingService:
    """
    Service class for managing ML model training tasks.
    Handles task creation, execution, monitoring, and results storage.
    Integrates with Weights & Biases for experiment tracking and MongoDB for data persistence.
    """
    
    def __init__(self, db: Database, wandb_config: WandbConfig):
        """
        Initialize the training service.
        
        Args:
            db: Database instance for storing training tasks and results
            wandb_config: Configuration for Weights & Biases integration
        """
        self.db = db
        self.wandb_config = wandb_config
        self.active_tasks: Dict[str, bool] = {}  # Track running tasks
        # Initialize Kubernetes client
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        self.k8s_client = CoreV1Api()
        self.minikube = k8s()
    
    async def create_training_task(self, task_data: Dict[str, Any]) -> TrainingTask:
        """
        Create a new training task in the database.
        
        Args:
            task_data: Dictionary containing task parameters and configuration
            
        Returns:
            Created TrainingTask instance
        """
        # Generate unique group_id for the experiment if not provided
        if "group_id" not in task_data:
            task_data["group_id"] = f"training-{uuid.uuid4()}"
            
        # Ensure group_id is stored in the task
        task = TrainingTask(
            name=task_data.get("name", "Training Task"),
            description=task_data.get("description", ""),
            model_type=task_data["model_type"],
            parameters=task_data.get("parameters", {}),
            env_config=task_data.get("env_config", {}),
            model_config=task_data.get("model_config", {}),
            state=task_data.get("state", TrainingState.PENDING),
            total_episodes=task_data.get("total_episodes", 100),
            group_id=task_data["group_id"]  # Ensure group_id is included
        )
        return await self.db.create_training_task(task)
    
    async def start_training(self, task_id: str) -> Optional[TrainingTask]:
        """
        Start a training task if it's in a valid state.
        
        Args:
            task_id: Unique identifier of the training task
            
        Returns:
            Updated TrainingTask instance if successful, None otherwise
        """
        logger.info(f"Attempting to start training task {task_id}")
        
        # Get the task from database
        task = await self.db.get_training_task(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return None
            
        logger.info(f"Found task {task_id} in state {task.state}")
        
        # Validate task state
        if task.state != TrainingState.PENDING:
            logger.error(f"Task {task_id} is in state {task.state}, cannot start")
            return None
        
        try:
            # Initialize Weights & Biases for experiment tracking
            init_wandb(self.wandb_config)
            logger.info(f"Initialized wandb run with ID {wandb.run.id}")
            
            # Update task state to RUNNING
            task.state = TrainingState.RUNNING
            task.wandb_run_id = wandb.run.id
            task = await self.db.update_training_task(task_id, task.model_dump())
            
            if not task:
                logger.error(f"Failed to update task {task_id} state")
                finish_wandb()
                return None
                
            logger.info(f"Successfully started task {task_id}")
            
            # Mark task as active and start training process
            self.active_tasks[task_id] = True
            await self._run_training(task_id, task)
            
            return task
            
        except Exception as e:
            logger.error(f"Error starting task {task_id}: {str(e)}")
            finish_wandb()
            return None

    async def _run_training(self, task_id: str, task: TrainingTask):
        """
        Run the training process for a specific task.
        
        Args:
            task_id: Unique identifier of the training task
            task: TrainingTask instance
        """
        try:
            # Get Kubernetes state
            state = self.minikube.k8s_state()
            node_name = list(state.keys())

            # Базовые параметры
            max_hardware = {
                "cpu": 8,
                "ram": 16000,
                "tx_bandwidth": 1000,
                "rx_bandwidth": 1000,
                "read_disks_bandwidth": 500,
                "write_disks_bandwidth": 500,
                "avg_latency": 300,
            }

            pod_usage = {
                "cpu": 2,
                "ram": 2000,
                "tx_bandwidth": 20,
                "rx_bandwidth": 20,
                "read_disks_bandwidth": 100,
                "write_disks_bandwidth": 100,
            }

            # Создаем информацию о узлах
            node_info = {}
            for node in node_name:
                node_info[node] = {
                    "cpu": int(state[node]["cpu"]),
                    "ram": round(bitmath.KiB(int(re.findall(r"\d+", state[node]["memory"])[0])).to_MB().value),
                    "tx_bandwidth": 100,
                    "rx_bandwidth": 100,
                    "read_disks_bandwidth": 300,
                    "write_disks_bandwidth": 300,
                    "avg_latency": 10 + (10 * list(node_name).index(node)),
                }

            # Create environment
            env = gym.make(
                "lwmecps-v3",
                node_name=node_name,
                max_hardware=max_hardware,
                pod_usage=pod_usage,
                node_info=node_info,
                num_nodes=len(node_name),
                namespace="default",
                deployment_name="mec-test-app",
                deployments=["mec-test-app"],
                max_pods=10000,
                group_id=task.group_id  # Используем group_id из задачи
            )

            # Initialize agent based on model type
            if task.model_type == ModelType.Q_LEARNING:
                agent = QLearningAgent(
                    env,
                    learning_rate=task.learning_rate,
                    discount_factor=task.discount_factor,
                    exploration_rate=task.exploration_rate,
                    exploration_decay=task.exploration_decay,
                    wandb_run_id=task.wandb_run_id
                )
            elif task.model_type == ModelType.DQN:
                agent = DQNAgent(
                    env,
                    learning_rate=task.learning_rate,
                    discount_factor=task.discount_factor,
                    exploration_rate=task.exploration_rate,
                    exploration_decay=task.exploration_decay,
                    wandb_run_id=task.wandb_run_id
                )
            elif task.model_type == ModelType.PPO:
                agent = PPO(
                    env,
                    learning_rate=task.learning_rate,
                    discount_factor=task.discount_factor,
                    exploration_rate=task.exploration_rate,
                    exploration_decay=task.exploration_decay,
                    wandb_run_id=task.wandb_run_id
                )
            else:
                raise ValueError(f"Unsupported model type: {task.model_type}")

            # Run training
            results = agent.train(task.episodes)

            # Save results
            for episode in range(task.episodes):
                metrics = {
                    "total_reward": results["episode_reward"][episode],
                    "steps": results["episode_steps"][episode],
                    "epsilon": results["episode_exploration"][episode],
                    "latency": results["episode_latency"][episode]
                }
                await self.save_training_result(task_id, episode, metrics)

            # Save model
            if task.model_type == ModelType.Q_LEARNING:
                agent.save_q_table(f"./models/q_table_{task_id}.pkl")
            else:
                agent.save_model(f"./models/model_{task_id}.pth")

            # Update task state
            task.state = TrainingState.COMPLETED
            await self.db.update_training_task(task_id, task.model_dump())

        except Exception as e:
            logger.error(f"Error in training process for task {task_id}: {str(e)}")
            task.state = TrainingState.FAILED
            await self.db.update_training_task(task_id, task.model_dump())
            finish_wandb()
        finally:
            self.active_tasks[task_id] = False
            env.close()

    async def pause_training(self, task_id: str) -> Optional[TrainingTask]:
        """
        Pause a running training task.
        
        Args:
            task_id: Unique identifier of the training task
            
        Returns:
            Updated TrainingTask instance if successful, None otherwise
        """
        task = await self.db.get_training_task(task_id)
        if not task or task.state != TrainingState.RUNNING:
            return None
        
        self.active_tasks[task_id] = False
        task.state = TrainingState.PAUSED
        return await self.db.update_training_task(task_id, task.model_dump())
    
    async def resume_training(self, task_id: str) -> Optional[TrainingTask]:
        """
        Resume a paused training task.
        
        Args:
            task_id: Unique identifier of the training task
            
        Returns:
            Updated TrainingTask instance if successful, None otherwise
        """
        task = await self.db.get_training_task(task_id)
        if not task or task.state != TrainingState.PAUSED:
            return None
        
        self.active_tasks[task_id] = True
        task.state = TrainingState.RUNNING
        return await self.db.update_training_task(task_id, task.model_dump())
    
    async def stop_training(self, task_id: str) -> Optional[TrainingTask]:
        """
        Stop a running or paused training task.
        
        Args:
            task_id: Unique identifier of the training task
            
        Returns:
            Updated TrainingTask instance if successful, None otherwise
        """
        task = await self.db.get_training_task(task_id)
        if not task or task.state not in [TrainingState.RUNNING, TrainingState.PAUSED]:
            return None
        
        self.active_tasks[task_id] = False
        task.state = TrainingState.FAILED
        finish_wandb()
        return await self.db.update_training_task(task_id, task.model_dump())
    
    async def save_training_result(self, task_id: str, episode: int, metrics: Dict[str, float]) -> TrainingResult:
        """
        Save training results for a specific episode.
        
        Args:
            task_id: Unique identifier of the training task
            episode: Current episode number
            metrics: Dictionary of training metrics
            
        Returns:
            Created TrainingResult instance
        """
        task = await self.db.get_training_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        # Convert metrics to lists if they're not already
        metrics_list = {}
        for key, value in metrics.items():
            if key not in task.metrics:
                task.metrics[key] = []
            task.metrics[key].append(value)
            metrics_list[key] = task.metrics[key]
        
        # Update task metrics
        task.metrics = metrics_list
        await self.db.update_training_task(task_id, task.model_dump())
        
        result = TrainingResult(
            task_id=task_id,
            episode=episode,
            metrics=metrics_list,
            wandb_run_id=task.wandb_run_id
        )
        
        # Log metrics to Weights & Biases
        log_metrics(metrics, step=episode)
        
        return await self.db.save_training_result(result)
    
    async def get_training_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current progress of a training task.
        
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
            "wandb_run_id": task.wandb_run_id
        } 
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
from lwmecps_gym.envs import LWMECPSEnv
import numpy as np
from gymnasium import spaces
import torch
import re
import bitmath

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Register the environment
register(
    id="lwmecps-v0",
    entry_point="lwmecps_gym.envs:LWMECPSEnv",
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
    
    async def create_training_task(self, task_data: Dict[str, Any]) -> TrainingTask:
        """
        Create a new training task in the database.
        
        Args:
            task_data: Dictionary containing task parameters and configuration
            
        Returns:
            Created TrainingTask instance
        """
        task = TrainingTask(**task_data)
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
    
    async def run_reconciliation(self, task_id: str, sample_size: int) -> ReconciliationResult:
        """
        Run model reconciliation on new data to validate model performance.
        
        Args:
            task_id: Unique identifier of the training task
            sample_size: Number of samples to use for reconciliation
            
        Returns:
            Created ReconciliationResult instance
        """
        task = await self.db.get_training_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        # Initialize Weights & Biases for reconciliation
        init_wandb(self.wandb_config)
        
        # Download model weights from Weights & Biases based on model type
        api = wandb.Api()
        run = api.run(f"{self.wandb_config.project_name}/{task.wandb_run_id}")
        
        if task.model_type == ModelType.Q_LEARNING:
            # For Q-learning, we need to download the Q-table artifact
            artifacts = run.logged_artifacts()
            q_table_artifact = next((art for art in artifacts if art.type == 'model'), None)
            if not q_table_artifact:
                raise ValueError("Q-table artifact not found in wandb run")
            model_weights = q_table_artifact.download()
        else:
            # For DQN and PPO models
            model_weights = run.file("model_weights.pth").download()
        
        # Run reconciliation logic (placeholder implementation)
        metrics = {
            "accuracy": 0.95,  # Example metrics
            "loss": 0.05
        }
        
        result = ReconciliationResult(
            task_id=task_id,
            model_type=task.model_type,
            wandb_run_id=wandb.run.id,
            metrics=metrics,
            sample_size=sample_size,
            model_weights_path=str(model_weights)
        )
        
        # Log reconciliation metrics
        log_metrics(metrics)
        finish_wandb()
        
        return await self.db.save_reconciliation_result(result)
    
    async def get_training_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current training progress and metrics.
        
        Args:
            task_id: Unique identifier of the training task
            
        Returns:
            Dictionary containing task info, latest metrics, and active status
        """
        task = await self.db.get_training_task(task_id)
        if not task:
            return None
        
        results = await self.db.get_training_results(task_id)
        
        return {
            "task": task,
            "latest_metrics": results[-1].metrics if results else None,
            "is_active": self.active_tasks.get(task_id, False)
        }
    
    async def _run_training(self, task_id: str, task: TrainingTask):
        """
        Запуск процесса обучения.
        
        Args:
            task_id (str): ID задачи обучения
            task (TrainingTask): Объект задачи обучения
        """
        try:
            # Get Kubernetes cluster state
            logger.info("Getting Kubernetes state...")
            nodes = self.k8s_client.list_node()
            node_names = [node.metadata.name for node in nodes.items]
            logger.info(f"Found nodes: {node_names}")
            
            # Define environment parameters
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
            
            # Get node info from Kubernetes state
            node_info = {}
            state = self.k8s_client.k8s_state()
            for node in node_names:
                if node in state:
                    node_info[node] = {
                        "cpu": int(state[node]["cpu"]),
                        "ram": round(
                            bitmath.KiB(int(re.findall(r"\d+", state[node]["memory"])[0]))
                            .to_MB()
                            .value
                        ),
                        "tx_bandwidth": 100,
                        "rx_bandwidth": 100,
                        "read_disks_bandwidth": 300,
                        "write_disks_bandwidth": 300,
                        "avg_latency": 10 + node_names.index(node) * 10,
                    }
            
            # Create Gym environment
            logger.info("Creating Gym environment...")
            env = gym.make(
                "lwmecps-v0",
                num_nodes=len(node_names),
                node_name=node_names,
                max_hardware=max_hardware,
                pod_usage=pod_usage,
                node_info=node_info,
                deployment_name=task.env_config.get("deployment_name", "mec-test-app"),
                namespace=task.env_config.get("namespace", "default"),
                deployments=task.env_config.get("deployments", ["mec-test-app"]),
                max_pods=task.env_config.get("max_pods", 10000),
            )
            logger.info("Environment created successfully")

            # Получение размерностей пространств
            obs_dim = 0
            if isinstance(env.observation_space, spaces.Box):
                obs_dim = env.observation_space.shape[0]
            elif isinstance(env.observation_space, spaces.Dict):
                # Для LWMECPSEnv: 7 метрик на узел + 1 метрика развертывания на узел
                num_nodes = task.env_config.get("num_nodes", 3)
                num_deployments = len(task.env_config.get("deployments", ["mec-test-app"]))
                obs_dim = num_nodes * (7 + num_deployments)  # 7 метрик оборудования + метрики развертываний
            else:
                raise ValueError(f"Unsupported observation space type: {type(env.observation_space)}")

            act_dim = env.action_space.n

            # Создание агента в зависимости от типа модели
            if task.model_type == ModelType.PPO:
                agent = PPO(
                    obs_dim=obs_dim,
                    act_dim=act_dim,
                    hidden_size=task.model_config.get("hidden_size", 64),
                    lr=task.model_config.get("lr", 3e-4),
                    gamma=task.model_config.get("gamma", 0.99),
                    lam=task.model_config.get("lam", 0.95),
                    clip_eps=task.model_config.get("clip_eps", 0.2),
                    ent_coef=task.model_config.get("ent_coef", 0.01),
                    vf_coef=task.model_config.get("vf_coef", 0.5),
                    n_steps=task.model_config.get("n_steps", 2048),
                    batch_size=task.model_config.get("batch_size", 64),
                    n_epochs=task.model_config.get("n_epochs", 10),
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    deployments=task.env_config.get("deployments", ["mec-test-app"])
                )
            elif task.model_type == ModelType.DQN:
                agent = DQNAgent(env)
            elif task.model_type == ModelType.Q_LEARNING:
                agent = QLearningAgent(env)
            else:
                raise ValueError(f"Unsupported model type: {task.model_type}")

            # Загрузка модели, если указан путь
            if task.model_path:
                if task.model_type == ModelType.PPO:
                    agent.load_model(task.model_path)
                elif task.model_type == ModelType.DQN:
                    agent.model.load_model(task.model_path)
                elif task.model_type == ModelType.Q_LEARNING:
                    agent.load_q_table(task.model_path)

            # Вычисление общего количества шагов
            total_timesteps = task.total_episodes * task.model_config.get("n_steps", 2048)

            # Запуск обучения
            results = agent.train(env, total_timesteps=total_timesteps)

            # Сохранение результатов
            task.state = TrainingState.COMPLETED
            task.metrics = results
            task.progress = 100.0
            task.current_episode = task.total_episodes
            await self.db.update_training_task(task_id, task.model_dump())

            # Сохранение модели
            if task.model_type == ModelType.PPO:
                agent.save_model(f"models/ppo_model_{task_id}.pth")
            elif task.model_type == ModelType.DQN:
                agent.model.save_model(f"models/dqn_model_{task_id}.pth")
            elif task.model_type == ModelType.Q_LEARNING:
                agent.save_q_table(f"models/q_table_{task_id}.pkl")

            # Завершение wandb
            finish_wandb()

        except Exception as e:
            logger.error(f"Error in training task {task_id}: {str(e)}")
            task.state = TrainingState.FAILED
            task.error_message = str(e)
            await self.db.update_training_task(task_id, task.model_dump())
            finish_wandb()
            raise 
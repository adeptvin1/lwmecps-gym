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
from .models.td3_learning import TD3
from .models.sac_learning import SAC
from gymnasium.envs.registration import register
from lwmecps_gym.envs import LWMECPSEnv3
import numpy as np
from gymnasium import spaces
import torch
import re
import bitmath
from lwmecps_gym.envs.kubernetes_api import k8s
import uuid
from bson.objectid import ObjectId

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
            # Convert task_id to ObjectId for MongoDB operations
            task_id_obj = ObjectId(task_id)
            logger.info(f"Starting training process for task {task_id}")
            
            # Get Kubernetes state
            logger.info("Fetching Kubernetes cluster state...")
            state = self.minikube.k8s_state()
            logger.info(f"Received state: {state}")
            
            if state is None:
                raise Exception("Failed to get Kubernetes cluster state. No valid nodes found.")
            
            if not isinstance(state, dict):
                raise Exception(f"Invalid state type: {type(state)}. Expected dict.")
                
            node_name = list(state.keys())
            logger.info(f"Found nodes: {node_name}")
            
            if not node_name:
                raise Exception("No nodes found in cluster state.")

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
                try:
                    logger.info(f"Processing node {node}")
                    if node not in state:
                        logger.warning(f"Node {node} not found in state. Skipping.")
                        continue
                        
                    node_state = state[node]
                    logger.info(f"Node state: {node_state}")
                    
                    if not isinstance(node_state, dict):
                        logger.warning(f"Invalid node state type for {node}: {type(node_state)}. Skipping.")
                        continue
                        
                    if not all(key in node_state for key in ['cpu', 'memory']):
                        logger.warning(f"Node {node} is missing required fields. Found keys: {list(node_state.keys())}")
                        continue

                    # Extract CPU value
                    cpu_value = node_state['cpu']
                    if isinstance(cpu_value, str):
                        cpu_value = int(cpu_value)
                    
                    # Extract memory value
                    memory_value = node_state['memory']
                    if isinstance(memory_value, str):
                        # Remove 'Ki' suffix and convert to integer
                        memory_value = memory_value.replace('Ki', '')
                        try:
                            memory_value = int(memory_value)
                        except ValueError:
                            logger.warning(f"Could not convert memory value {memory_value} to integer. Skipping node {node}.")
                            continue

                    node_info[node] = {
                        "cpu": cpu_value,
                        "ram": round(bitmath.KiB(memory_value).to_MB().value),
                        "tx_bandwidth": 100,
                        "rx_bandwidth": 100,
                        "read_disks_bandwidth": 300,
                        "write_disks_bandwidth": 300,
                        "avg_latency": 10 + (10 * list(node_name).index(node)),
                    }
                    logger.info(f"Created node info for {node}: {node_info[node]}")
                except Exception as e:
                    logger.error(f"Error processing node {node}: {str(e)}")
                    continue

            if not node_info:
                raise Exception("No valid nodes could be processed")

            logger.info(f"Final node_info: {node_info}")

            # Create environment
            logger.info("Creating environment")
            env = gym.make(
                "lwmecps-v3",
                node_name=list(node_info.keys()),
                max_hardware=max_hardware,
                pod_usage=pod_usage,
                node_info=node_info,
                num_nodes=len(node_info),
                namespace=task.namespace,
                deployments=[
                    "lwmecps-testapp-server-bs1",
                    "lwmecps-testapp-server-bs2",
                    "lwmecps-testapp-server-bs3",
                    "lwmecps-testapp-server-bs4"
                ],
                max_pods=task.max_pods,
                group_id=str(task.group_id),
                env_config={
                    "base_url": task.base_url,
                    "stabilization_time": task.stabilization_time
                }
            )
            logger.info("Environment created successfully")

            # Get observation and action dimensions
            try:
                # For LWMECPSEnv3, we need to calculate observation dimension manually
                # since it uses Dict space
                obs_dim = 0
                for node in node_info:
                    # Add dimensions for each node's metrics
                    obs_dim += 4  # CPU, RAM, TX, RX
                    # Add dimensions for each deployment's metrics
                    for deployment in [
                        "lwmecps-testapp-server-bs1",
                        "lwmecps-testapp-server-bs2",
                        "lwmecps-testapp-server-bs3",
                        "lwmecps-testapp-server-bs4"
                    ]:
                        obs_dim += 5  # CPU_usage, RAM_usage, TX_usage, RX_usage, Replicas
                    obs_dim += 1  # avg_latency
                
                # For Box action space, we need the shape
                act_dim = env.action_space.shape[0]  # Number of deployments
                logger.info(f"Observation dimension: {obs_dim}, Action dimension: {act_dim}")
            except Exception as e:
                logger.error(f"Failed to get environment dimensions: {str(e)}")
                raise

            # Initialize appropriate agent based on model type
            logger.info(f"Initializing agent of type {task.model_type}")
            if task.model_type == ModelType.Q_LEARNING:
                agent = QLearningAgent(
                    learning_rate=task.parameters.get("learning_rate", 0.1),
                    discount_factor=task.parameters.get("discount_factor", 0.95),
                    exploration_rate=task.parameters.get("exploration_rate", 1.0),
                    exploration_decay=task.parameters.get("exploration_decay", 0.995),
                    min_exploration_rate=task.parameters.get("min_exploration_rate", 0.01),
                    max_states=task.parameters.get("max_states", 10000)
                )
            elif task.model_type == ModelType.DQN:
                agent = DQNAgent(
                    env,
                    learning_rate=task.parameters.get("learning_rate", 0.001),
                    discount_factor=task.parameters.get("discount_factor", 0.99),
                    epsilon=task.parameters.get("epsilon", 0.1),
                    memory_size=task.parameters.get("memory_size", 10000),
                    batch_size=task.parameters.get("batch_size", 32)
                )
            elif task.model_type == ModelType.PPO:
                agent = PPO(
                    obs_dim=obs_dim,
                    act_dim=act_dim,
                    hidden_size=task.parameters.get("hidden_size", 256),  # Increased for complex state
                    lr=task.parameters.get("learning_rate", 3e-4),
                    gamma=task.parameters.get("discount_factor", 0.99),
                    lam=task.parameters.get("lambda", 0.95),
                    clip_eps=task.parameters.get("clip_epsilon", 0.2),
                    ent_coef=task.parameters.get("entropy_coef", 0.01),  # Increased for exploration
                    vf_coef=task.parameters.get("value_function_coef", 0.5),
                    n_steps=task.parameters.get("n_steps", 2048),
                    batch_size=task.parameters.get("batch_size", 64),
                    n_epochs=task.parameters.get("n_epochs", 10),
                    device=task.parameters.get("device", "cpu"),
                    deployments=[
                        "lwmecps-testapp-server-bs1",
                        "lwmecps-testapp-server-bs2",
                        "lwmecps-testapp-server-bs3",
                        "lwmecps-testapp-server-bs4"
                    ],
                    max_replicas=50  # Fixed value of 50
                )
            elif task.model_type == ModelType.TD3:
                agent = TD3(
                    obs_dim=obs_dim,
                    act_dim=act_dim,
                    hidden_size=task.parameters.get("hidden_size", 256),
                    lr=task.parameters.get("learning_rate", 3e-4),
                    gamma=task.parameters.get("discount_factor", 0.99),
                    tau=task.parameters.get("tau", 0.005),
                    policy_delay=task.parameters.get("policy_delay", 2),
                    noise_clip=task.parameters.get("noise_clip", 0.5),
                    noise=task.parameters.get("noise", 0.2),
                    batch_size=task.parameters.get("batch_size", 256),
                    device=task.parameters.get("device", "cpu"),
                    deployments=task.parameters.get("deployments", ["mec-test-app"])
                )
            elif task.model_type == ModelType.SAC:
                agent = SAC(
                    obs_dim=obs_dim,
                    act_dim=act_dim,
                    hidden_size=task.parameters.get("hidden_size", 256),
                    lr=task.parameters.get("learning_rate", 3e-4),
                    gamma=task.parameters.get("discount_factor", 0.99),
                    tau=task.parameters.get("tau", 0.005),
                    alpha=task.parameters.get("alpha", 0.2),
                    auto_entropy=task.parameters.get("auto_entropy", True),
                    target_entropy=task.parameters.get("target_entropy", -1.0),
                    batch_size=task.parameters.get("batch_size", 256),
                    device=task.parameters.get("device", "cpu"),
                    deployments=task.parameters.get("deployments", ["mec-test-app"])
                )
            else:
                raise ValueError(f"Unsupported model type: {task.model_type}")

            # Run training
            if task.model_type in [ModelType.PPO, ModelType.TD3, ModelType.SAC]:
                results = agent.train(env, total_timesteps=task.total_episodes * agent.n_steps)
            else:
                results = agent.train(env, task.total_episodes, wandb_run_id=task.wandb_run_id)

            # Save results
            if task.model_type in [ModelType.PPO, ModelType.TD3, ModelType.SAC]:
                for episode in range(len(results["episode_rewards"])):
                    metrics = {
                        "total_reward": results["episode_rewards"][episode],
                        "steps": results["episode_lengths"][episode],
                        "actor_loss": results["actor_losses"][episode],
                        "critic_loss": results["critic_losses"][episode],
                        "total_loss": results["total_losses"][episode],
                        "mean_reward": results["mean_rewards"][episode],
                        "mean_length": results["mean_lengths"][episode]
                    }
                    if task.model_type == ModelType.SAC:
                        metrics["alpha_loss"] = results["alpha_losses"][episode]
                    await self.save_training_result(task_id, episode, metrics)
            else:
                for episode in range(task.total_episodes):
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
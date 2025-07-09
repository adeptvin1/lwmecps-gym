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
import concurrent.futures
import asyncio
import threading
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_keys_to_str(data: Any) -> Any:
    if isinstance(data, dict):
        return {str(k): convert_keys_to_str(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_keys_to_str(i) for i in data]
    return data

# Register the environment
register(
    id="lwmecps-v3",
    entry_point="lwmecps_gym.envs:LWMECPSEnv3",
    max_episode_steps=5,
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
        # Sanitize dictionaries to ensure all keys are strings
        if "parameters" in task_data:
            task_data["parameters"] = convert_keys_to_str(task_data["parameters"])
        if "env_config" in task_data:
            task_data["env_config"] = convert_keys_to_str(task_data["env_config"])
        if "model_params" in task_data:
            task_data["model_params"] = convert_keys_to_str(task_data["model_params"])
            
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
            model_params=task_data.get("model_params", {}),
            state=task_data.get("state", TrainingState.PENDING),
            total_episodes=task_data.get("total_episodes", 100),
            group_id=task_data["group_id"],  # Ensure group_id is included
            namespace=task_data.get("namespace", "lwmecps-testapp"),
            max_pods=task_data.get("max_pods", 50),
            base_url=task_data.get("base_url", "http://34.51.217.76:8001"),
            stabilization_time=task_data.get("stabilization_time", 10)
        )
        return await self.db.create_training_task(task)
    
    async def start_training(self, task_id: str) -> Optional[TrainingTask]:
        """
        Start a new training task.
        
        Args:
            task_id: Unique identifier of the training task
            
        Returns:
            Updated TrainingTask instance if successful, None otherwise
        """
        task = await self.db.get_training_task(task_id)
        if not task:
            return None

        task.state = TrainingState.RUNNING
        await self.db.update_training_task(task_id, task.model_dump())

        # Start training in a separate thread, passing the running event loop
        loop = asyncio.get_running_loop()
        thread = threading.Thread(target=self._run_training, args=(task_id, task, self, loop))
        thread.start()

        return task

    def _run_training(self, task_id: str, task: TrainingTask, service_instance, loop):
        """
        The actual training process, designed to be run in a thread.
        
        Args:
            task_id: Unique identifier of the training task
            task: The TrainingTask instance
            service_instance: The instance of TrainingService to call back to
            loop: The asyncio event loop from the main thread
        """
        env = None
        db_thread = None
        try:
            # Create a new DB connection for this thread
            db_thread = Database()

            # Initialize wandb and update task by scheduling on the main loop
            init_wandb(self.wandb_config, task.name)
            task.wandb_run_id = wandb.run.id
            
            future = asyncio.run_coroutine_threadsafe(
                db_thread.update_training_task(task_id, task.model_dump()), 
                loop
            )
            future.result() # Wait for the update to complete

            # Update the service's main task object as well
            service_instance.task = task

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
                deployments=task.parameters.get("deployments", [
                    "lwmecps-testapp-server-bs1",
                    "lwmecps-testapp-server-bs2",
                    "lwmecps-testapp-server-bs3",
                    "lwmecps-testapp-server-bs4"
                ]),
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
                # Calculate max_replicas based on CPU capacity
                max_replicas = int(max_hardware["cpu"] / pod_usage["cpu"])
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
                    max_replicas=max_replicas  # Use calculated value
                )
            elif task.model_type == ModelType.TD3:
                max_replicas = int(max_hardware["cpu"] / pod_usage["cpu"])
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
                    deployments=[
                        "lwmecps-testapp-server-bs1",
                        "lwmecps-testapp-server-bs2",
                        "lwmecps-testapp-server-bs3",
                        "lwmecps-testapp-server-bs4"
                    ],
                    max_replicas=max_replicas
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
                    deployments=[
                        "lwmecps-testapp-server-bs1",
                        "lwmecps-testapp-server-bs2",
                        "lwmecps-testapp-server-bs3",
                        "lwmecps-testapp-server-bs4"
                    ],
                    max_replicas=task.parameters.get("max_replicas", 10)
                )
            else:
                raise ValueError(f"Unsupported model type: {task.model_type}")

            # Run training
            if task.model_type == ModelType.PPO:
                results = agent.train(
                    env, 
                    total_episodes=task.total_episodes, 
                    wandb_run_id=task.wandb_run_id,
                    training_service=service_instance,
                    task_id=task_id,
                    loop=loop,
                    db_connection=db_thread
                )
            elif task.model_type in [ModelType.TD3, ModelType.SAC]:
                results = agent.train(env, total_episodes=task.total_episodes, wandb_run_id=task.wandb_run_id)
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
                    future = asyncio.run_coroutine_threadsafe(
                        self.save_training_result(task_id, episode, metrics, db_connection=db_thread),
                        loop
                    )
                    future.result()
            else:
                for episode in range(task.total_episodes):
                    metrics = {
                        "total_reward": results["episode_reward"][episode],
                        "steps": results["episode_steps"][episode],
                        "epsilon": results["episode_exploration"][episode],
                        "latency": results["episode_latency"][episode]
                    }
                    future = asyncio.run_coroutine_threadsafe(
                        self.save_training_result(task_id, episode, metrics, db_connection=db_thread),
                        loop
                    )
                    future.result()

            # Save model
            if task.model_type == ModelType.Q_LEARNING:
                model_path = f"./models/model_{task.model_type.value}_{task_id}.pth"
                agent.save_q_table(model_path)
            else:
                model_path = f"./models/model_{task.model_type.value}_{task_id}.pth"
                logger.info(f"Attempting to save model to {model_path}")
                agent.save_model(model_path)
                
            # Save model to wandb if run_id is provided
            if task.wandb_run_id:
                # Fetch the latest task data to get updated metrics
                latest_task_future = asyncio.run_coroutine_threadsafe(
                    db_thread.get_training_task(task_id),
                    loop
                )
                latest_task = latest_task_future.result()

                logger.info(f"Saving model to wandb for task {task_id}")

                # Ensure all metadata keys are strings before logging to wandb
                metadata = {
                    'model_type': latest_task.model_type.value,
                    'total_episodes': latest_task.total_episodes,
                    'parameters': convert_keys_to_str(latest_task.parameters),
                    'env_config': convert_keys_to_str(latest_task.env_config),
                    'model_params': convert_keys_to_str(latest_task.model_params),
                    'training_metrics': convert_keys_to_str(latest_task.metrics)
                }

                artifact = wandb.Artifact(
                    name=f'{latest_task.model_type.value}_model_{task_id}',
                    type='model',
                    description=f'Model trained for {latest_task.total_episodes} episodes',
                    metadata=metadata
                )
                artifact.add_file(model_path)
                wandb.log_artifact(artifact)
                logger.info(f"Model successfully saved to wandb as artifact: {artifact.name}")

            # Update task state
            task.state = TrainingState.COMPLETED

            # Sanitize task dictionaries before final update
            task.parameters = convert_keys_to_str(task.parameters)
            task.env_config = convert_keys_to_str(task.env_config)
            task.model_params = convert_keys_to_str(task.model_params)
            task.metrics = convert_keys_to_str(task.metrics)

            future = asyncio.run_coroutine_threadsafe(
                db_thread.update_training_task(task_id, task.model_dump()),
                loop
            )
            future.result()
            
        except Exception as e:
            logger.error(f"Error in training process for task {task_id}: {str(e)}")
            task.state = TrainingState.FAILED
            if db_thread:
                future = asyncio.run_coroutine_threadsafe(
                    db_thread.update_training_task(task_id, task.model_dump()),
                    loop
                )
                future.result()
        finally:
            # Cleanup resources and finish wandb session
            self.active_tasks[task_id] = False
            if env:
                env.close()
            if db_thread:
                future = asyncio.run_coroutine_threadsafe(db_thread.close(), loop)
                future.result()
            # Always finish wandb session to prevent session leakage
            finish_wandb()

    async def update_training_progress(self, task_id: str, episode: int, progress: float, db_connection=None):
        """Update the progress of a training task."""
        db = db_connection or self.db
        task = await db.get_training_task(task_id)
        if task:
            task.current_episode = episode
            task.progress = progress
            await db.update_training_task(task_id, task.model_dump())

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
    
    async def save_training_result(self, task_id: str, episode: int, metrics: Dict[str, float], db_connection=None) -> TrainingResult:
        """
        Save training results for a specific episode.
        
        Args:
            task_id: Unique identifier of the training task
            episode: Current episode number
            metrics: Dictionary of training metrics
            db_connection: Optional database connection to use
            
        Returns:
            Created TrainingResult instance
        """
        db = db_connection or self.db
        task = await db.get_training_task(task_id)
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
        await db.update_training_task(task_id, task.model_dump())
        
        result = TrainingResult(
            task_id=task_id,
            episode=episode,
            metrics=metrics,
            wandb_run_id=task.wandb_run_id
        )
        
        # Log metrics to Weights & Biases
        log_metrics(metrics, step=episode)
        
        return await db.save_training_result(result)
    
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
    
    async def run_reconciliation(self, task_id: str, sample_size: int, group_id: Optional[str] = None) -> ReconciliationResult:
        """
        Run model reconciliation on new data.
        
        Args:
            task_id: Unique identifier of the training task
            sample_size: The number of steps to run the reconciliation for
            group_id: Optional group ID to use for reconciliation. If not provided, uses task's group_id
            
        Returns:
            ReconciliationResult instance
        """
        task = await self.db.get_training_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        if task.state != TrainingState.COMPLETED:
            raise ValueError(f"Task {task_id} is not completed")
            
        model_path = f"./models/model_{task.model_type.value}_{task_id}.pth"
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found at {model_path}")

        # Load model checkpoint to get correct dimensions
        import torch
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract dimensions from saved model
        saved_obs_dim = checkpoint.get('obs_dim', 100)  # fallback to 100
        saved_act_dim = checkpoint.get('act_dim', 4)    # fallback to 4
        saved_deployments = checkpoint.get('deployments', [
            "lwmecps-testapp-server-bs1",
            "lwmecps-testapp-server-bs2", 
            "lwmecps-testapp-server-bs3",
            "lwmecps-testapp-server-bs4"
        ])
        saved_max_replicas = checkpoint.get('max_replicas', 10)
        
        logger.info(f"Loaded model dimensions: obs_dim={saved_obs_dim}, act_dim={saved_act_dim}")
        
        # Use provided group_id or fallback to task's original group_id
        reconciliation_group_id = group_id if group_id is not None else task.group_id
        logger.info(f"Running reconciliation for task {task_id} with group_id: {reconciliation_group_id}")
        
        # Default parameters for reconciliation environment
        # The environment will get current k8s state automatically
        default_node_name = ["node1"]
        default_max_hardware = {
            "cpu": 8,
            "ram": 16000,
            "tx_bandwidth": 1000,
            "rx_bandwidth": 1000,
            "read_disks_bandwidth": 500,
            "write_disks_bandwidth": 500,
            "avg_latency": 300,
        }
        default_pod_usage = {
            "cpu": 2,
            "ram": 2000,
            "tx_bandwidth": 20,
            "rx_bandwidth": 20,
            "read_disks_bandwidth": 100,
            "write_disks_bandwidth": 100,
        }
        default_node_info = {
            "node1": {
                "cpu": 8,
                "memory": 16000,
                "tx_bandwidth": 1000,
                "rx_bandwidth": 1000,
                "read_disks_bandwidth": 500,
                "write_disks_bandwidth": 500,
                "avg_latency": 300,
            }
        }
        
        # Create environment for reconciliation
        env = gym.make(
            task.env_config.get("env_name", "lwmecps-v3"),
            node_name=default_node_name,
            max_hardware=default_max_hardware,
            pod_usage=default_pod_usage,
            node_info=default_node_info,
            num_nodes=len(default_node_name),
            namespace=task.namespace,
            deployments=saved_deployments,  # Use deployments from saved model
            max_pods=task.max_pods,
            group_id=reconciliation_group_id,
            base_url=task.base_url,
            stabilization_time=task.stabilization_time
        )

        # Use dimensions from saved model instead of calculating from environment
        obs_dim = saved_obs_dim
        act_dim = saved_act_dim

        # Create agent with correct parameters
        agent = None
        if task.model_type == ModelType.PPO:
            agent = PPO(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_size=task.parameters.get("hidden_size", 256),
                lr=task.parameters.get("learning_rate", 3e-4),
                gamma=task.parameters.get("discount_factor", 0.99),
                lam=task.parameters.get("lambda", 0.95),
                clip_eps=task.parameters.get("clip_epsilon", 0.2),
                ent_coef=task.parameters.get("entropy_coef", 0.01),
                vf_coef=task.parameters.get("value_function_coef", 0.5),
                n_steps=task.parameters.get("n_steps", 2048),
                batch_size=task.parameters.get("batch_size", 64),
                n_epochs=task.parameters.get("n_epochs", 10),
                device=task.parameters.get("device", "cpu"),
                deployments=saved_deployments,
                max_replicas=saved_max_replicas
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
                deployments=saved_deployments,
                max_replicas=saved_max_replicas
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
                deployments=saved_deployments,
                max_replicas=saved_max_replicas
            )

        if agent is None:
            raise ValueError(f"Unknown model type: {task.model_type}")

        # Load trained model
        agent.load_model(model_path)

        # Initialize wandb for inference logging
        init_wandb(self.wandb_config, f"inference_{task.name}_{task_id}")

        # Run inference loop
        obs, _ = env.reset()  # FIXED: Initialize obs properly
        total_reward = 0
        latencies = []
        throughputs = []
        success_count = 0
        rewards = []

        for step in range(sample_size):
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Collect metrics
            total_reward += reward
            rewards.append(reward)
            latencies.append(info.get("latency", 0.0))
            throughputs.append(info.get("throughput", 0.0))
            
            # Count success (you can adjust this criteria)
            if reward > 0:  # Positive reward indicates success
                success_count += 1
                
            if terminated or truncated:
                obs, _ = env.reset()
                
        env.close()
        
        # Calculate comprehensive inference metrics
        avg_reward = total_reward / sample_size
        avg_latency = np.mean(latencies) if latencies else 0.0
        avg_throughput = np.mean(throughputs) if throughputs else 0.0
        success_rate = success_count / sample_size
        latency_std = np.std(latencies) if latencies else 0.0
        reward_std = np.std(rewards) if rewards else 0.0
        
        # Calculate adaptation score (how well model performs compared to random baseline)
        # Assuming random baseline would get negative rewards
        adaptation_score = max(0.0, (avg_reward + 100) / 100)  # Normalize to [0, 1+]
        
        metrics = {
            "avg_reward": float(avg_reward),
            "avg_latency": float(avg_latency),
            "avg_throughput": float(avg_throughput),
            "success_rate": float(success_rate),
            "latency_std": float(latency_std),
            "reward_std": float(reward_std),
            "adaptation_score": float(adaptation_score)
        }

        # Log inference metrics to wandb
        wandb.log({
            "inference/avg_reward": avg_reward,
            "inference/avg_latency": avg_latency,
            "inference/avg_throughput": avg_throughput,
            "inference/success_rate": success_rate,
            "inference/latency_stability": latency_std,
            "inference/adaptation_score": adaptation_score,
            "inference/sample_size": sample_size
        })
        
        result = ReconciliationResult(
            task_id=task_id,
            model_type=task.model_type,
            wandb_run_id=wandb.run.id,
            metrics=metrics,
            sample_size=sample_size,
            model_weights_path=model_path
        )
        
        # Finish wandb run
        finish_wandb()
        
        return await self.db.save_reconciliation_result(result)
        
    def _get_agent(self, task: TrainingTask, obs_dim: int, act_dim: int):
        """Helper function to get agent instance."""
        if task.model_type == ModelType.PPO:
            return PPO(obs_dim, act_dim)
        elif task.model_type == ModelType.SAC:
            return SAC(obs_dim, act_dim)
        elif task.model_type == ModelType.TD3:
            return TD3(obs_dim, act_dim)
        else:
            raise ValueError(f"Unsupported model type: {task.model_type}") 
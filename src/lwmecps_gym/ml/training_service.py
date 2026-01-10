from typing import Dict, Any, Optional, List
import wandb
from datetime import datetime
import logging
from ..core.database import Database
from ..core.models import (
    TrainingTask, TrainingResult, ReconciliationResult, ReconciliationTask, 
    TrainingState, ModelType, TransferTask, TransferResult, MetaTask, MetaResult,
    TransferType, MetaAlgorithm
)
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
from .models.transfer_learning import TransferLearningAgent, PPOTransferAgent, SACTransferAgent
from .models.maml_learning import MAMLAgent, TaskDistribution
from .models.fomaml_learning import FOMAMLAgent, ImplicitFOMAMLAgent
from .models.heuristic_baseline import HeuristicBaseline
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

def get_env_max_replicas(env) -> int:
    """
    Safely get max_replicas from environment, handling TimeLimit wrapper.
    
    Args:
        env: Gymnasium environment (may be wrapped in TimeLimit)
        
    Returns:
        max_replicas value from the unwrapped environment
    """
    # Try direct access first
    if hasattr(env, 'max_replicas'):
        return env.max_replicas
    
    # Try unwrapped (for TimeLimit and other wrappers)
    if hasattr(env, 'unwrapped'):
        return env.unwrapped.max_replicas
    
    # Try env.env (TimeLimit wrapper pattern)
    if hasattr(env, 'env') and hasattr(env.env, 'max_replicas'):
        return env.env.max_replicas
    
    # Fallback: calculate from action space
    if hasattr(env, 'action_space') and hasattr(env.action_space, 'nvec'):
        return int(max(env.action_space.nvec) - 1)
    
    raise AttributeError(f"Could not find max_replicas in environment of type {type(env)}")

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
            logger.info(f"Received state: {type(state)} with {len(state) if isinstance(state, dict) else 'unknown'} items")
            
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
                    logger.info(f"Node state: {type(node_state)} with keys: {list(node_state.keys()) if isinstance(node_state, dict) else 'unknown'}")
                    
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
                    logger.error(f"Error processing node {node}: {type(e).__name__}: {e}")
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

            # Start the workload
            try:
                if hasattr(env.unwrapped, "start_workload"):
                    logger.info("Starting workload via env.unwrapped.start_workload()")
                    env.unwrapped.start_workload()
                else:
                    logger.warning("Environment unwrapped does not have start_workload method")
            except Exception as e:
                logger.error(f"Failed to start workload: {e}")
                raise e

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
                logger.error(f"Failed to get environment dimensions: {type(e).__name__}: {e}")
                raise

            # Initialize appropriate agent based on model type
            logger.info(f"Initializing agent of type {task.model_type}")
            if task.model_type == ModelType.Q_LEARNING:
                logger.info("Creating Q-learning agent with parameters:")
                logger.info(f"  - Learning rate: {task.parameters.get('learning_rate', 0.1)}")
                logger.info(f"  - Discount factor: {task.parameters.get('discount_factor', 0.95)}")
                logger.info(f"  - Exploration rate: {task.parameters.get('exploration_rate', 1.0)}")
                logger.info(f"  - Exploration decay: {task.parameters.get('exploration_decay', 0.995)}")
                logger.info(f"  - Min exploration rate: {task.parameters.get('min_exploration_rate', 0.01)}")
                logger.info(f"  - Max states: {task.parameters.get('max_states', 10000)}")
                
                agent = QLearningAgent(
                    learning_rate=task.parameters.get("learning_rate", 0.1),
                    discount_factor=task.parameters.get("discount_factor", 0.95),
                    exploration_rate=task.parameters.get("exploration_rate", 1.0),
                    exploration_decay=task.parameters.get("exploration_decay", 0.995),
                    min_exploration_rate=task.parameters.get("min_exploration_rate", 0.01),
                    max_states=task.parameters.get("max_states", 10000)
                )
                logger.info("Q-learning agent created successfully")
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
                # Use max_replicas from environment to ensure consistency
                # The environment calculates it based on hardware constraints
                max_replicas = get_env_max_replicas(env)
                task_max_replicas = task.parameters.get("max_replicas")
                if task_max_replicas is not None and task_max_replicas != max_replicas:
                    logger.warning(
                        f"max_replicas mismatch: task parameters specify {task_max_replicas}, "
                        f"but environment calculates {max_replicas} based on hardware constraints. "
                        f"Using environment value {max_replicas} to ensure action space consistency."
                    )
                logger.info(f"Creating PPO agent with obs_dim={obs_dim}, act_dim={act_dim}, max_replicas={max_replicas} (from environment)")
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
                    max_replicas=max_replicas
                )
            elif task.model_type == ModelType.TD3:
                # Use max_replicas from environment to ensure consistency
                # The environment calculates it based on hardware constraints
                max_replicas = get_env_max_replicas(env)
                task_max_replicas = task.parameters.get("max_replicas")
                if task_max_replicas is not None and task_max_replicas != max_replicas:
                    logger.warning(
                        f"max_replicas mismatch: task parameters specify {task_max_replicas}, "
                        f"but environment calculates {max_replicas} based on hardware constraints. "
                        f"Using environment value {max_replicas} to ensure action space consistency."
                    )
                logger.info(f"Creating TD3 agent with obs_dim={obs_dim}, act_dim={act_dim}, max_replicas={max_replicas} (from environment)")
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
                # Use max_replicas from environment to ensure consistency
                # The environment calculates it based on hardware constraints
                max_replicas = get_env_max_replicas(env)
                task_max_replicas = task.parameters.get("max_replicas")
                if task_max_replicas is not None and task_max_replicas != max_replicas:
                    logger.warning(
                        f"max_replicas mismatch: task parameters specify {task_max_replicas}, "
                        f"but environment calculates {max_replicas} based on hardware constraints. "
                        f"Using environment value {max_replicas} to ensure action space consistency."
                    )
                logger.info(f"Creating SAC agent with obs_dim={obs_dim}, act_dim={act_dim}, max_replicas={max_replicas} (from environment)")
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
                    max_replicas=max_replicas
                )
            elif task.model_type == ModelType.HEURISTIC:
                # Use max_replicas from environment
                max_replicas = get_env_max_replicas(env)
                heuristic_type = task.parameters.get("heuristic_type", "uniform")
                static_replicas = task.parameters.get("static_replicas", None)
                
                logger.info(f"Creating HeuristicBaseline agent with type={heuristic_type}, max_replicas={max_replicas}")
                agent = HeuristicBaseline(
                    heuristic_type=heuristic_type,
                    num_deployments=act_dim,
                    max_replicas=max_replicas,
                    static_replicas=static_replicas
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
            elif task.model_type == ModelType.HEURISTIC:
                # Heuristic baseline (no actual training, just evaluation)
                logger.info(f"Running {task.parameters.get('heuristic_type', 'uniform')} heuristic baseline")
                results = agent.train(env, total_episodes=task.total_episodes, wandb_run_id=task.wandb_run_id)
            else:
                # Q-learning and DQN training
                if task.model_type == ModelType.Q_LEARNING:
                    logger.info(f"Starting Q-learning training for {task.total_episodes} episodes")
                    logger.info(f"Environment action space: {env.action_space}")
                    logger.info(f"Environment observation space: {env.observation_space}")
                
                results = agent.train(env, task.total_episodes, wandb_run_id=task.wandb_run_id)
                
                if task.model_type == ModelType.Q_LEARNING:
                    logger.info("Q-learning training completed successfully")
                    if results:
                        logger.info(f"Training results keys: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
                        if isinstance(results, dict) and 'accuracy' in results:
                            logger.info(f"Final accuracy: {results['accuracy'][-1] if results['accuracy'] else 'N/A'}")
                        if isinstance(results, dict) and 'total_reward' in results:
                            logger.info(f"Final total reward: {results['total_reward'][-1] if results['total_reward'] else 'N/A'}")

            # Save results
            if task.model_type in [ModelType.PPO, ModelType.TD3, ModelType.SAC, ModelType.HEURISTIC]:
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
                    elif task.model_type == ModelType.HEURISTIC:
                        # For heuristic, add latency metric
                        if "episode_latencies" in results and episode < len(results["episode_latencies"]):
                            metrics["avg_latency"] = results["episode_latencies"][episode]
                    future = asyncio.run_coroutine_threadsafe(
                        self.save_training_result(task_id, episode, metrics, db_connection=db_thread),
                        loop
                    )
                    future.result()
            else:
                # Q-learning and DQN results processing
                if task.model_type == ModelType.Q_LEARNING:
                    logger.info(f"Processing Q-learning results for {task.total_episodes} episodes")
                    logger.info(f"Q-learning results structure: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
                
                for episode in range(task.total_episodes):
                    metrics = {
                        "total_reward": results["episode_reward"][episode],
                        "steps": results["episode_steps"][episode],
                        "epsilon": results["episode_exploration"][episode],
                        "latency": results["episode_latency"][episode]
                    }
                    
                    if task.model_type == ModelType.Q_LEARNING and episode % 10 == 0:  # Log every 10th episode
                        logger.info(f"Q-learning episode {episode}: reward={metrics['total_reward']:.2f}, steps={metrics['steps']}, epsilon={metrics['epsilon']:.3f}")
                    
                    future = asyncio.run_coroutine_threadsafe(
                        self.save_training_result(task_id, episode, metrics, db_connection=db_thread),
                        loop
                    )
                    future.result()
                
                if task.model_type == ModelType.Q_LEARNING:
                    logger.info("Q-learning results processing completed")

            # Save model
            os.makedirs("./models", exist_ok=True)  # Ensure models directory exists
            if task.model_type == ModelType.Q_LEARNING:
                model_path = f"./models/model_{task.model_type.value}_{task_id}.pth"
                logger.info(f"Saving Q-learning model to {model_path}")
                logger.info(f"Q-table size before saving: {len(agent.q_table) if hasattr(agent, 'q_table') else 'Unknown'}")
                agent.save_model(model_path)
                logger.info("Q-learning model saved successfully")
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
            task.model_path = model_path  # Save model path to task

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
            # Safely convert exception to string to avoid numpy serialization issues
            try:
                error_msg = str(e)
            except Exception:
                error_msg = f"Error type: {type(e).__name__}"
            
            # Special logging for Q-learning errors
            if task.model_type == ModelType.Q_LEARNING:
                logger.error(f"Q-learning training failed for task {task_id}: {error_msg}")
                logger.error(f"Q-learning error type: {type(e).__name__}")
                if hasattr(e, '__traceback__'):
                    import traceback
                    logger.error(f"Q-learning traceback: {traceback.format_exc()}")
            else:
                logger.error(f"Error in training process for task {task_id}: {error_msg}")
            
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
    
    async def create_reconciliation_task(self, task_id: str, sample_size: int, group_id: Optional[str] = None) -> ReconciliationTask:
        """
        Create a new reconciliation task for the trained model.
        
        Args:
            task_id: ID of the training task
            sample_size: Number of steps to perform
            group_id: Optional experiment group ID
            
        Returns:
            Created ReconciliationTask instance
        """
        task = await self.db.get_training_task(task_id)
        if not task:
            raise ValueError(f"Training task {task_id} not found")
        
        if task.state != TrainingState.COMPLETED:
            raise ValueError(f"Training task {task_id} is not completed")
        
        # Check if model_path is set, if not generate it from pattern
        if not task.model_path:
            task.model_path = f"./models/model_{task.model_type.value}_{task_id}.pth"
            logger.info(f"Generated model_path: {task.model_path}")
            # Update the training task in database with model_path
            await self.db.update_training_task(task_id, task.model_dump())
        
        # Check if model file exists
        if not os.path.exists(task.model_path):
            raise ValueError(f"Model file not found at {task.model_path}")
        
        # Use provided group_id or generate a new one
        reconciliation_group_id = group_id if group_id is not None else f"reconciliation-{task_id}-{uuid.uuid4()}"
        
        reconciliation_task = ReconciliationTask(
            name=f"Reconciliation for {task.name}",
            description=f"Reconciliation task for training task {task_id}",
            training_task_id=task_id,
            model_type=task.model_type,
            total_steps=sample_size,
            group_id=reconciliation_group_id,
            namespace=task.namespace,
            max_pods=task.max_pods,
            base_url=task.base_url,
            stabilization_time=task.stabilization_time,
            model_path=task.model_path
        )
        
        return await self.db.create_reconciliation_task(reconciliation_task)
    
    async def create_reconciliation_task_from_wandb(
        self, 
        wandb_run_id: str,
        artifact_name: Optional[str] = None,
        sample_size: int = 100,
        group_id: Optional[str] = None,
        namespace: Optional[str] = None,
        max_pods: Optional[int] = None,
        base_url: Optional[str] = None,
        stabilization_time: Optional[int] = None
    ) -> ReconciliationTask:
        """
        Create a reconciliation task from a wandb artifact without requiring a training task in DB.
        
        Args:
            wandb_run_id: WandB run ID where the model artifact is stored
            artifact_name: Optional artifact name. If not provided, will search for model artifacts
            sample_size: Number of steps to perform
            group_id: Optional experiment group ID
            namespace: Optional namespace (defaults to "lwmecps-testapp")
            max_pods: Optional max pods (defaults to 50)
            base_url: Optional base URL (defaults to "http://34.51.217.76:8001")
            stabilization_time: Optional stabilization time (defaults to 10)
            
        Returns:
            Created ReconciliationTask instance
        """
        import wandb
        from wandb import Api
        
        logger.info(f"Creating reconciliation task from wandb run {wandb_run_id}")
        
        # Initialize wandb API
        api = Api()
        
        # Get the run
        try:
            project_name = self.wandb_config.project_name
            entity = self.wandb_config.entity
            if entity:
                run_path = f"{entity}/{project_name}/{wandb_run_id}"
            else:
                run_path = f"{project_name}/{wandb_run_id}"
            
            logger.info(f"Fetching wandb run: {run_path}")
            run = api.run(run_path)
        except Exception as e:
            raise ValueError(f"Failed to fetch wandb run {wandb_run_id}: {str(e)}")
        
        # Find model artifact
        artifacts = list(run.logged_artifacts())
        model_artifact = None
        
        if artifact_name:
            # Try to find artifact by exact name
            for art in artifacts:
                if art.name == artifact_name and art.type == 'model':
                    model_artifact = art
                    break
        else:
            # Find first model artifact
            for art in artifacts:
                if art.type == 'model':
                    model_artifact = art
                    break
        
        if not model_artifact:
            raise ValueError(f"Model artifact not found in wandb run {wandb_run_id}. Available artifacts: {[a.name for a in artifacts]}")
        
        logger.info(f"Found model artifact: {model_artifact.name}")
        
        # Download artifact
        artifact_dir = model_artifact.download()
        logger.info(f"Artifact downloaded to: {artifact_dir}")
        
        # Get artifact metadata
        metadata = model_artifact.metadata or {}
        logger.info(f"Artifact metadata: {metadata}")
        
        # Extract model type from metadata or artifact name
        model_type_str = metadata.get('model_type', '')
        if not model_type_str:
            # Try to infer from artifact name
            if 'ppo' in model_artifact.name.lower():
                model_type_str = 'ppo'
            elif 'sac' in model_artifact.name.lower():
                model_type_str = 'sac'
            elif 'td3' in model_artifact.name.lower():
                model_type_str = 'td3'
            elif 'dqn' in model_artifact.name.lower():
                model_type_str = 'dqn'
            elif 'q_learning' in model_artifact.name.lower() or 'q-learning' in model_artifact.name.lower():
                model_type_str = 'q_learning'
            else:
                raise ValueError(f"Could not determine model type from artifact {model_artifact.name}. Please specify artifact_name or ensure metadata contains model_type.")
        
        try:
            model_type = ModelType(model_type_str)
        except ValueError:
            raise ValueError(f"Invalid model type: {model_type_str}. Supported types: {[e.value for e in ModelType]}")
        
        # Find model file in artifact directory
        model_files = []
        for root, dirs, files in os.walk(artifact_dir):
            for file in files:
                if file.endswith('.pth') or file.endswith('.pt'):
                    model_files.append(os.path.join(root, file))
        
        if not model_files:
            raise ValueError(f"No model file (.pth or .pt) found in artifact {model_artifact.name}")
        
        model_path = model_files[0]  # Use first model file found
        if len(model_files) > 1:
            logger.warning(f"Multiple model files found, using: {model_path}")
        
        logger.info(f"Using model file: {model_path}")
        
        # Load checkpoint to verify and extract dimensions
        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            saved_obs_dim = checkpoint.get('obs_dim')
            saved_act_dim = checkpoint.get('act_dim')
            if saved_obs_dim is None or saved_act_dim is None:
                logger.warning(f"Model checkpoint missing obs_dim or act_dim, will use defaults")
        except Exception as e:
            logger.warning(f"Could not load checkpoint to verify dimensions: {e}")
        
        # Extract parameters from metadata or use defaults
        env_config = metadata.get('env_config', {})
        parameters = metadata.get('parameters', {})
        
        # Use provided values or defaults from metadata or hardcoded defaults
        reconciliation_group_id = group_id if group_id is not None else f"reconciliation-wandb-{wandb_run_id}-{uuid.uuid4()}"
        namespace_val = namespace if namespace is not None else env_config.get('namespace', 'lwmecps-testapp')
        max_pods_val = max_pods if max_pods is not None else env_config.get('max_pods', 50)
        base_url_val = base_url if base_url is not None else env_config.get('base_url', 'http://34.51.217.76:8001')
        stabilization_time_val = stabilization_time if stabilization_time is not None else env_config.get('stabilization_time', 10)
        
        # Copy model to local models directory for consistency
        os.makedirs("./models", exist_ok=True)
        local_model_path = f"./models/wandb_model_{model_type.value}_{wandb_run_id}.pth"
        import shutil
        shutil.copy2(model_path, local_model_path)
        logger.info(f"Model copied to local path: {local_model_path}")
        
        reconciliation_task = ReconciliationTask(
            name=f"Reconciliation from WandB {wandb_run_id}",
            description=f"Reconciliation task for model from wandb run {wandb_run_id}, artifact {model_artifact.name}",
            training_task_id=None,  # No training task in DB
            model_type=model_type,
            total_steps=sample_size,
            group_id=reconciliation_group_id,
            namespace=namespace_val,
            max_pods=max_pods_val,
            base_url=base_url_val,
            stabilization_time=stabilization_time_val,
            model_path=local_model_path
        )
        
        return await self.db.create_reconciliation_task(reconciliation_task)
    
    async def start_reconciliation_task(self, task_id: str) -> Optional[ReconciliationTask]:
        """
        Start a reconciliation task.
        
        Args:
            task_id: Unique identifier of the reconciliation task
            
        Returns:
            Updated ReconciliationTask instance if successful, None otherwise
        """
        task = await self.db.get_reconciliation_task(task_id)
        if not task:
            return None

        task.state = TrainingState.RUNNING
        await self.db.update_reconciliation_task(task_id, task.model_dump())

        # Start reconciliation in a separate thread, passing the running event loop
        loop = asyncio.get_running_loop()
        thread = threading.Thread(target=self._run_reconciliation, args=(task_id, task, self, loop))
        thread.start()

        return task
    
    def _run_reconciliation(self, task_id: str, task: ReconciliationTask, service_instance, loop):
        """
        The actual reconciliation process, designed to be run in a thread.
        
        Args:
            task_id: Unique identifier of the reconciliation task
            task: The ReconciliationTask instance
            service_instance: The instance of TrainingService to call back to
            loop: The asyncio event loop from the main thread
        """
        env = None
        db_thread = None
        try:
            # Create a new DB connection for this thread
            db_thread = Database()

            # Initialize wandb and update task by scheduling on the main loop
            init_wandb(self.wandb_config, f"reconciliation_{task.name}_{task_id}")
            task.wandb_run_id = wandb.run.id
            
            future = asyncio.run_coroutine_threadsafe(
                db_thread.update_reconciliation_task(task_id, task.model_dump()), 
                loop
            )
            future.result() # Wait for the update to complete

            # Update the service's main task object as well
            service_instance.reconciliation_task = task

            # Convert task_id to ObjectId for MongoDB operations
            task_id_obj = ObjectId(task_id)
            logger.info(f"Starting reconciliation process for task {task_id}")
            
            # Get the original training task if available
            training_task = None
            if task.training_task_id:
                training_task_future = asyncio.run_coroutine_threadsafe(
                    db_thread.get_training_task(str(task.training_task_id)), 
                    loop
                )
                training_task = training_task_future.result()
                
                if not training_task:
                    logger.warning(f"Training task {task.training_task_id} not found, proceeding without it")
            
            # Determine model path
            model_path = task.model_path
            if not model_path:
                if training_task:
                    model_path = training_task.model_path or f"./models/model_{training_task.model_type.value}_{training_task.id}.pth"
                else:
                    raise ValueError("Model path not specified in reconciliation task and no training task available")
            
            if not os.path.exists(model_path):
                raise ValueError(f"Model file not found: {model_path}")
            
            # Load model checkpoint to get dimensions
            logger.info(f"Loading checkpoint from: {model_path}")
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            saved_obs_dim = checkpoint.get('obs_dim')
            saved_act_dim = checkpoint.get('act_dim')
            saved_deployments = checkpoint.get('deployments', [])
            saved_max_replicas = checkpoint.get('max_replicas', 10)
            
            # Extract hidden_size from checkpoint if available, or infer from network structure
            saved_hidden_size = checkpoint.get('hidden_size')
            if saved_hidden_size is None:
                # Try to infer from actor network structure
                if 'actor' in checkpoint:
                    actor_state = checkpoint['actor']
                    # Look for first linear layer weight (net.0.weight)
                    if 'net.0.weight' in actor_state:
                        # Shape is [hidden_size, obs_dim]
                        saved_hidden_size = actor_state['net.0.weight'].shape[0]
                        logger.info(f"Inferred hidden_size={saved_hidden_size} from actor network structure")
                    elif '0.weight' in actor_state:
                        # Some implementations use a plain Sequential without a 'net.' prefix
                        saved_hidden_size = actor_state['0.weight'].shape[0]
                        logger.info(f"Inferred hidden_size={saved_hidden_size} from actor network structure (0.weight)")
                elif 'model_state_dict' in checkpoint:
                    # For PPO/TD3 models
                    model_state = checkpoint['model_state_dict']
                    # Look for first layer weight
                    for key, tensor in model_state.items():
                        # TD3/SAC-style networks
                        if 'net.0.weight' in key or 'actor.net.0.weight' in key:
                            saved_hidden_size = tensor.shape[0]
                            logger.info(f"Inferred hidden_size={saved_hidden_size} from model structure ({key})")
                            break
                        # PPO ActorCritic uses nn.Sequential under 'actor'/'critic' directly:
                        # actor.0.weight / critic.0.weight
                        if key.endswith('actor.0.weight') or 'actor.0.weight' in key:
                            saved_hidden_size = tensor.shape[0]
                            logger.info(f"Inferred hidden_size={saved_hidden_size} from PPO ActorCritic ({key})")
                            break
                        if key.endswith('critic.0.weight') or 'critic.0.weight' in key:
                            saved_hidden_size = tensor.shape[0]
                            logger.info(f"Inferred hidden_size={saved_hidden_size} from PPO ActorCritic ({key})")
                            break
            
            if saved_hidden_size is None:
                # For NN-based policies, silently falling back to defaults defeats the purpose of reconciliation.
                # Fail fast with a clear message to force a 1:1 architecture restore.
                if task.model_type in (ModelType.PPO, ModelType.TD3, ModelType.SAC):
                    raise ValueError(
                        "Checkpoint does not contain 'hidden_size' and it could not be inferred from 'model_state_dict'. "
                        "Reconciliation requires exact architecture match. "
                        "Please re-save the model checkpoint including 'hidden_size' (and ideally 'max_replicas')."
                    )
                logger.warning(
                    "Could not infer hidden_size from checkpoint (non-NN model type); continuing without it. "
                    f"Available keys: {list(checkpoint.keys())}"
                )
            
            logger.info(f"Raw saved_deployments: {saved_deployments} (type: {type(saved_deployments)})")
            
            if saved_obs_dim is None or saved_act_dim is None:
                raise ValueError("Model checkpoint missing required dimensions")
            
            logger.info(f"Loaded model dimensions: obs_dim={saved_obs_dim}, act_dim={saved_act_dim}, hidden_size={saved_hidden_size}")
            logger.info(f"Loaded deployments: {saved_deployments}")
            logger.info(f"Loaded max_replicas: {saved_max_replicas}")

            # Create a minimal training_task-like object if not available
            if not training_task:
                from types import SimpleNamespace
                # Try to extract parameters from checkpoint metadata
                checkpoint_params = {}
                try:
                    # Checkpoint already loaded above, reuse it
                    # Some checkpoints may have parameters stored
                    if 'parameters' in checkpoint:
                        checkpoint_params = checkpoint.get('parameters', {})
                    # Add hidden_size to params if we extracted it
                    if saved_hidden_size is not None:
                        checkpoint_params['hidden_size'] = saved_hidden_size
                except:
                    pass
                
                training_task = SimpleNamespace(
                    model_type=task.model_type,
                    parameters=checkpoint_params,  # Use parameters from checkpoint if available
                    namespace=task.namespace,
                    max_pods=task.max_pods,
                    base_url=task.base_url,
                    stabilization_time=task.stabilization_time,
                    env_config={},
                    model_path=model_path
                )

            result = self._perform_reconciliation(
                training_task, task, saved_obs_dim, saved_act_dim, 
                saved_deployments, saved_max_replicas, saved_hidden_size, model_path,
                db_thread, loop, task_id
            )
            
            # Update task state to completed
            task.state = TrainingState.COMPLETED
            task.progress = 100.0
            future = asyncio.run_coroutine_threadsafe(
                db_thread.update_reconciliation_task(task_id, task.model_dump()), 
                loop
            )
            future.result()
            
        except Exception as e:
            logger.error(f"Error in reconciliation thread: {type(e).__name__}: {e}")
            # Update task state to failed
            task.state = TrainingState.FAILED
            task.error_message = f"{type(e).__name__}: {e}"
            task.progress = 0.0
            future = asyncio.run_coroutine_threadsafe(
                db_thread.update_reconciliation_task(task_id, task.model_dump()), 
                loop
            )
            future.result()
        finally:
            # Clean up
            if env:
                env.close()
            if db_thread:
                asyncio.run_coroutine_threadsafe(db_thread.close(), loop)
            finish_wandb()

    def _perform_reconciliation(self, training_task: Any, reconciliation_task: ReconciliationTask, 
                               saved_obs_dim: int, saved_act_dim: int, saved_deployments: List[str], 
                               saved_max_replicas: int, saved_hidden_size: Optional[int], model_path: str, db_thread, loop, task_id: str) -> ReconciliationResult:
        """
        Perform the actual reconciliation process.
        """
        # Get Kubernetes state - same approach as in training
        logger.info("Fetching Kubernetes cluster state...")
        state = self.minikube.k8s_state()
        logger.info(f"Received state: {type(state)} with {len(state) if isinstance(state, dict) else 'unknown'} items")
        
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

        # Создаем информацию о узлах - same approach as in training
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
        
        # Create environment for reconciliation
        logger.info("Creating environment")
        env_config_dict = getattr(training_task, 'env_config', {}) or {}
        if not isinstance(env_config_dict, dict):
            env_config_dict = {}
        env = gym.make(
            env_config_dict.get("env_name", "lwmecps-v3"),
            node_name=list(node_info.keys()),
            max_hardware=max_hardware,
            pod_usage=pod_usage,
            node_info=node_info,
            num_nodes=len(node_info),
            namespace=reconciliation_task.namespace,
            deployments=saved_deployments,  # Use deployments from saved model
            max_pods=reconciliation_task.max_pods,
            group_id=reconciliation_task.group_id,
            base_url=reconciliation_task.base_url,
            stabilization_time=reconciliation_task.stabilization_time
        )
        logger.info("Environment created successfully")

        # Start the workload
        try:
            if hasattr(env.unwrapped, "start_workload"):
                logger.info("Starting workload via env.unwrapped.start_workload()")
                env.unwrapped.start_workload()
            else:
                logger.warning("Environment unwrapped does not have start_workload method")
        except Exception as e:
            logger.error(f"Failed to start workload: {e}")
            raise e

        # Use dimensions from saved model instead of calculating from environment
        obs_dim = saved_obs_dim
        act_dim = saved_act_dim

        # Get parameters safely (works with both TrainingTask and SimpleNamespace)
        params = getattr(training_task, 'parameters', {}) or {}
        if not isinstance(params, dict):
            params = {}
        
        # Use hidden_size from checkpoint if available, otherwise from params, otherwise default
        hidden_size = saved_hidden_size if saved_hidden_size is not None else params.get("hidden_size", 256)
        logger.info(f"Using hidden_size={hidden_size} (from checkpoint: {saved_hidden_size}, from params: {params.get('hidden_size')}, default: 256)")
        
        # Get model type safely
        model_type = getattr(training_task, 'model_type', reconciliation_task.model_type)
        
        # Create agent with correct parameters
        agent = None
        if model_type == ModelType.PPO:
            # Use max_replicas from environment to ensure consistency
            max_replicas = get_env_max_replicas(env)
            logger.info(f"Creating PPO agent with max_replicas={max_replicas} (from environment)")
            agent = PPO(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_size=hidden_size,
                lr=params.get("learning_rate", 3e-4),
                gamma=params.get("discount_factor", 0.99),
                lam=params.get("lambda", 0.95),
                clip_eps=params.get("clip_epsilon", 0.2),
                ent_coef=params.get("entropy_coef", 0.01),
                vf_coef=params.get("value_function_coef", 0.5),
                n_steps=params.get("n_steps", 2048),
                batch_size=params.get("batch_size", 64),
                n_epochs=params.get("n_epochs", 10),
                device=params.get("device", "cpu"),
                deployments=saved_deployments,
                max_replicas=max_replicas
            )
        elif model_type == ModelType.SAC:
            # Use max_replicas from environment to ensure consistency
            # The environment calculates it based on hardware constraints
            max_replicas = get_env_max_replicas(env)
            logger.info(f"Creating SAC agent with max_replicas={max_replicas} (from environment)")
            agent = SAC(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_size=hidden_size,
                lr=params.get("learning_rate", 3e-4),
                gamma=params.get("discount_factor", 0.99),
                tau=params.get("tau", 0.005),
                alpha=params.get("alpha", 0.2),
                auto_entropy=params.get("auto_entropy", True),
                target_entropy=params.get("target_entropy", -1.0),
                batch_size=params.get("batch_size", 256),
                device=params.get("device", "cpu"),
                deployments=saved_deployments,
                max_replicas=max_replicas
            )
        elif model_type == ModelType.TD3:
            # Use max_replicas from environment to ensure consistency
            max_replicas = get_env_max_replicas(env)
            logger.info(f"Creating TD3 agent with max_replicas={max_replicas} (from environment)")
            agent = TD3(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_size=hidden_size,
                lr=params.get("learning_rate", 3e-4),
                gamma=params.get("discount_factor", 0.99),
                tau=params.get("tau", 0.005),
                policy_delay=params.get("policy_delay", 2),
                noise_clip=params.get("noise_clip", 0.5),
                noise=params.get("noise", 0.2),
                batch_size=params.get("batch_size", 256),
                device=params.get("device", "cpu"),
                deployments=saved_deployments,
                max_replicas=max_replicas
            )
        elif model_type == ModelType.Q_LEARNING:
            agent = QLearningAgent(
                learning_rate=params.get("learning_rate", 0.1),
                discount_factor=params.get("discount_factor", 0.95),
                exploration_rate=params.get("exploration_rate", 1.0),
                exploration_decay=params.get("exploration_decay", 0.995),
                min_exploration_rate=params.get("min_exploration_rate", 0.01),
                max_states=params.get("max_states", 10000)
            )
        elif model_type == ModelType.DQN:
            agent = DQNAgent(
                env,
                learning_rate=params.get("learning_rate", 0.001),
                discount_factor=params.get("discount_factor", 0.99),
                epsilon=params.get("epsilon", 0.1),
                memory_size=params.get("memory_size", 10000),
                batch_size=params.get("batch_size", 32)
            )

        if agent is None:
            model_type_val = getattr(training_task, 'model_type', reconciliation_task.model_type)
            raise ValueError(f"Unknown model type: {model_type_val}")

        # Load trained model
        agent.load_model(model_path)

        # Run inference loop
        obs, _ = env.reset()
        total_reward = 0
        latencies = []
        throughputs = []
        success_count = 0
        rewards = []

        for step in range(reconciliation_task.total_steps):
            # Handle different agent types' select_action return values
            model_type = getattr(training_task, 'model_type', reconciliation_task.model_type)
            if model_type == ModelType.PPO:
                # PPO returns (action, log_prob, value)
                action, log_prob, value = agent.select_action(obs)
            elif model_type in [ModelType.Q_LEARNING, ModelType.DQN]:
                # Q-learning and DQN use choose_action method
                action = agent.choose_action(obs)
            else:
                # SAC and TD3 return just action
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
                
            # Log step-level metrics to wandb
            wandb.log({
                "reconciliation/step": step,
                "reconciliation/reward": reward,
                "reconciliation/latency": info.get("latency", 0.0),
                "reconciliation/throughput": info.get("throughput", 0.0),
                "reconciliation/cumulative_reward": total_reward,
                "reconciliation/success_rate": success_count / (step + 1)
            })
            
            # Update progress in database
            if step % 10 == 0:  # Update every 10 steps
                progress = (step / reconciliation_task.total_steps) * 100
                reconciliation_task.current_step = step
                reconciliation_task.progress = progress
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        db_thread.update_reconciliation_task(task_id, reconciliation_task.model_dump()), 
                        loop
                    )
                    future.result(timeout=5)
                except Exception as e:
                    logger.error(f"Failed to update reconciliation progress: {type(e).__name__}: {e}")
                    
            if terminated or truncated:
                obs, _ = env.reset()
                
        env.close()
        
        # Calculate comprehensive inference metrics
        avg_reward = total_reward / reconciliation_task.total_steps
        avg_latency = np.mean(latencies) if latencies else 0.0
        avg_throughput = np.mean(throughputs) if throughputs else 0.0
        success_rate = success_count / reconciliation_task.total_steps
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

        # Log final metrics to wandb
        wandb.log({
            "reconciliation/final_avg_reward": avg_reward,
            "reconciliation/final_avg_latency": avg_latency,
            "reconciliation/final_avg_throughput": avg_throughput,
            "reconciliation/final_success_rate": success_rate,
            "reconciliation/final_latency_stability": latency_std,
            "reconciliation/final_adaptation_score": adaptation_score,
            "reconciliation/total_steps": reconciliation_task.total_steps
        })
        
        result = ReconciliationResult(
            task_id=str(reconciliation_task.training_task_id) if reconciliation_task.training_task_id else "wandb-reconciliation",
            model_type=reconciliation_task.model_type,
            wandb_run_id=wandb.run.id,
            metrics=metrics,
            sample_size=reconciliation_task.total_steps,
            model_weights_path=model_path
        )
        
        return result

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
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Extract dimensions from saved model
        saved_obs_dim = checkpoint.get('obs_dim', 100)  # fallback to 100
        saved_act_dim = checkpoint.get('act_dim', 4)    # fallback to 4
        saved_deployments = checkpoint.get('deployments', [])
        saved_max_replicas = checkpoint.get('max_replicas', 10)
        
        logger.info(f"Loaded model dimensions: obs_dim={saved_obs_dim}, act_dim={saved_act_dim}")
        
        # Use provided group_id or fallback to task's original group_id
        reconciliation_group_id = group_id if group_id is not None else task.group_id
        logger.info(f"Running reconciliation for task {task_id} with group_id: {reconciliation_group_id}")
        
        # Get Kubernetes state - same approach as in training
        logger.info("Fetching Kubernetes cluster state...")
        state = self.minikube.k8s_state()
        logger.info(f"Received state: {type(state)} with {len(state) if isinstance(state, dict) else 'unknown'} items")
        
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

        # Создаем информацию о узлах - same approach as in training
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
        
        # Create environment for reconciliation
        logger.info("Creating environment")
        env = gym.make(
            task.env_config.get("env_name", "lwmecps-v3"),
            node_name=list(node_info.keys()),
            max_hardware=max_hardware,
            pod_usage=pod_usage,
            node_info=node_info,
            num_nodes=len(node_info),
            namespace=task.namespace,
            deployments=saved_deployments,  # Use deployments from saved model
            max_pods=task.max_pods,
            group_id=reconciliation_group_id,
            base_url=task.base_url,
            stabilization_time=task.stabilization_time
        )
        logger.info("Environment created successfully")

        # Start the workload
        try:
            if hasattr(env.unwrapped, "start_workload"):
                logger.info("Starting workload via env.unwrapped.start_workload()")
                env.unwrapped.start_workload()
            else:
                logger.warning("Environment unwrapped does not have start_workload method")
        except Exception as e:
            logger.error(f"Failed to start workload: {e}")
            raise e

        # Use dimensions from saved model instead of calculating from environment
        obs_dim = saved_obs_dim
        act_dim = saved_act_dim

        # Create agent with correct parameters
        agent = None
        if task.model_type == ModelType.PPO:
            # Use max_replicas from environment to ensure consistency
            max_replicas = get_env_max_replicas(env)
            logger.info(f"Creating PPO agent with max_replicas={max_replicas} (from environment)")
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
                max_replicas=max_replicas
            )
        elif task.model_type == ModelType.SAC:
            # Use max_replicas from environment to ensure consistency
            # The environment calculates it based on hardware constraints
            max_replicas = get_env_max_replicas(env)
            logger.info(f"Creating SAC agent with max_replicas={max_replicas} (from environment)")
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
                max_replicas=max_replicas
            )
        elif task.model_type == ModelType.TD3:
            # Use max_replicas from environment to ensure consistency
            max_replicas = get_env_max_replicas(env)
            logger.info(f"Creating TD3 agent with max_replicas={max_replicas} (from environment)")
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
                max_replicas=max_replicas
            )
        elif task.model_type == ModelType.Q_LEARNING:
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
            # Handle different agent types' select_action return values
            if task.model_type == ModelType.PPO:
                # PPO returns (action, log_prob, value)
                action, log_prob, value = agent.select_action(obs)
            elif task.model_type in [ModelType.Q_LEARNING, ModelType.DQN]:
                # Q-learning and DQN use choose_action method
                action = agent.choose_action(obs)
            else:
                # SAC and TD3 return just action
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
        
    async def pause_reconciliation_task(self, task_id: str) -> Optional[ReconciliationTask]:
        """Pause a reconciliation task."""
        task = await self.db.get_reconciliation_task(task_id)
        if not task or task.state != TrainingState.RUNNING:
            return None
        
        task.state = TrainingState.PAUSED
        return await self.db.update_reconciliation_task(task_id, task.model_dump())
    
    async def stop_reconciliation_task(self, task_id: str) -> Optional[ReconciliationTask]:
        """Stop a reconciliation task."""
        task = await self.db.get_reconciliation_task(task_id)
        if not task:
            return None
        
        task.state = TrainingState.FAILED
        task.error_message = "Stopped by user"
        return await self.db.update_reconciliation_task(task_id, task.model_dump())
    
    async def get_reconciliation_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get current reconciliation progress."""
        task = await self.db.get_reconciliation_task(task_id)
        if not task:
            return None
        
        return {
            "id": task.id,
            "state": task.state,
            "progress": task.progress,
            "current_step": task.current_step,
            "total_steps": task.total_steps,
            "error_message": task.error_message,
            "created_at": task.created_at,
            "updated_at": task.updated_at
        }

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

    # Transfer Learning Methods
    async def create_transfer_task(self, task_data: Dict[str, Any]) -> TransferTask:
        """
        Create a new transfer learning task.
        
        Args:
            task_data: Dictionary containing transfer learning parameters
            
        Returns:
            Created TransferTask instance
        """
        # Validate source and target tasks exist
        source_task = await self.db.get_training_task(task_data["source_task_id"])
        target_task = await self.db.get_training_task(task_data["target_task_id"])
        
        if not source_task:
            raise ValueError(f"Source task {task_data['source_task_id']} not found")
        if not target_task:
            raise ValueError(f"Target task {task_data['target_task_id']} not found")
        
        # Generate unique group_id if not provided
        if "group_id" not in task_data:
            task_data["group_id"] = f"transfer-{uuid.uuid4()}"
        
        transfer_task = TransferTask(
            name=task_data.get("name", f"Transfer from {source_task.name} to {target_task.name}"),
            description=task_data.get("description", ""),
            source_task_id=task_data["source_task_id"],
            target_task_id=task_data["target_task_id"],
            transfer_type=task_data.get("transfer_type", TransferType.FINE_TUNING),
            frozen_layers=task_data.get("frozen_layers", []),
            learning_rate=task_data.get("learning_rate", 1e-4),
            total_episodes=task_data.get("total_episodes", 50),
            group_id=task_data["group_id"],
            namespace=task_data.get("namespace", "lwmecps-testapp"),
            max_pods=task_data.get("max_pods", 50),
            base_url=task_data.get("base_url", "http://34.51.217.76:8001"),
            stabilization_time=task_data.get("stabilization_time", 10)
        )
        
        return await self.db.create_transfer_task(transfer_task)
    
    async def start_transfer_training(self, task_id: str) -> Optional[TransferTask]:
        """Start transfer learning training."""
        task = await self.db.get_transfer_task(task_id)
        if not task:
            return None
        
        task.state = TrainingState.RUNNING
        await self.db.update_transfer_task(task_id, task.model_dump())
        
        # Start training in a separate thread
        loop = asyncio.get_running_loop()
        thread = threading.Thread(target=self._run_transfer_training, args=(task_id, task, self, loop))
        thread.start()
        
        return task
    
    def _run_transfer_training(self, task_id: str, task: TransferTask, service_instance, loop):
        """Run transfer learning training in a separate thread."""
        env = None
        db_thread = None
        try:
            # Create new DB connection for this thread
            db_thread = Database()
            
            # Initialize wandb
            init_wandb(self.wandb_config, f"transfer_{task.name}_{task_id}")
            task.wandb_run_id = wandb.run.id
            
            # Update task with wandb run ID
            future = asyncio.run_coroutine_threadsafe(
                db_thread.update_transfer_task(task_id, task.model_dump()),
                loop
            )
            future.result()
            
            # Get source and target tasks
            source_task_future = asyncio.run_coroutine_threadsafe(
                db_thread.get_training_task(str(task.source_task_id)),
                loop
            )
            target_task_future = asyncio.run_coroutine_threadsafe(
                db_thread.get_training_task(str(task.target_task_id)),
                loop
            )
            
            source_task = source_task_future.result()
            target_task = target_task_future.result()
            
            if not source_task or not target_task:
                raise ValueError("Source or target task not found")
            
            # Create environment for target task
            env = self._create_environment_for_task(target_task)
            
            # Use max_replicas from environment to ensure consistency
            max_replicas = get_env_max_replicas(env)
            logger.info(f"Creating transfer learning agent with max_replicas={max_replicas} (from environment)")
            
            # Create transfer learning agent
            source_model_path = source_task.model_path or f"./models/model_{source_task.model_type.value}_{source_task.id}.pth"
            
            if source_task.model_type == ModelType.PPO:
                agent = PPOTransferAgent(
                    source_model_path=source_model_path,
                    target_obs_dim=env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else 100,
                    target_act_dim=env.action_space.shape[0] if hasattr(env.action_space, 'shape') else 4,
                    transfer_type=task.transfer_type.value,
                    frozen_layers=task.frozen_layers,
                    learning_rate=task.learning_rate,
                    max_replicas=max_replicas,
                    deployments=target_task.parameters.get("deployments", [])
                )
            elif source_task.model_type == ModelType.SAC:
                agent = SACTransferAgent(
                    source_model_path=source_model_path,
                    target_obs_dim=env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else 100,
                    target_act_dim=env.action_space.shape[0] if hasattr(env.action_space, 'shape') else 4,
                    transfer_type=task.transfer_type.value,
                    frozen_layers=task.frozen_layers,
                    learning_rate=task.learning_rate,
                    max_replicas=max_replicas,
                    deployments=target_task.parameters.get("deployments", [])
                )
            else:
                raise ValueError(f"Transfer learning not supported for model type: {source_task.model_type}")
            
            # Run training
            results = agent.train(
                env,
                total_episodes=task.total_episodes,
                wandb_run_id=task.wandb_run_id,
                training_service=service_instance,
                task_id=task_id,
                loop=loop,
                db_connection=db_thread
            )
            
            # Save model
            model_path = f"./models/transfer_{task.transfer_type.value}_{task_id}.pth"
            agent.save_model(model_path)
            
            # Update task state
            task.state = TrainingState.COMPLETED
            task.model_path = model_path
            
            future = asyncio.run_coroutine_threadsafe(
                db_thread.update_transfer_task(task_id, task.model_dump()),
                loop
            )
            future.result()
            
        except Exception as e:
            logger.error(f"Error in transfer training: {str(e)}")
            task.state = TrainingState.FAILED
            task.error_message = str(e)
            if db_thread:
                future = asyncio.run_coroutine_threadsafe(
                    db_thread.update_transfer_task(task_id, task.model_dump()),
                    loop
                )
                future.result()
        finally:
            if env:
                env.close()
            if db_thread:
                future = asyncio.run_coroutine_threadsafe(db_thread.close(), loop)
                future.result()
            finish_wandb()
    
    # Meta-Learning Methods
    async def create_meta_task(self, task_data: Dict[str, Any]) -> MetaTask:
        """
        Create a new meta-learning task.
        
        Args:
            task_data: Dictionary containing meta-learning parameters
            
        Returns:
            Created MetaTask instance
        """
        # Generate unique group_id if not provided
        if "group_id" not in task_data:
            task_data["group_id"] = f"meta-{task_data['meta_algorithm']}-{uuid.uuid4()}"
        
        meta_task = MetaTask(
            name=task_data.get("name", f"Meta-learning with {task_data['meta_algorithm']}"),
            description=task_data.get("description", ""),
            meta_algorithm=task_data["meta_algorithm"],
            inner_lr=task_data.get("inner_lr", 0.01),
            outer_lr=task_data.get("outer_lr", 0.001),
            adaptation_steps=task_data.get("adaptation_steps", 5),
            meta_batch_size=task_data.get("meta_batch_size", 4),
            task_distribution=task_data.get("task_distribution", {}),
            total_episodes=task_data.get("total_episodes", 100),
            group_id=task_data["group_id"],
            namespace=task_data.get("namespace", "lwmecps-testapp"),
            max_pods=task_data.get("max_pods", 50),
            base_url=task_data.get("base_url", "http://34.51.217.76:8001"),
            stabilization_time=task_data.get("stabilization_time", 10)
        )
        
        return await self.db.create_meta_task(meta_task)
    
    async def start_meta_training(self, task_id: str) -> Optional[MetaTask]:
        """Start meta-learning training."""
        task = await self.db.get_meta_task(task_id)
        if not task:
            return None
        
        task.state = TrainingState.RUNNING
        await self.db.update_meta_task(task_id, task.model_dump())
        
        # Start training in a separate thread
        loop = asyncio.get_running_loop()
        thread = threading.Thread(target=self._run_meta_training, args=(task_id, task, self, loop))
        thread.start()
        
        return task
    
    def _run_meta_training(self, task_id: str, task: MetaTask, service_instance, loop):
        """Run meta-learning training in a separate thread."""
        env = None
        db_thread = None
        try:
            # Create new DB connection for this thread
            db_thread = Database()
            
            # Initialize wandb
            init_wandb(self.wandb_config, f"meta_{task.meta_algorithm}_{task.name}_{task_id}")
            task.wandb_run_id = wandb.run.id
            
            # Update task with wandb run ID
            future = asyncio.run_coroutine_threadsafe(
                db_thread.update_meta_task(task_id, task.model_dump()),
                loop
            )
            future.result()
            
            # Create task distribution
            task_distribution = TaskDistribution(
                task_types=["kubernetes_scaling", "load_balancing", "resource_optimization"],
                task_params={
                    "kubernetes_scaling": {
                        "max_pods": 20,
                        "cpu_requirement": 1.0,
                        "memory_requirement": 1000,
                        "scaling_strategy": "horizontal"
                    },
                    "load_balancing": {
                        "traffic_pattern": "uniform",
                        "latency_threshold": 0.5,
                        "balancing_algorithm": "round_robin"
                    },
                    "resource_optimization": {
                        "optimization_target": "cpu",
                        "constraint_weight": 0.5,
                        "optimization_method": "gradient_based"
                    }
                },
                num_tasks=10
            )
            
            # Create meta-learning agent
            if task.meta_algorithm == MetaAlgorithm.MAML:
                agent = MAMLAgent(
                    obs_dim=100,  # Will be determined dynamically
                    act_dim=4,    # Will be determined dynamically
                    hidden_size=256,
                    inner_lr=task.inner_lr,
                    outer_lr=task.outer_lr,
                    adaptation_steps=task.adaptation_steps,
                    meta_batch_size=task.meta_batch_size,
                    task_distribution=task_distribution,
                    max_replicas=task.max_pods,
                    deployments=["lwmecps-testapp-server-bs1", "lwmecps-testapp-server-bs2", 
                               "lwmecps-testapp-server-bs3", "lwmecps-testapp-server-bs4"]
                )
            elif task.meta_algorithm == MetaAlgorithm.FOMAML:
                agent = FOMAMLAgent(
                    obs_dim=100,
                    act_dim=4,
                    hidden_size=256,
                    inner_lr=task.inner_lr,
                    outer_lr=task.outer_lr,
                    adaptation_steps=task.adaptation_steps,
                    meta_batch_size=task.meta_batch_size,
                    task_distribution=task_distribution,
                    max_replicas=task.max_pods,
                    deployments=["lwmecps-testapp-server-bs1", "lwmecps-testapp-server-bs2", 
                               "lwmecps-testapp-server-bs3", "lwmecps-testapp-server-bs4"]
                )
            elif task.meta_algorithm == MetaAlgorithm.IMPLICIT_FOMAML:
                agent = ImplicitFOMAMLAgent(
                    obs_dim=100,
                    act_dim=4,
                    hidden_size=256,
                    inner_lr=task.inner_lr,
                    outer_lr=task.outer_lr,
                    adaptation_steps=task.adaptation_steps,
                    meta_batch_size=task.meta_batch_size,
                    task_distribution=task_distribution,
                    max_replicas=task.max_pods,
                    deployments=["lwmecps-testapp-server-bs1", "lwmecps-testapp-server-bs2", 
                               "lwmecps-testapp-server-bs3", "lwmecps-testapp-server-bs4"]
                )
            else:
                raise ValueError(f"Unsupported meta-algorithm: {task.meta_algorithm}")
            
            # Create environment factory
            def env_factory(task_config):
                return self._create_environment_for_meta_task(task_config, task)
            
            # Run meta-training
            results = agent.meta_train(
                env_factory=env_factory,
                meta_episodes=task.total_episodes,
                wandb_run_id=task.wandb_run_id,
                training_service=service_instance,
                task_id=task_id,
                loop=loop,
                db_connection=db_thread
            )
            
            # Save model
            model_path = f"./models/meta_{task.meta_algorithm.value}_{task_id}.pth"
            agent.save_model(model_path)
            
            # Update task state
            task.state = TrainingState.COMPLETED
            task.model_path = model_path
            
            future = asyncio.run_coroutine_threadsafe(
                db_thread.update_meta_task(task_id, task.model_dump()),
                loop
            )
            future.result()
            
        except Exception as e:
            logger.error(f"Error in meta training: {str(e)}")
            task.state = TrainingState.FAILED
            task.error_message = str(e)
            if db_thread:
                future = asyncio.run_coroutine_threadsafe(
                    db_thread.update_meta_task(task_id, task.model_dump()),
                    loop
                )
                future.result()
        finally:
            if env:
                env.close()
            if db_thread:
                future = asyncio.run_coroutine_threadsafe(db_thread.close(), loop)
                future.result()
            finish_wandb()
    
    def _create_environment_for_task(self, task: TrainingTask) -> gym.Env:
        """Create environment for a specific task."""
        # This is a simplified version - in practice you'd use task parameters
        return gym.make(
            "lwmecps-v3",
            node_name=["node1", "node2"],
            max_hardware={"cpu": 8, "ram": 16000, "tx_bandwidth": 1000, "rx_bandwidth": 1000, 
                         "read_disks_bandwidth": 500, "write_disks_bandwidth": 500, "avg_latency": 300},
            pod_usage={"cpu": 2, "ram": 2000, "tx_bandwidth": 20, "rx_bandwidth": 20, 
                      "read_disks_bandwidth": 100, "write_disks_bandwidth": 100},
            node_info={"node1": {"cpu": 4, "ram": 8000, "tx_bandwidth": 100, "rx_bandwidth": 100, 
                                "read_disks_bandwidth": 300, "write_disks_bandwidth": 300, "avg_latency": 10},
                      "node2": {"cpu": 4, "ram": 8000, "tx_bandwidth": 100, "rx_bandwidth": 100, 
                                "read_disks_bandwidth": 300, "write_disks_bandwidth": 300, "avg_latency": 20}},
            num_nodes=2,
            namespace=task.namespace,
            deployments=["lwmecps-testapp-server-bs1", "lwmecps-testapp-server-bs2", 
                       "lwmecps-testapp-server-bs3", "lwmecps-testapp-server-bs4"],
            max_pods=task.max_pods,
            group_id=task.group_id,
            env_config={"base_url": task.base_url, "stabilization_time": task.stabilization_time}
        )
    
    def _create_environment_for_meta_task(self, task_config: Dict[str, Any], meta_task: MetaTask) -> gym.Env:
        """Create environment for a meta-learning task."""
        # Create environment based on task configuration
        return gym.make(
            "lwmecps-v3",
            node_name=["node1", "node2"],
            max_hardware={"cpu": 8, "ram": 16000, "tx_bandwidth": 1000, "rx_bandwidth": 1000, 
                         "read_disks_bandwidth": 500, "write_disks_bandwidth": 500, "avg_latency": 300},
            pod_usage={"cpu": 2, "ram": 2000, "tx_bandwidth": 20, "rx_bandwidth": 20, 
                      "read_disks_bandwidth": 100, "write_disks_bandwidth": 100},
            node_info={"node1": {"cpu": 4, "ram": 8000, "tx_bandwidth": 100, "rx_bandwidth": 100, 
                                "read_disks_bandwidth": 300, "write_disks_bandwidth": 300, "avg_latency": 10},
                      "node2": {"cpu": 4, "ram": 8000, "tx_bandwidth": 100, "rx_bandwidth": 100, 
                                "read_disks_bandwidth": 300, "write_disks_bandwidth": 300, "avg_latency": 20}},
            num_nodes=2,
            namespace=meta_task.namespace,
            deployments=["lwmecps-testapp-server-bs1", "lwmecps-testapp-server-bs2", 
                       "lwmecps-testapp-server-bs3", "lwmecps-testapp-server-bs4"],
            max_pods=meta_task.max_pods,
            group_id=meta_task.group_id,
            env_config={"base_url": meta_task.base_url, "stabilization_time": meta_task.stabilization_time}
        ) 
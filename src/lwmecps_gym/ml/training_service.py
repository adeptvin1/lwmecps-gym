from typing import Dict, Any, Optional
import wandb
from datetime import datetime
import logging
from ..core.database import Database
from ..core.models import TrainingTask, TrainingResult, ReconciliationResult, TrainingState
from ..core.wandb_config import init_wandb, log_metrics, log_model, finish_wandb
from ..core.wandb_config import WandbConfig
import gymnasium as gym
from kubernetes import client, config
from kubernetes.client import CoreV1Api

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        result = TrainingResult(
            task_id=task_id,
            episode=episode,
            metrics=metrics,
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
        
        # Download model weights from Weights & Biases
        api = wandb.Api()
        run = api.run(f"{self.wandb_config.project_name}/{task.wandb_run_id}")
        model_weights = run.file("model_weights.pth").download()
        
        # Initialize Weights & Biases for reconciliation
        init_wandb(self.wandb_config)
        
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
        Internal method to run the actual training process.
        
        Args:
            task_id: Unique identifier of the training task
            task: TrainingTask instance containing training parameters
        """
        try:
            logger.info(f"Starting training process for task {task_id}")
            
            # Register custom Gym environment
            logger.info("Registering environment...")
            gym.envs.register(
                id="lwmecps-v0",
                entry_point="lwmecps_gym.envs:LWMECPSEnv",
            )
            
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
            
            # Create Gym environment
            logger.info("Creating Gym environment...")
            env = gym.make(
                "lwmecps-v0",
                num_nodes=len(node_names),
                node_name=node_names,
                max_hardware=max_hardware,
                pod_usage=pod_usage,
                node_info={},  # We'll need to implement a way to get node info
                deployment_name="mec-test-app",
                namespace="default",
                deployments=["mec-test-app"],
                max_pods=10000,
            )
            logger.info("Environment created successfully")
            
            # Create and train Q-Learning agent
            logger.info(f"Creating Q-Learning agent with parameters: {task.parameters}")
            agent = QLearningAgent(env, **task.parameters)
            
            # Run training episodes
            logger.info(f"Starting training for {task.total_episodes} episodes...")
            results = agent.train(episodes=task.total_episodes)
            logger.info(f"Training completed with results: {results}")
            
            # Save trained model
            model_path = f"./models/q_table_{task_id}.pkl"
            logger.info(f"Saving model to {model_path}")
            agent.save_q_table(model_path)
            
            # Save training results
            logger.info("Saving training result to database")
            training_result = TrainingResult(
                task_id=task_id,
                episode=task.total_episodes,
                metrics=results,
                wandb_run_id=task.wandb_run_id,
                model_weights_path=model_path
            )
            await self.db.save_training_result(training_result)
            
            # Update task status
            logger.info("Updating task status to completed")
            await self.db.update_training_task(task_id, {
                "current_episode": task.total_episodes,
                "progress": 1.0,
                "metrics": results,
                "state": TrainingState.COMPLETED
            })
            
            logger.info(f"Training completed for task {task_id}")
            
        except Exception as e:
            logger.error(f"Error in training task {task_id}: {str(e)}", exc_info=True)
            await self.db.update_training_task(task_id, {
                "state": TrainingState.FAILED,
                "error_message": str(e)
            })
        finally:
            # Cleanup
            self.active_tasks[task_id] = False
            finish_wandb()
            logger.info(f"Training process finished for task {task_id}") 
from typing import Dict, Any, Optional
import wandb
from datetime import datetime
import logging
from ..core.database import Database
from ..core.models import TrainingTask, TrainingResult, ReconciliationResult, TrainingState
from ..core.wandb_config import init_wandb, log_metrics, log_model, finish_wandb
from ..core.wandb_config import WandbConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingService:
    def __init__(self, db: Database, wandb_config: WandbConfig):
        self.db = db
        self.wandb_config = wandb_config
        self.active_tasks: Dict[str, bool] = {}
    
    async def create_training_task(self, task_data: Dict[str, Any]) -> TrainingTask:
        """Create a new training task"""
        task = TrainingTask(**task_data)
        return await self.db.create_training_task(task)
    
    async def start_training(self, task_id: str) -> Optional[TrainingTask]:
        """Start a training task"""
        logger.info(f"Attempting to start training task {task_id}")
        
        # Get the task
        task = await self.db.get_training_task(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return None
            
        logger.info(f"Found task {task_id} in state {task.state}")
        
        # Check if task is in a valid state to start
        if task.state != TrainingState.PENDING:
            logger.error(f"Task {task_id} is in state {task.state}, cannot start")
            return None
        
        try:
            # Initialize wandb
            init_wandb(self.wandb_config)
            logger.info(f"Initialized wandb run with ID {wandb.run.id}")
            
            # Update task state
            task.state = TrainingState.RUNNING
            task.wandb_run_id = wandb.run.id
            task = await self.db.update_training_task(task_id, task.model_dump())
            
            if not task:
                logger.error(f"Failed to update task {task_id} state")
                finish_wandb()
                return None
                
            logger.info(f"Successfully started task {task_id}")
            
            # Start training in background
            self.active_tasks[task_id] = True
            # Here you would start your actual training loop
            
            return task
            
        except Exception as e:
            logger.error(f"Error starting task {task_id}: {str(e)}")
            finish_wandb()
            return None
    
    async def pause_training(self, task_id: str) -> Optional[TrainingTask]:
        """Pause a training task"""
        task = await self.db.get_training_task(task_id)
        if not task or task.state != TrainingState.RUNNING:
            return None
        
        self.active_tasks[task_id] = False
        task.state = TrainingState.PAUSED
        return await self.db.update_training_task(task_id, task.model_dump())
    
    async def resume_training(self, task_id: str) -> Optional[TrainingTask]:
        """Resume a paused training task"""
        task = await self.db.get_training_task(task_id)
        if not task or task.state != TrainingState.PAUSED:
            return None
        
        self.active_tasks[task_id] = True
        task.state = TrainingState.RUNNING
        return await self.db.update_training_task(task_id, task.model_dump())
    
    async def stop_training(self, task_id: str) -> Optional[TrainingTask]:
        """Stop a training task"""
        task = await self.db.get_training_task(task_id)
        if not task or task.state not in [TrainingState.RUNNING, TrainingState.PAUSED]:
            return None
        
        self.active_tasks[task_id] = False
        task.state = TrainingState.FAILED
        finish_wandb()
        return await self.db.update_training_task(task_id, task.model_dump())
    
    async def save_training_result(self, task_id: str, episode: int, metrics: Dict[str, float]) -> TrainingResult:
        """Save training results for an episode"""
        task = await self.db.get_training_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        result = TrainingResult(
            task_id=task_id,
            episode=episode,
            metrics=metrics,
            wandb_run_id=task.wandb_run_id
        )
        
        # Log metrics to wandb
        log_metrics(metrics, step=episode)
        
        return await self.db.save_training_result(result)
    
    async def run_reconciliation(self, task_id: str, sample_size: int) -> ReconciliationResult:
        """Run model reconciliation on new data"""
        task = await self.db.get_training_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        # Download model weights from wandb
        api = wandb.Api()
        run = api.run(f"{self.wandb_config.project_name}/{task.wandb_run_id}")
        model_weights = run.file("model_weights.pth").download()
        
        # Initialize wandb for reconciliation
        init_wandb(self.wandb_config)
        
        # Here you would run your reconciliation logic
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
        """Get current training progress"""
        task = await self.db.get_training_task(task_id)
        if not task:
            return None
        
        results = await self.db.get_training_results(task_id)
        
        return {
            "task": task,
            "latest_metrics": results[-1].metrics if results else None,
            "is_active": self.active_tasks.get(task_id, False)
        } 
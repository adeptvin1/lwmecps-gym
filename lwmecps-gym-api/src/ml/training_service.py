from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
from bson.objectid import ObjectId
from ..models.database import Database
from ..models.wandb_config import init_wandb, log_metrics, log_model, finish_wandb
from ..models.models import TrainingTask, TrainingResult, ReconciliationResult, TrainingState
import asyncio

logger = logging.getLogger(__name__)

class TrainingService:
    def __init__(self, db: Database):
        self.db = db
        self._training_tasks: Dict[str, asyncio.Task] = {}

    async def create_training_task(self, task_data: Dict[str, Any]) -> TrainingTask:
        """Create a new training task"""
        try:
            # Create TrainingTask instance
            task = TrainingTask(**task_data)
            
            # Save to database
            task = await self.db.create_training_task(task)
            
            logger.info(f"Created training task: {task.id}")
            return task
        except Exception as e:
            logger.error(f"Failed to create training task: {str(e)}")
            raise

    async def start_training(self, task_id: str) -> Optional[TrainingTask]:
        """Start a training task"""
        try:
            # Convert task_id to ObjectId
            object_id = ObjectId(task_id)

            # Get task from database
            task = await self.db.get_training_task(object_id)
            if not task:
                return None

            # Check if task can be started
            if task.state != TrainingState.PENDING:
                return None

            # Update task state
            task.state = TrainingState.RUNNING
            task = await self.db.update_training_task(object_id, {"state": task.state})

            # Initialize wandb
            init_wandb()

            # Start training in background
            training_task = asyncio.create_task(self._run_training(task_id, task))
            self._training_tasks[task_id] = training_task

            return task
        except Exception as e:
            logger.error(f"Failed to start training task: {str(e)}")
            raise

    async def pause_training(self, task_id: str) -> Optional[TrainingTask]:
        """Pause a training task"""
        try:
            # Convert task_id to ObjectId
            object_id = ObjectId(task_id)

            # Get task from database
            task = await self.db.get_training_task(object_id)
            if not task or task.state != TrainingState.RUNNING:
                return None

            # Update task state
            task.state = TrainingState.PAUSED
            return await self.db.update_training_task(object_id, {"state": task.state})
        except Exception as e:
            logger.error(f"Failed to pause training task: {str(e)}")
            raise

    async def resume_training(self, task_id: str) -> Optional[TrainingTask]:
        """Resume a paused training task"""
        try:
            # Convert task_id to ObjectId
            object_id = ObjectId(task_id)

            # Get task from database
            task = await self.db.get_training_task(object_id)
            if not task or task.state != TrainingState.PAUSED:
                return None

            # Update task state
            task.state = TrainingState.RUNNING
            return await self.db.update_training_task(object_id, {"state": task.state})
        except Exception as e:
            logger.error(f"Failed to resume training task: {str(e)}")
            raise

    async def stop_training(self, task_id: str) -> Optional[TrainingTask]:
        """Stop a training task"""
        try:
            # Convert task_id to ObjectId
            object_id = ObjectId(task_id)

            # Get task from database
            task = await self.db.get_training_task(object_id)
            if not task or task.state not in [TrainingState.RUNNING, TrainingState.PAUSED]:
                return None

            # Update task state
            task.state = TrainingState.FAILED
            return await self.db.update_training_task(object_id, {"state": task.state})
        except Exception as e:
            logger.error(f"Failed to stop training task: {str(e)}")
            raise

    async def save_training_result(self, task_id: str, episode: int, metrics: Dict[str, float]) -> TrainingResult:
        """Save training result"""
        try:
            # Convert task_id to ObjectId
            object_id = ObjectId(task_id)

            # Create TrainingResult instance
            result = TrainingResult(
                task_id=object_id,
                episode=episode,
                metrics=metrics
            )

            # Save to database
            return await self.db.save_training_result(result)
        except Exception as e:
            logger.error(f"Failed to save training result: {str(e)}")
            raise

    async def run_reconciliation(self, task_id: str, sample_size: int) -> ReconciliationResult:
        """Run model reconciliation"""
        try:
            # Convert task_id to ObjectId
            object_id = ObjectId(task_id)

            # Get task from database
            task = await self.db.get_training_task(object_id)
            if not task:
                raise ValueError("Task not found")

            # Create ReconciliationResult instance
            result = ReconciliationResult(
                task_id=object_id,
                model_type=task.model_type,
                metrics={},  # Add actual metrics here
                sample_size=sample_size
            )

            # Save to database
            return await self.db.save_reconciliation_result(result)
        except Exception as e:
            logger.error(f"Failed to run reconciliation: {str(e)}")
            raise

    async def get_training_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get current training progress"""
        try:
            # Convert task_id to ObjectId
            object_id = ObjectId(task_id)

            task = await self.db.get_training_task(object_id)
            if not task:
                return None

            return {
                "state": task.state,
                "current_episode": task.current_episode,
                "total_episodes": task.total_episodes,
                "progress": task.progress,
                "metrics": task.metrics
            }
        except Exception as e:
            logger.error(f"Failed to get training progress: {str(e)}")
            raise

    async def _run_training(self, task_id: str, task: TrainingTask):
        """Run training process"""
        try:
            # Convert task_id to ObjectId
            object_id = ObjectId(task_id)

            # Training loop
            for episode in range(task.current_episode, task.total_episodes):
                # Check if training should be paused
                current_task = await self.db.get_training_task(object_id)
                if current_task.state == TrainingState.PAUSED:
                    await asyncio.sleep(1)
                    continue

                # Check if training should be stopped
                if current_task.state == TrainingState.FAILED:
                    break

                # Training step
                metrics = await self._train_step(task, episode)

                # Save result
                training_result = TrainingResult(
                    task_id=object_id,
                    episode=episode,
                    metrics=metrics
                )
                await self.db.save_training_result(training_result)

                # Log metrics
                log_metrics(metrics)

                # Update task progress
                progress = (episode + 1) / task.total_episodes
                await self.db.update_training_task(object_id, {
                    "current_episode": episode + 1,
                    "progress": progress
                })

            # Mark task as completed
            await self.db.update_training_task(object_id, {
                "state": TrainingState.COMPLETED
            })

            # Finish wandb run
            finish_wandb()

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            await self.db.update_training_task(object_id, {
                "state": TrainingState.FAILED,
                "error_message": str(e)
            })
            finish_wandb()
            raise

    async def _train_step(self, task: TrainingTask, episode: int) -> Dict[str, float]:
        """Perform a single training step"""
        # Implement actual training logic here
        # This is a placeholder
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "reward": 0.0
        } 
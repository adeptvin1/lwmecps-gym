"""
API endpoints for meta-learning functionality.

This module provides REST API endpoints for managing meta-learning
training tasks, adaptation to new tasks, and monitoring progress.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import List, Dict, Any, Optional
from ...core.models import TrainingTask, TrainingResult, TrainingState, ModelType
from ...core.database import Database
from ...core.wandb_config import WandbConfig
from ...ml.meta_learning_service import MetaLearningService
import logging
from bson.objectid import ObjectId
import os

logger = logging.getLogger(__name__)
router = APIRouter()

async def get_db():
    db = Database()
    try:
        yield db
    finally:
        await db.close()

async def get_wandb_config():
    return WandbConfig(
        api_key=os.getenv("WANDB_API_KEY", ""),
        project_name=os.getenv("WANDB_PROJECT", "lwmecps-gym"),
        entity=os.getenv("WANDB_ENTITY", "")
    )

async def get_meta_learning_service(
    db: Database = Depends(get_db),
    wandb_config: WandbConfig = Depends(get_wandb_config)
) -> MetaLearningService:
    return MetaLearningService(db, wandb_config)

@router.post("/meta-tasks", response_model=TrainingTask)
async def create_meta_training_task(
    task_data: Dict[str, Any] = Body(..., description="Meta-learning task configuration"),
    service: MetaLearningService = Depends(get_meta_learning_service)
):
    """
    Create a new meta-learning training task.
    
    Args:
        task_data: Dictionary containing:
            - model_type: Type of base algorithm (META_PPO, META_SAC, META_TD3, META_DQN)
            - meta_method: Meta-learning method ("maml" or "fomaml")
            - tasks: List of task configurations for meta-learning
            - meta_parameters: Meta-learning specific parameters
            - parameters: Base algorithm parameters
            - env_config: Environment configuration
            - name: Task name
            - description: Task description
            - group_id: Experiment group ID
            - total_episodes: Total number of episodes per task
            - namespace: Kubernetes namespace
            - max_pods: Maximum number of pods
            - base_url: Base URL for environment
            - stabilization_time: Stabilization time after pod movements
    
    Returns:
        Created TrainingTask instance
    """
    try:
        # Validate required fields
        if "model_type" not in task_data:
            raise HTTPException(status_code=400, detail="model_type is required")
        
        if task_data["model_type"] not in [ModelType.META_PPO, ModelType.META_SAC, ModelType.META_TD3, ModelType.META_DQN]:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported model type for meta-learning: {task_data['model_type']}. "
                       f"Supported types: {[ModelType.META_PPO, ModelType.META_SAC, ModelType.META_TD3, ModelType.META_DQN]}"
            )
        
        if "tasks" not in task_data or not task_data["tasks"]:
            raise HTTPException(status_code=400, detail="tasks parameter is required and must not be empty")
        
        return await service.create_meta_training_task(task_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating meta-learning task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating meta-learning task: {str(e)}")

@router.get("/meta-tasks", response_model=List[TrainingTask])
async def list_meta_training_tasks(
    skip: int = Query(0, description="Number of tasks to skip"),
    limit: int = Query(10, description="Maximum number of tasks to return"),
    db: Database = Depends(get_db)
):
    """
    List all meta-learning training tasks.
    
    Args:
        skip: Number of tasks to skip
        limit: Maximum number of tasks to return
    
    Returns:
        List of TrainingTask instances
    """
    try:
        # Filter for meta-learning tasks only
        meta_types = [ModelType.META_PPO, ModelType.META_SAC, ModelType.META_TD3, ModelType.META_DQN]
        tasks = await db.list_training_tasks(skip, limit)
        return [task for task in tasks if task.model_type in meta_types]
    except Exception as e:
        logger.error(f"Error listing meta-learning tasks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing meta-learning tasks: {str(e)}")

@router.get("/meta-tasks/{task_id}", response_model=TrainingTask)
async def get_meta_training_task(
    task_id: str,
    db: Database = Depends(get_db)
):
    """
    Get a specific meta-learning training task.
    
    Args:
        task_id: Unique identifier of the training task
    
    Returns:
        TrainingTask instance
    """
    try:
        task = await db.get_training_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Meta-learning task not found")
        
        # Check if it's a meta-learning task
        meta_types = [ModelType.META_PPO, ModelType.META_SAC, ModelType.META_TD3, ModelType.META_DQN]
        if task.model_type not in meta_types:
            raise HTTPException(status_code=404, detail="Task is not a meta-learning task")
        
        return task
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting meta-learning task {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting meta-learning task: {str(e)}")

@router.post("/meta-tasks/{task_id}/start", response_model=TrainingTask)
async def start_meta_training(
    task_id: str,
    service: MetaLearningService = Depends(get_meta_learning_service)
):
    """
    Start a meta-learning training task.
    
    Args:
        task_id: Unique identifier of the training task
    
    Returns:
        Updated TrainingTask instance
    """
    try:
        task = await service.start_meta_training(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Meta-learning task not found or cannot be started")
        return task
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error starting meta-learning task {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting meta-learning task: {str(e)}")

@router.get("/meta-tasks/{task_id}/progress")
async def get_meta_training_progress(
    task_id: str,
    service: MetaLearningService = Depends(get_meta_learning_service)
):
    """
    Get the current progress of a meta-learning training task.
    
    Args:
        task_id: Unique identifier of the training task
    
    Returns:
        Dictionary containing progress information
    """
    try:
        progress = await service.get_meta_training_progress(task_id)
        if not progress:
            raise HTTPException(status_code=404, detail="Meta-learning task not found")
        return progress
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting meta-learning progress for task {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting meta-learning progress: {str(e)}")

@router.post("/meta-tasks/{task_id}/adapt")
async def adapt_to_new_task(
    task_id: str,
    new_task_config: Dict[str, Any] = Body(..., description="New task configuration for adaptation"),
    service: MetaLearningService = Depends(get_meta_learning_service)
):
    """
    Adapt a trained meta-learning model to a new task.
    
    Args:
        task_id: ID of the trained meta-learning task
        new_task_config: Configuration for the new task, containing:
            - node_name: List of node names
            - max_hardware: Hardware specifications
            - pod_usage: Pod resource usage
            - node_info: Node information
            - num_nodes: Number of nodes
            - deployments: List of deployments
            - adaptation_episodes: Number of episodes for adaptation
    
    Returns:
        Dictionary of adaptation metrics
    """
    try:
        adaptation_metrics = await service.adapt_to_new_task(task_id, new_task_config)
        return {
            "task_id": task_id,
            "adaptation_metrics": adaptation_metrics,
            "message": "Successfully adapted to new task"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error adapting meta-learning model {task_id} to new task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adapting to new task: {str(e)}")

@router.get("/meta-tasks/{task_id}/results", response_model=List[TrainingResult])
async def get_meta_training_results(
    task_id: str,
    db: Database = Depends(get_db)
):
    """
    Get training results for a meta-learning task.
    
    Args:
        task_id: Unique identifier of the training task
    
    Returns:
        List of TrainingResult instances
    """
    try:
        return await db.get_training_results(task_id)
    except Exception as e:
        logger.error(f"Error getting meta-learning results for task {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting meta-learning results: {str(e)}")

@router.delete("/meta-tasks/{task_id}")
async def delete_meta_training_task(
    task_id: str,
    db: Database = Depends(get_db)
):
    """
    Delete a specific meta-learning training task.
    
    Args:
        task_id: Unique identifier of the training task
    
    Returns:
        Success message
    """
    try:
        # First check if task exists and is a meta-learning task
        task = await db.get_training_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Meta-learning task not found")
        
        meta_types = [ModelType.META_PPO, ModelType.META_SAC, ModelType.META_TD3, ModelType.META_DQN]
        if task.model_type not in meta_types:
            raise HTTPException(status_code=404, detail="Task is not a meta-learning task")
        
        # Delete the task and its associated results
        await db.db.training_tasks.delete_one({"_id": ObjectId(task_id)})
        await db.db.training_results.delete_many({"task_id": ObjectId(task_id)})
        
        logger.info(f"Successfully deleted meta-learning task {task_id} and its associated results")
        return {"message": f"Meta-learning task {task_id} and its associated results deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting meta-learning task {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting meta-learning task: {str(e)}")

@router.get("/meta-tasks/{task_id}/metrics")
async def get_meta_learning_metrics(
    task_id: str,
    service: MetaLearningService = Depends(get_meta_learning_service)
):
    """
    Get detailed metrics for a meta-learning training task.
    
    Args:
        task_id: Unique identifier of the training task
    
    Returns:
        Dictionary containing detailed metrics
    """
    try:
        progress = await service.get_meta_training_progress(task_id)
        if not progress:
            raise HTTPException(status_code=404, detail="Meta-learning task not found")
        
        # Extract metrics from progress
        metrics = progress.get("metrics", {})
        
        return {
            "task_id": task_id,
            "meta_method": progress.get("meta_method", "unknown"),
            "num_tasks": progress.get("num_tasks", 0),
            "meta_parameters": progress.get("meta_parameters", {}),
            "training_metrics": {
                "meta_losses": metrics.get("meta_losses", []),
                "adaptation_metrics": metrics.get("adaptation_metrics", []),
                "episode_rewards": metrics.get("episode_rewards", []),
                "episode_lengths": metrics.get("episode_lengths", [])
            },
            "state": progress.get("state", "unknown"),
            "wandb_run_id": progress.get("wandb_run_id")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting meta-learning metrics for task {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting meta-learning metrics: {str(e)}")

@router.get("/supported-meta-algorithms")
async def get_supported_meta_algorithms():
    """
    Get list of supported meta-learning algorithms and methods.
    
    Returns:
        Dictionary containing supported algorithms and methods
    """
    return {
        "supported_algorithms": [
            {
                "type": ModelType.META_PPO,
                "name": "Meta PPO",
                "description": "Proximal Policy Optimization with meta-learning",
                "base_algorithm": "PPO"
            },
            {
                "type": ModelType.META_SAC,
                "name": "Meta SAC", 
                "description": "Soft Actor-Critic with meta-learning",
                "base_algorithm": "SAC"
            },
            {
                "type": ModelType.META_TD3,
                "name": "Meta TD3",
                "description": "Twin Delayed Deep Deterministic Policy Gradient with meta-learning",
                "base_algorithm": "TD3"
            },
            {
                "type": ModelType.META_DQN,
                "name": "Meta DQN",
                "description": "Deep Q-Network with meta-learning",
                "base_algorithm": "DQN"
            }
        ],
        "supported_methods": [
            {
                "method": "maml",
                "name": "MAML",
                "description": "Model-Agnostic Meta-Learning - learns good initialization parameters"
            },
            {
                "method": "fomaml", 
                "name": "FOMAML",
                "description": "First-Order MAML - simplified version of MAML using only first-order gradients"
            }
        ]
    }

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
from ...core.models import TrainingTask, TrainingResult, ReconciliationResult, ReconciliationTask
from ...core.database import Database
from ...core.wandb_config import WandbConfig
from ...ml.training_service import TrainingService
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

async def get_training_service(
    db: Database = Depends(get_db),
    wandb_config: WandbConfig = Depends(get_wandb_config)
) -> TrainingService:
    return TrainingService(db, wandb_config)

@router.post("/tasks", response_model=TrainingTask)
async def create_training_task(
    task_data: Dict[str, Any],
    service: TrainingService = Depends(get_training_service)
):
    """Create a new training task"""
    return await service.create_training_task(task_data)

@router.get("/tasks", response_model=List[TrainingTask])
async def list_training_tasks(
    skip: int = 0,
    limit: int = 10,
    db: Database = Depends(get_db)
):
    """List all training tasks"""
    return await db.list_training_tasks(skip, limit)

@router.get("/tasks/{task_id}", response_model=TrainingTask)
async def get_training_task(
    task_id: str,
    db: Database = Depends(get_db)
):
    """Get a specific training task"""
    task = await db.get_training_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@router.post("/tasks/{task_id}/start", response_model=TrainingTask)
async def start_training(
    task_id: str,
    service: TrainingService = Depends(get_training_service)
):
    """Start a training task"""
    task = await service.start_training(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found or cannot be started")
    return task

@router.post("/tasks/{task_id}/pause", response_model=TrainingTask)
async def pause_training(
    task_id: str,
    service: TrainingService = Depends(get_training_service)
):
    """Pause a training task"""
    task = await service.pause_training(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found or cannot be paused")
    return task

@router.post("/tasks/{task_id}/resume", response_model=TrainingTask)
async def resume_training(
    task_id: str,
    service: TrainingService = Depends(get_training_service)
):
    """Resume a paused training task"""
    task = await service.resume_training(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found or cannot be resumed")
    return task

@router.post("/tasks/{task_id}/stop", response_model=TrainingTask)
async def stop_training(
    task_id: str,
    service: TrainingService = Depends(get_training_service)
):
    """Stop a training task"""
    task = await service.stop_training(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found or cannot be stopped")
    return task

@router.get("/tasks/{task_id}/results", response_model=List[TrainingResult])
async def get_training_results(
    task_id: str,
    db: Database = Depends(get_db)
):
    """Get training results for a task"""
    return await db.get_training_results(task_id)

@router.post("/tasks/{task_id}/reconcile", response_model=ReconciliationTask)
async def create_reconciliation_task(
    task_id: str,
    sample_size: int = Query(
        ..., 
        description="Количество шагов для выполнения reconciliation",
        example=100,
        ge=1
    ),
    group_id: Optional[str] = Query(
        None, 
        description="ID группы экспериментов для reconciliation. Если не указан, используется group_id из задачи обучения",
        example="reconciliation-group-1"
    ),
    service: TrainingService = Depends(get_training_service)
):
    """
    Создание задачи reconciliation для обученной модели
    
    Создает задачу reconciliation для проверки работы обученной модели 
    на новых данных или в новой среде.
    
    Args:
        task_id: ID задачи обучения
        sample_size: Количество шагов для выполнения
        group_id: Опциональный ID группы экспериментов
        
    Returns:
        Созданная задача reconciliation
    """
    try:
        return await service.create_reconciliation_task(task_id, sample_size, group_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# Reconciliation Tasks Management
@router.get("/reconciliation-tasks", response_model=List[ReconciliationTask])
async def list_reconciliation_tasks(
    skip: int = 0,
    limit: int = 10,
    db: Database = Depends(get_db)
):
    """List all reconciliation tasks"""
    return await db.list_reconciliation_tasks(skip, limit)

@router.get("/reconciliation-tasks/{task_id}", response_model=ReconciliationTask)
async def get_reconciliation_task(
    task_id: str,
    db: Database = Depends(get_db)
):
    """Get a specific reconciliation task"""
    task = await db.get_reconciliation_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Reconciliation task not found")
    return task

@router.post("/reconciliation-tasks/{task_id}/start", response_model=ReconciliationTask)
async def start_reconciliation_task(
    task_id: str,
    service: TrainingService = Depends(get_training_service)
):
    """Start a reconciliation task"""
    task = await service.start_reconciliation_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Reconciliation task not found or cannot be started")
    return task

@router.post("/reconciliation-tasks/{task_id}/pause", response_model=ReconciliationTask)
async def pause_reconciliation_task(
    task_id: str,
    service: TrainingService = Depends(get_training_service)
):
    """Pause a reconciliation task"""
    task = await service.pause_reconciliation_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Reconciliation task not found or cannot be paused")
    return task

@router.post("/reconciliation-tasks/{task_id}/stop", response_model=ReconciliationTask)
async def stop_reconciliation_task(
    task_id: str,
    service: TrainingService = Depends(get_training_service)
):
    """Stop a reconciliation task"""
    task = await service.stop_reconciliation_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Reconciliation task not found or cannot be stopped")
    return task

@router.get("/reconciliation-tasks/{task_id}/progress")
async def get_reconciliation_progress(
    task_id: str,
    service: TrainingService = Depends(get_training_service)
):
    """Get current reconciliation progress"""
    progress = await service.get_reconciliation_progress(task_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Reconciliation task not found")
    return progress

@router.get("/tasks/{task_id}/progress")
async def get_training_progress(
    task_id: str,
    service: TrainingService = Depends(get_training_service)
):
    """Get current training progress"""
    progress = await service.get_training_progress(task_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Task not found")
    return progress

@router.delete("/tasks/{task_id}")
async def delete_training_task(
    task_id: str,
    db: Database = Depends(get_db)
):
    """Delete a specific training task"""
    try:
        # First check if task exists
        task = await db.get_training_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
            
        # Delete the task and its associated results
        await db.db.training_tasks.delete_one({"_id": ObjectId(task_id)})
        await db.db.training_results.delete_many({"task_id": ObjectId(task_id)})
        await db.db.reconciliation_results.delete_many({"task_id": ObjectId(task_id)})
        
        logger.info(f"Successfully deleted task {task_id} and its associated results")
        return {"message": f"Task {task_id} and its associated results deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting task {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting task: {str(e)}")

@router.delete("/tasks")
async def delete_all_training_tasks(
    db: Database = Depends(get_db)
):
    """Delete all training tasks and their results"""
    try:
        # Delete all tasks and their associated results
        await db.db.training_tasks.delete_many({})
        await db.db.training_results.delete_many({})
        await db.db.reconciliation_results.delete_many({})
        
        logger.info("Successfully deleted all tasks and their associated results")
        return {"message": "All tasks and their associated results deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting all tasks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting all tasks: {str(e)}") 
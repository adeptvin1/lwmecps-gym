from fastapi import APIRouter, HTTPException, Depends, status
from typing import List, Dict, Any, Optional
from ...models.models import TrainingTask, TrainingResult, ReconciliationResult, TrainingState, PyObjectId
from ...models.database import Database
from ...ml.training_service import TrainingService
import logging
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
    db: Database = Depends(get_db)
) -> TrainingService:
    return TrainingService(db)

def validate_object_id(task_id: str) -> PyObjectId:
    try:
        return PyObjectId(task_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid task ID: {str(e)}")

@router.post("/tasks", response_model=TrainingTask, status_code=status.HTTP_201_CREATED)
async def create_training_task(task: TrainingTask):
    return await db.create_training_task(task)

@router.get("/tasks", response_model=List[TrainingTask])
async def list_training_tasks(
    skip: int = 0,
    limit: int = 10,
    db: Database = Depends(get_db)
):
    """List all training tasks"""
    return await db.list_training_tasks(skip, limit)

@router.get("/tasks/{task_id}", response_model=TrainingTask)
async def get_training_task(task_id: PyObjectId):
    task = await db.get_training_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@router.post("/tasks/{task_id}/start")
async def start_training(task_id: PyObjectId):
    task = await db.get_training_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.state != TrainingState.PENDING:
        raise HTTPException(status_code=400, detail="Task is not in PENDING state")
    
    task.state = TrainingState.RUNNING
    updated_task = await db.update_training_task(task_id, task)
    if not updated_task:
        raise HTTPException(status_code=500, detail="Failed to update task")
    return {"message": "Training started"}

@router.post("/tasks/{task_id}/pause")
async def pause_training(task_id: PyObjectId):
    task = await db.get_training_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.state != TrainingState.RUNNING:
        raise HTTPException(status_code=400, detail="Task is not in RUNNING state")
    
    task.state = TrainingState.PAUSED
    updated_task = await db.update_training_task(task_id, task)
    if not updated_task:
        raise HTTPException(status_code=500, detail="Failed to update task")
    return {"message": "Training paused"}

@router.post("/tasks/{task_id}/resume")
async def resume_training(task_id: PyObjectId):
    task = await db.get_training_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.state != TrainingState.PAUSED:
        raise HTTPException(status_code=400, detail="Task is not in PAUSED state")
    
    task.state = TrainingState.RUNNING
    updated_task = await db.update_training_task(task_id, task)
    if not updated_task:
        raise HTTPException(status_code=500, detail="Failed to update task")
    return {"message": "Training resumed"}

@router.post("/tasks/{task_id}/stop")
async def stop_training(task_id: PyObjectId):
    task = await db.get_training_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.state not in [TrainingState.RUNNING, TrainingState.PAUSED]:
        raise HTTPException(status_code=400, detail="Task is not in RUNNING or PAUSED state")
    
    task.state = TrainingState.COMPLETED
    updated_task = await db.update_training_task(task_id, task)
    if not updated_task:
        raise HTTPException(status_code=500, detail="Failed to update task")
    return {"message": "Training stopped"}

@router.get("/tasks/{task_id}/results", response_model=List[TrainingResult])
async def get_training_results(task_id: PyObjectId):
    task = await db.get_training_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return await db.get_training_results(task_id)

@router.get("/tasks/{task_id}/reconciliation", response_model=List[ReconciliationResult])
async def get_reconciliation_results(task_id: PyObjectId):
    task = await db.get_training_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return await db.get_reconciliation_results(task_id)

@router.get("/tasks/{task_id}/progress")
async def get_training_progress(
    task_id: str,
    service: TrainingService = Depends(get_training_service)
):
    """Get current training progress"""
    validate_object_id(task_id)
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
        object_id = validate_object_id(task_id)
        # First check if task exists
        task = await db.get_training_task(object_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
            
        # Delete the task and its associated results
        await db.db.training_tasks.delete_one({"_id": object_id})
        await db.db.training_results.delete_many({"task_id": object_id})
        await db.db.reconciliation_results.delete_many({"task_id": object_id})
        
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
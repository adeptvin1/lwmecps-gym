import os
from typing import List, Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic_settings import BaseSettings
from bson import ObjectId
from datetime import datetime
from .models import TrainingTask, TrainingResult, ReconciliationResult

class DatabaseSettings(BaseSettings):
    """Database configuration settings"""
    mongodb_url: str = "mongodb://localhost:27017"
    database_name: str = "lwmecps_gym"
    
    class Config:
        env_prefix = "DB_"

class Database:
    def __init__(self):
        mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self.client = AsyncIOMotorClient(mongodb_uri)
        self.db = self.client.lwmecps_gym
        
    async def create_training_task(self, task: TrainingTask) -> TrainingTask:
        """Create a new training task"""
        task_dict = task.model_dump(by_alias=True, exclude={'id'})
        result = await self.db.training_tasks.insert_one(task_dict)
        task.id = str(result.inserted_id)
        return task
    
    async def get_training_task(self, task_id: str) -> Optional[TrainingTask]:
        """Get a training task by ID"""
        try:
            result = await self.db.training_tasks.find_one({"_id": ObjectId(task_id)})
            if result:
                return TrainingTask(**result)
            return None
        except Exception as e:
            print(f"Error getting training task: {e}")
            return None
    
    async def update_training_task(self, task_id: str, update_data: Dict[str, Any]) -> Optional[TrainingTask]:
        """Update a training task"""
        try:
            update_data["updated_at"] = datetime.utcnow()
            result = await self.db.training_tasks.find_one_and_update(
                {"_id": ObjectId(task_id)},
                {"$set": update_data},
                return_document=True
            )
            if result:
                return TrainingTask(**result)
            return None
        except Exception as e:
            print(f"Error updating training task: {e}")
            return None
    
    async def list_training_tasks(self, skip: int = 0, limit: int = 10) -> List[TrainingTask]:
        """List training tasks with pagination"""
        try:
            cursor = self.db.training_tasks.find().skip(skip).limit(limit)
            tasks = await cursor.to_list(length=limit)
            return [TrainingTask(**task) for task in tasks]
        except Exception as e:
            print(f"Error listing training tasks: {e}")
            return []
    
    async def save_training_result(self, result: TrainingResult) -> TrainingResult:
        """Save a training result"""
        try:
            result_dict = result.model_dump(by_alias=True, exclude={'id'})
            db_result = await self.db.training_results.insert_one(result_dict)
            result.id = str(db_result.inserted_id)
            return result
        except Exception as e:
            print(f"Error saving training result: {e}")
            raise
    
    async def get_training_results(self, task_id: str) -> List[TrainingResult]:
        """Get all training results for a task"""
        try:
            cursor = self.db.training_results.find({"task_id": ObjectId(task_id)})
            results = await cursor.to_list(length=None)
            return [TrainingResult(**result) for result in results]
        except Exception as e:
            print(f"Error getting training results: {e}")
            return []
    
    async def save_reconciliation_result(self, result: ReconciliationResult) -> ReconciliationResult:
        """Save a reconciliation result"""
        try:
            result_dict = result.model_dump(by_alias=True, exclude={'id'})
            db_result = await self.db.reconciliation_results.insert_one(result_dict)
            result.id = str(db_result.inserted_id)
            return result
        except Exception as e:
            print(f"Error saving reconciliation result: {e}")
            raise
    
    async def get_reconciliation_results(self, task_id: str) -> List[ReconciliationResult]:
        """Get all reconciliation results for a task"""
        try:
            cursor = self.db.reconciliation_results.find({"task_id": ObjectId(task_id)})
            results = await cursor.to_list(length=None)
            return [ReconciliationResult(**result) for result in results]
        except Exception as e:
            print(f"Error getting reconciliation results: {e}")
            return []
    
    async def close(self):
        """Close the database connection"""
        self.client.close() 
from typing import List, Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
import logging
from bson import ObjectId
from .models import TrainingTask, TrainingResult, ReconciliationResult, PyObjectId
from .config import settings

logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        self.client = None
        self.db = None

    async def connect_to_database(self):
        """Connect to MongoDB."""
        try:
            self.client = AsyncIOMotorClient(settings.mongodb_uri)
            self.db = self.client[settings.mongodb_db]
            # Test the connection
            await self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB.")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    async def close_database_connection(self):
        """Close the database connection."""
        if self.client:
            self.client.close()
            logger.info("Database connection closed.")

    async def close(self):
        """Alias for close_database_connection for compatibility."""
        await self.close_database_connection()

    async def create_training_task(self, task: TrainingTask) -> TrainingTask:
        try:
            task_dict = task.model_dump(by_alias=True, exclude={"task_id"})
            result = await self.db.training_tasks.insert_one(task_dict)
            task.task_id = str(result.inserted_id)
            return task
        except Exception as e:
            logger.error(f"Failed to create training task: {str(e)}")
            raise

    async def get_training_task(self, task_id: PyObjectId) -> Optional[TrainingTask]:
        try:
            task = await self.db.training_tasks.find_one({"_id": ObjectId(task_id)})
            if task:
                return TrainingTask(**task)
            return None
        except Exception as e:
            logger.error(f"Failed to get training task: {str(e)}")
            raise

    async def update_training_task(self, task_id: PyObjectId, task: TrainingTask) -> Optional[TrainingTask]:
        try:
            task_dict = task.model_dump(by_alias=True, exclude={"task_id"})
            result = await self.db.training_tasks.update_one(
                {"_id": ObjectId(task_id)},
                {"$set": task_dict}
            )
            if result.modified_count == 0:
                return None
            return await self.get_training_task(task_id)
        except Exception as e:
            logger.error(f"Failed to update training task: {str(e)}")
            raise

    async def list_training_tasks(self, skip: int = 0, limit: int = 10) -> List[TrainingTask]:
        try:
            cursor = self.db.training_tasks.find().skip(skip).limit(limit)
            tasks = await cursor.to_list(length=limit)
            return [TrainingTask(**task) for task in tasks]
        except Exception as e:
            logger.error(f"Failed to list training tasks: {str(e)}")
            raise

    async def save_training_result(self, result: TrainingResult) -> TrainingResult:
        try:
            result_dict = result.model_dump(by_alias=True)
            await self.db.training_results.insert_one(result_dict)
            return result
        except Exception as e:
            logger.error(f"Failed to save training result: {str(e)}")
            raise

    async def get_training_results(self, task_id: PyObjectId) -> List[TrainingResult]:
        try:
            results = []
            cursor = self.db.training_results.find({"task_id": ObjectId(task_id)})
            async for result in cursor:
                results.append(TrainingResult(**result))
            return results
        except Exception as e:
            logger.error(f"Failed to get training results: {str(e)}")
            raise

    async def save_reconciliation_result(self, result: ReconciliationResult) -> ReconciliationResult:
        try:
            result_dict = result.model_dump(by_alias=True)
            await self.db.reconciliation_results.insert_one(result_dict)
            return result
        except Exception as e:
            logger.error(f"Failed to save reconciliation result: {str(e)}")
            raise

    async def get_reconciliation_results(self, task_id: PyObjectId) -> List[ReconciliationResult]:
        try:
            results = []
            cursor = self.db.reconciliation_results.find({"task_id": ObjectId(task_id)})
            async for result in cursor:
                results.append(ReconciliationResult(**result))
            return results
        except Exception as e:
            logger.error(f"Failed to get reconciliation results: {str(e)}")
            raise

db = Database() 
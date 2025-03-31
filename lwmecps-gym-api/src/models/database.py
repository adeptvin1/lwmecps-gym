from typing import List, Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient
import logging
from bson.objectid import ObjectId
from .models import TrainingTask, TrainingResult, ReconciliationResult
from .config import settings

logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        self.client = None
        self.db = None

    async def connect(self):
        try:
            self.client = AsyncIOMotorClient(settings.mongodb_uri)
            self.db = self.client[settings.mongodb_db]
            logger.info("Connected to MongoDB")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise

    async def close(self):
        if self.client:
            self.client.close()
            logger.info("Closed MongoDB connection")

    async def create_training_task(self, task: TrainingTask) -> TrainingTask:
        try:
            result = await self.db.training_tasks.insert_one(task.model_dump(by_alias=True))
            task.id = str(result.inserted_id)
            return task
        except Exception as e:
            logger.error(f"Failed to create training task: {str(e)}")
            raise

    async def get_training_task(self, task_id: ObjectId) -> Optional[TrainingTask]:
        try:
            result = await self.db.training_tasks.find_one({"_id": task_id})
            return TrainingTask(**result) if result else None
        except Exception as e:
            logger.error(f"Failed to get training task: {str(e)}")
            raise

    async def update_training_task(self, task_id: ObjectId, update_data: Dict[str, Any]) -> Optional[TrainingTask]:
        try:
            result = await self.db.training_tasks.find_one_and_update(
                {"_id": task_id},
                {"$set": update_data},
                return_document=True
            )
            return TrainingTask(**result) if result else None
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

    async def get_training_results(self, task_id: ObjectId) -> List[TrainingResult]:
        try:
            cursor = self.db.training_results.find({"task_id": task_id})
            results = await cursor.to_list(length=None)
            return [TrainingResult(**result) for result in results]
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

    async def get_reconciliation_results(self, task_id: ObjectId) -> List[ReconciliationResult]:
        try:
            cursor = self.db.reconciliation_results.find({"task_id": task_id})
            results = await cursor.to_list(length=None)
            return [ReconciliationResult(**result) for result in results]
        except Exception as e:
            logger.error(f"Failed to get reconciliation results: {str(e)}")
            raise 
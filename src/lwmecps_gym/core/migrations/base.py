from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase

class Migration(ABC):
    """Base class for MongoDB migrations"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.migration_collection = db.migrations
    
    @property
    @abstractmethod
    def version(self) -> int:
        """Migration version number"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Migration description"""
        pass
    
    @abstractmethod
    async def up(self) -> None:
        """Apply migration"""
        pass
    
    @abstractmethod
    async def down(self) -> None:
        """Rollback migration"""
        pass
    
    async def is_applied(self) -> bool:
        """Check if migration is already applied"""
        result = await self.migration_collection.find_one({
            "version": self.version
        })
        return result is not None
    
    async def mark_as_applied(self) -> None:
        """Mark migration as applied"""
        await self.migration_collection.insert_one({
            "version": self.version,
            "description": self.description,
            "applied_at": datetime.utcnow()
        })
    
    async def mark_as_rolled_back(self) -> None:
        """Mark migration as rolled back"""
        await self.migration_collection.delete_one({
            "version": self.version
        }) 
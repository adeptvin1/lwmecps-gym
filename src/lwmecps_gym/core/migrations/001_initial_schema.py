from .base import Migration

class InitialSchemaMigration(Migration):
    """Initial database schema migration"""
    
    @property
    def version(self) -> int:
        return 1
    
    @property
    def description(self) -> str:
        return "Create initial collections and indexes"
    
    async def up(self) -> None:
        # Create collections
        collections = await self.db.list_collection_names()
        print(f"Existing collections before migration: {collections}")
        
        # Create collections
        await self.db.create_collection("training_tasks")
        await self.db.create_collection("training_results")
        await self.db.create_collection("reconciliation_results")
        
        # Verify collections were created
        collections = await self.db.list_collection_names()
        print(f"Collections after migration: {collections}")
        
        # Create indexes for training_tasks
        await self.db.training_tasks.create_index("state")
        await self.db.training_tasks.create_index("created_at")
        await self.db.training_tasks.create_index("model_type")
        
        # Create indexes for training_results
        await self.db.training_results.create_index("task_id")
        await self.db.training_results.create_index("timestamp")
        await self.db.training_results.create_index("episode")
        
        # Create indexes for reconciliation_results
        await self.db.reconciliation_results.create_index("task_id")
        await self.db.reconciliation_results.create_index("timestamp")
        await self.db.reconciliation_results.create_index("model_type")
    
    async def down(self) -> None:
        # Drop collections
        await self.db.training_tasks.drop()
        await self.db.training_results.drop()
        await self.db.reconciliation_results.drop() 
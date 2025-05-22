import importlib
import inspect
import os
from typing import List, Type
from motor.motor_asyncio import AsyncIOMotorDatabase
from .base import Migration

class MigrationManager:
    """Manager for MongoDB migrations"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.migrations: List[Type[Migration]] = []
    
    def load_migrations(self, migrations_dir: str) -> None:
        """Load all migration classes from the specified directory"""
        # Get all Python files in the migrations directory
        migration_files = [
            f[:-3] for f in os.listdir(migrations_dir)
            if f.endswith('.py') and not f.startswith('__') and f != 'base.py' and f != 'manager.py' and f != 'run_migrations.py'
        ]
        
        print(f"Found migration files: {migration_files}")
        
        # Use correct package name for migrations
        package = 'lwmecps_gym.core.migrations'
        
        # Import all migration modules
        for migration_file in migration_files:
            print(f"Loading migration from file: {migration_file}")
            module = importlib.import_module(f"{package}.{migration_file}")
            
            # Find all Migration subclasses in the module
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, Migration) and 
                    obj != Migration):
                    print(f"Found migration class: {name}")
                    self.migrations.append(obj)
        
        # Sort migrations by version (instantiate to get version)
        self.migrations.sort(key=lambda x: x(self.db).version)
        print(f"Loaded {len(self.migrations)} migrations")
    
    async def apply_migrations(self) -> None:
        """Apply all pending migrations"""
        for migration_class in self.migrations:
            migration = migration_class(self.db)
            
            if not await migration.is_applied():
                try:
                    await migration.up()
                    await migration.mark_as_applied()
                    print(f"Applied migration {migration.version}: {migration.description}")
                except Exception as e:
                    print(f"Error applying migration {migration.version}: {str(e)}")
                    raise
    
    async def rollback_migrations(self, target_version: int = None) -> None:
        """Rollback migrations to the specified version"""
        # Sort migrations in reverse order
        migrations = sorted(self.migrations, key=lambda x: x(self.db).version, reverse=True)
        
        for migration_class in migrations:
            migration = migration_class(self.db)
            
            if await migration.is_applied():
                if target_version is not None and migration.version <= target_version:
                    break
                    
                try:
                    await migration.down()
                    await migration.mark_as_rolled_back()
                    print(f"Rolled back migration {migration.version}: {migration.description}")
                except Exception as e:
                    print(f"Error rolling back migration {migration.version}: {str(e)}")
                    raise
    
    async def get_current_version(self) -> int:
        """Get the current database version"""
        result = await self.db.migrations.find_one(
            sort=[("version", -1)]
        )
        return result["version"] if result else 0
    
    async def get_pending_migrations(self) -> List[Type[Migration]]:
        """Get list of pending migrations"""
        current_version = await self.get_current_version()
        return [
            migration for migration in self.migrations
            if migration(self.db).version > current_version
        ] 
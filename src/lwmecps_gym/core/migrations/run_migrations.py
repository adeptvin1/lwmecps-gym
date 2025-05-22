import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from .manager import MigrationManager

async def run_migrations():
    # Get MongoDB connection string from environment variable or use default
    mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    database_name = os.getenv("MONGODB_DATABASE", "lwmecps_gym")
    
    # Connect to MongoDB
    client = AsyncIOMotorClient(mongodb_url)
    db = client[database_name]
    
    try:
        # Initialize migration manager
        manager = MigrationManager(db)
        
        # Load migrations from the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        manager.load_migrations(current_dir)
        
        # Get current version
        current_version = await manager.get_current_version()
        print(f"Current database version: {current_version}")
        
        # Get pending migrations
        pending_migrations = await manager.get_pending_migrations()
        if pending_migrations:
            print(f"Found {len(pending_migrations)} pending migrations:")
            for migration in pending_migrations:
                print(f"- Version {migration.version}: {migration.description}")
            
            # Apply migrations
            await manager.apply_migrations()
            print("All migrations applied successfully")
        else:
            print("No pending migrations found")
            
    finally:
        # Close MongoDB connection
        client.close()

if __name__ == "__main__":
    asyncio.run(run_migrations()) 
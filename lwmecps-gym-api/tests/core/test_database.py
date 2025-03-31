import pytest
from datetime import datetime
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from lwmecps_gym_api.core.database import Database
from lwmecps_gym_api.core.models import TrainingTask, TrainingResult, ReconciliationResult, TrainingState, ModelType

@pytest.fixture
async def db():
    """Create a test database connection"""
    db = Database()
    yield db
    await db.close()

@pytest.fixture
def training_task():
    """Create a test training task"""
    return TrainingTask(
        name="Test Task",
        description="Test Description",
        model_type=ModelType.DQN,
        parameters={"learning_rate": 0.001},
        total_episodes=100
    )

@pytest.fixture
def training_result(training_task):
    """Create a test training result"""
    return TrainingResult(
        task_id=training_task.id,
        episode=1,
        metrics={"reward": 100.0},
        wandb_run_id="test_run_123"
    )

@pytest.fixture
def reconciliation_result(training_task):
    """Create a test reconciliation result"""
    return ReconciliationResult(
        task_id=training_task.id,
        model_type=ModelType.DQN,
        wandb_run_id="test_run_123",
        metrics={"accuracy": 0.95},
        sample_size=100,
        model_weights_path="models/test_model.pt"
    )

@pytest.mark.asyncio
async def test_create_training_task(db, training_task):
    """Test creating a training task"""
    result = await db.create_training_task(training_task)
    assert result.id is not None
    assert result.name == training_task.name
    assert result.model_type == training_task.model_type

@pytest.mark.asyncio
async def test_get_training_task(db, training_task):
    """Test getting a training task"""
    # First create the task
    created_task = await db.create_training_task(training_task)
    
    # Then get it
    retrieved_task = await db.get_training_task(created_task.id)
    assert retrieved_task is not None
    assert retrieved_task.id == created_task.id
    assert retrieved_task.name == created_task.name

@pytest.mark.asyncio
async def test_update_training_task(db, training_task):
    """Test updating a training task"""
    # First create the task
    created_task = await db.create_training_task(training_task)
    
    # Update the task
    update_data = {
        "state": TrainingState.RUNNING,
        "current_episode": 10
    }
    updated_task = await db.update_training_task(created_task.id, update_data)
    assert updated_task is not None
    assert updated_task.state == TrainingState.RUNNING
    assert updated_task.current_episode == 10

@pytest.mark.asyncio
async def test_list_training_tasks(db, training_task):
    """Test listing training tasks"""
    # Create multiple tasks
    tasks = []
    for i in range(3):
        task = TrainingTask(
            name=f"Test Task {i}",
            model_type=ModelType.DQN,
            total_episodes=100
        )
        tasks.append(await db.create_training_task(task))
    
    # List tasks
    retrieved_tasks = await db.list_training_tasks(skip=0, limit=10)
    assert len(retrieved_tasks) >= 3
    assert all(isinstance(task, TrainingTask) for task in retrieved_tasks)

@pytest.mark.asyncio
async def test_save_training_result(db, training_task, training_result):
    """Test saving a training result"""
    # First create the task
    await db.create_training_task(training_task)
    
    # Save the result
    saved_result = await db.save_training_result(training_result)
    assert saved_result.id is not None
    assert saved_result.task_id == training_result.task_id
    assert saved_result.episode == training_result.episode

@pytest.mark.asyncio
async def test_get_training_results(db, training_task, training_result):
    """Test getting training results"""
    # First create the task and save a result
    await db.create_training_task(training_task)
    await db.save_training_result(training_result)
    
    # Get results
    results = await db.get_training_results(training_task.id)
    assert len(results) > 0
    assert all(isinstance(result, TrainingResult) for result in results)

@pytest.mark.asyncio
async def test_save_reconciliation_result(db, training_task, reconciliation_result):
    """Test saving a reconciliation result"""
    # First create the task
    await db.create_training_task(training_task)
    
    # Save the result
    saved_result = await db.save_reconciliation_result(reconciliation_result)
    assert saved_result.id is not None
    assert saved_result.task_id == reconciliation_result.task_id
    assert saved_result.model_type == reconciliation_result.model_type

@pytest.mark.asyncio
async def test_get_reconciliation_results(db, training_task, reconciliation_result):
    """Test getting reconciliation results"""
    # First create the task and save a result
    await db.create_training_task(training_task)
    await db.save_reconciliation_result(reconciliation_result)
    
    # Get results
    results = await db.get_reconciliation_results(training_task.id)
    assert len(results) > 0
    assert all(isinstance(result, ReconciliationResult) for result in results) 
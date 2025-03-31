import pytest
from datetime import datetime
from bson import ObjectId
from lwmecps_gym_api.core.models import TrainingTask, TrainingResult, ReconciliationResult, TrainingState, ModelType
from lwmecps_gym_api.core.database import Database

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
        id=str(ObjectId()),
        name="Test Task",
        description="Test Description",
        model_type=ModelType.DQN,
        parameters={"learning_rate": 0.001},
        total_episodes=100,
        current_episode=0,
        state=TrainingState.CREATED,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

@pytest.fixture
def training_result(training_task):
    """Create a test training result"""
    return TrainingResult(
        id=str(ObjectId()),
        task_id=training_task.id,
        episode=1,
        metrics={"reward": 100.0},
        wandb_run_id="test_run_123",
        created_at=datetime.utcnow()
    )

@pytest.fixture
def reconciliation_result(training_task):
    """Create a test reconciliation result"""
    return ReconciliationResult(
        id=str(ObjectId()),
        task_id=training_task.id,
        model_type=ModelType.DQN,
        wandb_run_id="test_run_123",
        metrics={"accuracy": 0.95},
        sample_size=100,
        model_weights_path="models/test_model.pt",
        created_at=datetime.utcnow()
    ) 
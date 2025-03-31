import pytest
from datetime import datetime
from bson import ObjectId
from lwmecps_gym_api.core.models import (
    TrainingTask,
    TrainingResult,
    ReconciliationResult,
    TrainingState,
    ModelType,
    TrainingTaskCreate,
    TrainingTaskUpdate,
    TrainingResultCreate,
    ReconciliationResultCreate
)

def test_training_task_creation():
    """Test creating a TrainingTask"""
    task = TrainingTask(
        name="Test Task",
        description="Test Description",
        model_type=ModelType.DQN,
        parameters={"learning_rate": 0.001},
        total_episodes=100
    )
    
    assert task.id is not None
    assert task.name == "Test Task"
    assert task.description == "Test Description"
    assert task.model_type == ModelType.DQN
    assert task.parameters == {"learning_rate": 0.001}
    assert task.total_episodes == 100
    assert task.current_episode == 0
    assert task.state == TrainingState.CREATED
    assert task.created_at is not None
    assert task.updated_at is not None

def test_training_task_from_dict():
    """Test creating a TrainingTask from dictionary"""
    task_dict = {
        "id": str(ObjectId()),
        "name": "Test Task",
        "description": "Test Description",
        "model_type": ModelType.DQN,
        "parameters": {"learning_rate": 0.001},
        "total_episodes": 100,
        "current_episode": 0,
        "state": TrainingState.CREATED,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    task = TrainingTask(**task_dict)
    assert task.id == task_dict["id"]
    assert task.name == task_dict["name"]
    assert task.description == task_dict["description"]
    assert task.model_type == task_dict["model_type"]
    assert task.parameters == task_dict["parameters"]
    assert task.total_episodes == task_dict["total_episodes"]
    assert task.current_episode == task_dict["current_episode"]
    assert task.state == task_dict["state"]
    assert task.created_at == task_dict["created_at"]
    assert task.updated_at == task_dict["updated_at"]

def test_training_result_creation():
    """Test creating a TrainingResult"""
    result = TrainingResult(
        task_id=str(ObjectId()),
        episode=1,
        metrics={"reward": 100.0},
        wandb_run_id="test_run_123"
    )
    
    assert result.id is not None
    assert result.task_id is not None
    assert result.episode == 1
    assert result.metrics == {"reward": 100.0}
    assert result.wandb_run_id == "test_run_123"
    assert result.created_at is not None

def test_reconciliation_result_creation():
    """Test creating a ReconciliationResult"""
    result = ReconciliationResult(
        task_id=str(ObjectId()),
        model_type=ModelType.DQN,
        wandb_run_id="test_run_123",
        metrics={"accuracy": 0.95},
        sample_size=100,
        model_weights_path="models/test_model.pt"
    )
    
    assert result.id is not None
    assert result.task_id is not None
    assert result.model_type == ModelType.DQN
    assert result.wandb_run_id == "test_run_123"
    assert result.metrics == {"accuracy": 0.95}
    assert result.sample_size == 100
    assert result.model_weights_path == "models/test_model.pt"
    assert result.created_at is not None

def test_training_task_create():
    """Test TrainingTaskCreate model"""
    task_create = TrainingTaskCreate(
        name="Test Task",
        description="Test Description",
        model_type=ModelType.DQN,
        parameters={"learning_rate": 0.001},
        total_episodes=100
    )
    
    assert task_create.name == "Test Task"
    assert task_create.description == "Test Description"
    assert task_create.model_type == ModelType.DQN
    assert task_create.parameters == {"learning_rate": 0.001}
    assert task_create.total_episodes == 100

def test_training_task_update():
    """Test TrainingTaskUpdate model"""
    task_update = TrainingTaskUpdate(
        name="Updated Task",
        state=TrainingState.RUNNING
    )
    
    assert task_update.name == "Updated Task"
    assert task_update.state == TrainingState.RUNNING
    assert task_update.description is None
    assert task_update.parameters is None
    assert task_update.total_episodes is None
    assert task_update.current_episode is None

def test_training_result_create():
    """Test TrainingResultCreate model"""
    result_create = TrainingResultCreate(
        task_id=str(ObjectId()),
        episode=1,
        metrics={"reward": 100.0},
        wandb_run_id="test_run_123"
    )
    
    assert result_create.task_id is not None
    assert result_create.episode == 1
    assert result_create.metrics == {"reward": 100.0}
    assert result_create.wandb_run_id == "test_run_123"

def test_reconciliation_result_create():
    """Test ReconciliationResultCreate model"""
    result_create = ReconciliationResultCreate(
        task_id=str(ObjectId()),
        model_type=ModelType.DQN,
        wandb_run_id="test_run_123",
        metrics={"accuracy": 0.95},
        sample_size=100,
        model_weights_path="models/test_model.pt"
    )
    
    assert result_create.task_id is not None
    assert result_create.model_type == ModelType.DQN
    assert result_create.wandb_run_id == "test_run_123"
    assert result_create.metrics == {"accuracy": 0.95}
    assert result_create.sample_size == 100
    assert result_create.model_weights_path == "models/test_model.pt" 
import pytest
from fastapi.testclient import TestClient
from datetime import datetime
from bson import ObjectId
from lwmecps_gym_api.core.models import TrainingTask, TrainingState, ModelType
from lwmecps_gym_api.main import app

client = TestClient(app)

@pytest.fixture
def mock_training_task():
    return {
        "name": "Test Task",
        "description": "Test Description",
        "model_type": ModelType.DQN,
        "parameters": {
            "learning_rate": 0.001,
            "batch_size": 32
        },
        "total_episodes": 100
    }

@pytest.fixture
def mock_training_result():
    return {
        "task_id": str(ObjectId()),
        "episode": 1,
        "metrics": {
            "reward": 100.0,
            "loss": 0.5
        },
        "model_weights_path": "models/test_model.pt",
        "wandb_run_id": "test_run_123"
    }

def test_create_training_task(mock_training_task):
    """Test creating a new training task"""
    response = client.post("/api/training/tasks", json=mock_training_task)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == mock_training_task["name"]
    assert data["model_type"] == mock_training_task["model_type"]
    assert data["state"] == TrainingState.PENDING

def test_list_training_tasks():
    """Test listing training tasks"""
    response = client.get("/api/training/tasks")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_training_task():
    """Test getting a specific training task"""
    # First create a task
    task_data = {
        "name": "Test Task",
        "model_type": ModelType.DQN,
        "total_episodes": 100
    }
    create_response = client.post("/api/training/tasks", json=task_data)
    task_id = create_response.json()["id"]
    
    # Then get it
    response = client.get(f"/api/training/tasks/{task_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == task_id

def test_start_training():
    """Test starting a training task"""
    # First create a task
    task_data = {
        "name": "Test Task",
        "model_type": ModelType.DQN,
        "total_episodes": 100
    }
    create_response = client.post("/api/training/tasks", json=task_data)
    task_id = create_response.json()["id"]
    
    # Then start it
    response = client.post(f"/api/training/tasks/{task_id}/start")
    assert response.status_code == 200
    data = response.json()
    assert data["state"] == TrainingState.RUNNING

def test_pause_training():
    """Test pausing a training task"""
    # First create and start a task
    task_data = {
        "name": "Test Task",
        "model_type": ModelType.DQN,
        "total_episodes": 100
    }
    create_response = client.post("/api/training/tasks", json=task_data)
    task_id = create_response.json()["id"]
    client.post(f"/api/training/tasks/{task_id}/start")
    
    # Then pause it
    response = client.post(f"/api/training/tasks/{task_id}/pause")
    assert response.status_code == 200
    data = response.json()
    assert data["state"] == TrainingState.PAUSED

def test_resume_training():
    """Test resuming a paused training task"""
    # First create, start, and pause a task
    task_data = {
        "name": "Test Task",
        "model_type": ModelType.DQN,
        "total_episodes": 100
    }
    create_response = client.post("/api/training/tasks", json=task_data)
    task_id = create_response.json()["id"]
    client.post(f"/api/training/tasks/{task_id}/start")
    client.post(f"/api/training/tasks/{task_id}/pause")
    
    # Then resume it
    response = client.post(f"/api/training/tasks/{task_id}/resume")
    assert response.status_code == 200
    data = response.json()
    assert data["state"] == TrainingState.RUNNING

def test_stop_training():
    """Test stopping a training task"""
    # First create and start a task
    task_data = {
        "name": "Test Task",
        "model_type": ModelType.DQN,
        "total_episodes": 100
    }
    create_response = client.post("/api/training/tasks", json=task_data)
    task_id = create_response.json()["id"]
    client.post(f"/api/training/tasks/{task_id}/start")
    
    # Then stop it
    response = client.post(f"/api/training/tasks/{task_id}/stop")
    assert response.status_code == 200
    data = response.json()
    assert data["state"] == TrainingState.FAILED

def test_get_training_results():
    """Test getting training results for a task"""
    # First create a task
    task_data = {
        "name": "Test Task",
        "model_type": ModelType.DQN,
        "total_episodes": 100
    }
    create_response = client.post("/api/training/tasks", json=task_data)
    task_id = create_response.json()["id"]
    
    # Then get its results
    response = client.get(f"/api/training/tasks/{task_id}/results")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_delete_training_task():
    """Test deleting a training task"""
    # First create a task
    task_data = {
        "name": "Test Task",
        "model_type": ModelType.DQN,
        "total_episodes": 100
    }
    create_response = client.post("/api/training/tasks", json=task_data)
    task_id = create_response.json()["id"]
    
    # Then delete it
    response = client.delete(f"/api/training/tasks/{task_id}")
    assert response.status_code == 200
    
    # Verify it's deleted
    get_response = client.get(f"/api/training/tasks/{task_id}")
    assert get_response.status_code == 404 
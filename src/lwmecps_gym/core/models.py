from datetime import datetime
from typing import Optional, List, Dict, Any, Annotated
from pydantic import BaseModel, Field, BeforeValidator
from enum import Enum
from bson import ObjectId

def validate_object_id(v: Any) -> str:
    if isinstance(v, ObjectId):
        return str(v)
    if isinstance(v, str):
        if ObjectId.is_valid(v):
            return v
    raise ValueError("Invalid objectid")

PyObjectId = Annotated[str, BeforeValidator(validate_object_id)]

class ModelType(str, Enum):
    DQN = "dqn"
    Q_LEARNING = "q_learning"
    PPO = "ppo"
    TD3 = "td3"
    SAC = "sac"
    # Add more model types as needed

class TrainingState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class TrainingTask(BaseModel):
    """MongoDB model for training tasks"""
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    name: str
    description: Optional[str] = None
    model_type: ModelType
    parameters: Dict[str, Any] = Field(default_factory=dict)
    env_config: Dict[str, Any] = Field(default_factory=dict)
    model_config: Dict[str, Any] = Field(default_factory=dict)
    state: TrainingState = TrainingState.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    wandb_run_id: Optional[str] = None
    current_episode: int = 0
    total_episodes: int
    progress: float = 0.0
    metrics: Dict[str, List[float]] = Field(default_factory=dict)
    error_message: Optional[str] = None
    model_path: Optional[str] = None
    group_id: str
    namespace: str = "default"
    deployment_name: str = "mec-test-app"
    max_pods: int = 10000
    base_url: str = "http://localhost:8001"
    stabilization_time: int = 10

    model_config = {
        "allow_population_by_field_name": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            ObjectId: str
        },
        "arbitrary_types_allowed": True
    }

class TrainingResult(BaseModel):
    """MongoDB model for training results"""
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    task_id: PyObjectId
    episode: int
    metrics: Dict[str, float]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_weights_path: Optional[str] = None
    wandb_run_id: str

    model_config = {
        "allow_population_by_field_name": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            ObjectId: str
        },
        "arbitrary_types_allowed": True
    }

class ReconciliationResult(BaseModel):
    """MongoDB model for model reconciliation results"""
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    task_id: PyObjectId
    model_type: ModelType
    wandb_run_id: str
    metrics: Dict[str, float]
    sample_size: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_weights_path: str

    model_config = {
        "allow_population_by_field_name": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            ObjectId: str
        },
        "arbitrary_types_allowed": True
    } 
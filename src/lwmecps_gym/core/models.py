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
    model_params: Dict[str, Any] = Field(default_factory=dict)
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
    namespace: str = "lwmecps-testapp"
    max_pods: int = 50
    base_url: str = "http://34.51.217.76:8001"
    stabilization_time: int = 10

    model_config = {
        "allow_population_by_field_name": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            ObjectId: str
        },
        "arbitrary_types_allowed": True
    }

class ReconciliationTask(BaseModel):
    """MongoDB model for reconciliation tasks"""
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    name: str
    description: Optional[str] = None
    training_task_id: PyObjectId = Field(description="ID исходной задачи обучения")
    model_type: ModelType = Field(description="Тип модели (ppo, sac, td3, dqn)")
    state: TrainingState = TrainingState.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    wandb_run_id: Optional[str] = None
    current_step: int = 0
    total_steps: int = Field(description="Общее количество шагов для выполнения", example=100)
    progress: float = 0.0
    metrics: Dict[str, List[float]] = Field(default_factory=dict)
    error_message: Optional[str] = None
    model_path: Optional[str] = None
    group_id: str = Field(description="ID группы экспериментов для reconciliation")
    namespace: str = "lwmecps-testapp"
    max_pods: int = 50
    base_url: str = "http://34.51.217.76:8001"
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
    task_id: PyObjectId = Field(description="ID исходной задачи обучения")
    model_type: ModelType = Field(description="Тип модели (ppo, sac, td3, dqn)")
    wandb_run_id: str = Field(description="ID запуска в Weights & Biases для reconciliation")
    metrics: Dict[str, float] = Field(
        description="Метрики производительности модели", 
        example={
            "avg_reward": 85.5,
            "avg_latency": 0.12,
            "avg_throughput": 150.0,
            "success_rate": 0.92,
            "latency_std": 0.02,
            "reward_std": 12.3,
            "adaptation_score": 0.85
        }
    )
    sample_size: int = Field(description="Количество выполненных шагов", example=100)
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, 
        description="Время завершения reconciliation"
    )
    model_weights_path: str = Field(
        description="Путь к файлу весов модели",
        example="./models/model_ppo_64f1b2a3c9d4e5f6a7b8c9d0.pth"
    )

    model_config = {
        "allow_population_by_field_name": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            ObjectId: str
        },
        "arbitrary_types_allowed": True
    } 
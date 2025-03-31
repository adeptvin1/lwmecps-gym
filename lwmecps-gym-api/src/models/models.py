from datetime import datetime
from typing import Optional, List, Dict, Any, Annotated
from pydantic import BaseModel, Field, BeforeValidator
from enum import Enum
from bson import ObjectId

def validate_object_id(v: Any) -> ObjectId:
    if isinstance(v, ObjectId):
        return v
    if isinstance(v, str):
        return ObjectId(v)
    raise ValueError("Invalid ObjectId")

PyObjectId = Annotated[ObjectId, BeforeValidator(validate_object_id)]

class PyObjectId(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not isinstance(v, (str, ObjectId)):
            raise ValueError('Invalid ObjectId')
        if isinstance(v, str):
            try:
                ObjectId(v)
            except Exception:
                raise ValueError('Invalid ObjectId')
        return str(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema):
        field_schema.update(type='string', format='objectid')
        return field_schema

class TrainingState(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class ModelType(str, Enum):
    DQN = "dqn"
    PPO = "ppo"
    A2C = "a2c"
    SAC = "sac"

class TrainingTask(BaseModel):
    """MongoDB model for training tasks"""
    task_id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    state: TrainingState = TrainingState.PENDING
    model_name: str
    model_version: str
    dataset_name: str
    dataset_version: str
    hyperparameters: Dict[str, float]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    class Config:
        json_encoders = {ObjectId: str}
        populate_by_name = True

class TrainingResult(BaseModel):
    """MongoDB model for training results"""
    task_id: PyObjectId = Field(alias="_id")
    metrics: Dict[str, float]
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {ObjectId: str}
        populate_by_name = True

class ReconciliationResult(BaseModel):
    """MongoDB model for model reconciliation results"""
    task_id: PyObjectId = Field(alias="_id")
    metrics: Dict[str, float]
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {ObjectId: str}
        populate_by_name = True 
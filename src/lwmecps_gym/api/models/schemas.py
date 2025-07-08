from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class GymResponse(BaseModel):
    """Base response model for gym endpoints"""
    status: str
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

class GymRequest(BaseModel):
    """Base request model for gym endpoints"""
    parameters: Optional[Dict[str, Any]] = None

class ReconciliationRequest(BaseModel):
    """Request model for reconciliation endpoint"""
    sample_size: int = Field(
        ..., 
        description="Количество шагов для выполнения reconciliation", 
        example=100,
        ge=1
    )
    group_id: Optional[str] = Field(
        None, 
        description="ID группы экспериментов для reconciliation. Если не указан, используется group_id из задачи обучения",
        example="reconciliation-group-1"
    ) 
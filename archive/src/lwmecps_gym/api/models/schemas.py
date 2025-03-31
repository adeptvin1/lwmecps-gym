from pydantic import BaseModel
from typing import Dict, Any, Optional

class GymResponse(BaseModel):
    """Base response model for gym endpoints"""
    status: str
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

class GymRequest(BaseModel):
    """Base request model for gym endpoints"""
    parameters: Optional[Dict[str, Any]] = None 
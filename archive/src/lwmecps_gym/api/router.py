from fastapi import APIRouter
from .endpoints import training

router = APIRouter()

# Register all endpoint routers
router.include_router(training.router, prefix="/training", tags=["training"]) 
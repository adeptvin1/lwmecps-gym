from fastapi import APIRouter
from .endpoints import training, meta_learning

router = APIRouter()

# Register all endpoint routers
router.include_router(training.router, prefix="/training", tags=["training"])
router.include_router(meta_learning.router, prefix="/meta-learning", tags=["meta-learning"]) 
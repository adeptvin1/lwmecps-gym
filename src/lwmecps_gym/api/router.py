from fastapi import APIRouter
from .endpoints import gym

router = APIRouter()

# Register all endpoint routers
router.include_router(gym.router, prefix="/gym", tags=["gym"]) 
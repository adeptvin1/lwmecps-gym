from fastapi import APIRouter


router = APIRouter()


@router.get("/executor")
async def executor(model: str):
    return None

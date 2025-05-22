from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import uvicorn
from prometheus_client import make_asgi_app
# from typing import List
import logging

from lwmecps_gym.api.router import router as api_router
from lwmecps_gym.core.config import settings
from lwmecps_gym.core.database import Database

app = FastAPI(
    title="LWMECPS GYM API",
    description="API для обучения моделей машинного обучения",
    version="0.0.1"
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Регистрация роутеров
app.include_router(api_router, prefix="/api")

# События приложения
@app.on_event("startup")
async def startup_db_client():
    try:
        db = Database()
        await db.initialize()
        logging.info("Database initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize database: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_db_client():
    try:
        db = Database()
        await db.close()
        logging.info("Database connection closed")
    except Exception as e:
        logging.error(f"Error closing database connection: {e}")

# Обработка ошибок
@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)},
    )

# Корневой эндпоинт
@app.get("/")
async def root():
    return {
        "message": "Welcome to LWMECPS GYM API",
        "docs": "/docs"
    }

# Add health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )

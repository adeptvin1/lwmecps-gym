from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import uvicorn
from prometheus_client import make_asgi_app

from .api.routes import api_router
from .models.config import settings
from .models.database import db

app = FastAPI(
    title="LWME CPS Gym API",
    description="API for LWME CPS Gym training and evaluation",
    version="1.0.0"
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
async def startup_event():
    """Initialize database connection on startup."""
    await db.connect_to_database()

@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection on shutdown."""
    await db.close_database_connection()

# Обработка ошибок
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )

# Корневой эндпоинт
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to LWME CPS Gym API"}

# Add health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    ) 
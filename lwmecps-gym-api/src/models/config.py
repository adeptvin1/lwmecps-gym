from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # API Server settings
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    debug: bool = os.getenv("API_DEBUG", "False").lower() == "true"

    # MongoDB settings
    mongodb_uri: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    mongodb_db: str = os.getenv("MONGODB_DB", "lwmecps_gym")

    # Weights & Biases settings
    wandb_project: str = os.getenv("WANDB_PROJECT", "lwmecps-gym")
    wandb_entity: Optional[str] = os.getenv("WANDB_ENTITY")
    wandb_api_key: Optional[str] = os.getenv("WANDB_API_KEY")
    wandb_mode: str = os.getenv("WANDB_MODE", "online")
    wandb_log_dir: str = os.getenv("WANDB_LOG_DIR", "wandb_logs")

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings() 
from typing import Optional
from pydantic_settings import BaseSettings
import wandb
from pathlib import Path
import torch
import os

class WandbConfig(BaseSettings):
    """Weights & Biases configuration settings"""
    project_name: str = "lwmecps-gym"
    entity: Optional[str] = None
    api_key: Optional[str] = None
    mode: str = "online"  # or "offline"
    log_dir: str = "wandb_logs"
    tags: list[str] = []
    
    class Config:
        env_prefix = "WANDB_"

def init_wandb(config: WandbConfig) -> None:
    """Initialize Weights & Biases with the given configuration"""
    # Create log directory if it doesn't exist
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project=config.project_name,
        entity=config.entity,
        mode=config.mode,
        dir=str(log_dir),
        tags=config.tags,
        config=config.model_dump(),
    )

def log_metrics(metrics: dict, step: Optional[int] = None) -> None:
    """Log metrics to Weights & Biases"""
    wandb.log(metrics, step=step)

def log_model(model: any, name: str) -> None:
    """Log model to Weights & Biases"""
    wandb.watch(model, log="all", log_freq=10)

def finish_wandb() -> None:
    """Finish the current wandb run"""
    wandb.finish()

def save_model(model: any, name: str, model_type: str = "model") -> None:
    """Save model to Weights & Biases as an artifact"""
    artifact = wandb.Artifact(name=name, type=model_type)
    # Save model to a temporary file
    model_path = f"{name}.pt"
    torch.save(model.state_dict(), model_path)
    # Add the model file to the artifact
    artifact.add_file(model_path)
    # Log the artifact
    wandb.log_artifact(artifact)
    # Clean up the temporary file
    os.remove(model_path) 
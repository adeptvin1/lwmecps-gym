from typing import Dict, Any, Optional
import wandb
import logging
from .config import settings

logger = logging.getLogger(__name__)

def init_wandb() -> None:
    """Initialize Weights & Biases"""
    try:
        wandb.init(
            project=settings.wandb_project,
            entity=settings.wandb_entity,
            mode=settings.wandb_mode,
            dir=settings.wandb_log_dir
        )
        logger.info("Initialized Weights & Biases")
    except Exception as e:
        logger.error(f"Failed to initialize Weights & Biases: {str(e)}")
        raise

def log_metrics(metrics: Dict[str, Any]) -> None:
    """Log metrics to Weights & Biases"""
    try:
        wandb.log(metrics)
    except Exception as e:
        logger.error(f"Failed to log metrics to Weights & Biases: {str(e)}")
        raise

def log_model(model: Any, name: str) -> None:
    """Log model to Weights & Biases"""
    try:
        wandb.save(model, name)
    except Exception as e:
        logger.error(f"Failed to log model to Weights & Biases: {str(e)}")
        raise

def finish_wandb() -> None:
    """Finish Weights & Biases run"""
    try:
        wandb.finish()
        logger.info("Finished Weights & Biases run")
    except Exception as e:
        logger.error(f"Failed to finish Weights & Biases run: {str(e)}")
        raise 
import time
import os
import logging
import requests
from typing import Dict, List, Union


def logging_config():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def start_experiment_group(group_id: str, base_url: str = "http://localhost:8001") -> None:
    """
    Start an existing experiment group.
    
    Args:
        group_id (str): ID of the group to start
        base_url (str): Base URL of the test application API
    """
    logger = logging_config()
    try:
        response = requests.post(
            f"{base_url}/api/manage_group",
            params={
                "group_id": group_id,
                "state": "running"
            }
        )
        response.raise_for_status()
        logger.info(f"Started experiment group {group_id}")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to start experiment group: {str(e)}")
        raise


def get_metrics(group_id: str, base_url: str = "http://localhost:8001") -> Dict[str, Dict[str, Union[float, int]]]:
    """
    Get metrics for the experiment group.
    
    Args:
        group_id (str): ID of the group
        base_url (str): Base URL of the test application API
        
    Returns:
        Dict[str, Dict[str, Union[float, int]]]: Metrics for each node
    """
    logger = logging_config()
    try:
        # Get group stats
        response = requests.get(
            f"{base_url}/api/group_stats",
            params={"group_id": group_id}
        )
        response.raise_for_status()
        stats = response.json()
        
        # Process metrics for each experiment
        metrics = {}
        for exp_id, exp_stats in stats.get("experiments_stats", {}).items():
            metrics[exp_id] = {
                "avg_latency": exp_stats.get("average_latency", 0.0),
                "concurrent_users": exp_stats.get("current_users", 0)
            }
        
        logger.info(f"Retrieved metrics for group {group_id}")
        return metrics
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get metrics: {str(e)}")
        raise 
import time
import os
import logging
import requests
from typing import Dict, List, Union, Optional, Any
from requests.exceptions import RequestException, Timeout
from tenacity import retry, stop_after_attempt, wait_exponential


def logging_config():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def validate_metrics(metrics: Dict[str, Dict[str, Any]]) -> bool:
    """
    Validate metrics data structure and values.
    
    Args:
        metrics (Dict[str, Dict[str, Any]]): Metrics to validate
        
    Returns:
        bool: True if metrics are valid, False otherwise
    """
    if not metrics:
        return False
        
    for exp_id, exp_metrics in metrics.items():
        if not isinstance(exp_metrics, dict):
            return False
        if 'avg_latency' not in exp_metrics or 'concurrent_users' not in exp_metrics:
            return False
        if not isinstance(exp_metrics['avg_latency'], (int, float)) or not isinstance(exp_metrics['concurrent_users'], (int, float)):
            return False
        if exp_metrics['avg_latency'] < 0 or exp_metrics['concurrent_users'] < 0:
            return False
    return True


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def start_experiment_group(group_id: str, base_url: str = "http://localhost:8001", timeout: int = 30) -> None:
    """
    Start an existing experiment group.
    
    Args:
        group_id (str): ID of the group to start
        base_url (str): Base URL of the test application API
        timeout (int): Request timeout in seconds
        
    Raises:
        RequestException: If the request fails
        ValueError: If group_id is invalid
    """
    logger = logging_config()
    
    if not group_id or not isinstance(group_id, str):
        raise ValueError("Invalid group_id")
        
    try:
        response = requests.post(
            f"{base_url}/api/manage_group",
            params={
                "group_id": group_id,
                "state": "start"
            },
            timeout=timeout
        )
        response.raise_for_status()
        
        # Validate response
        if response.status_code != 200:
            raise RequestException(f"Unexpected status code: {response.status_code}")
            
        logger.info(f"Started experiment group {group_id}")
        
    except Timeout:
        logger.error(f"Timeout while starting experiment group {group_id}")
        raise
    except RequestException as e:
        logger.error(f"Failed to start experiment group: {str(e)}")
        raise


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_metrics(group_id: str, base_url: str = "http://localhost:8001", timeout: int = 30) -> Dict[str, Dict[str, Any]]:
    """
    Get metrics for the experiment group.
    
    Args:
        group_id (str): ID of the group
        base_url (str): Base URL of the test application API
        timeout (int): Request timeout in seconds
        
    Returns:
        Dict[str, Dict[str, Any]]: Metrics for the group
        
    Raises:
        RequestException: If the request fails
        ValueError: If group_id is invalid or metrics are invalid
    """
    logger = logging_config()
    
    if not group_id or not isinstance(group_id, str):
        raise ValueError("Invalid group_id")
        
    try:
        # Get group stats
        response = requests.get(
            f"{base_url}/api/group_stats",
            params={"group_id": group_id},
            timeout=timeout
        )
        response.raise_for_status()
        
        # Validate response
        if response.status_code != 200:
            raise RequestException(f"Unexpected status code: {response.status_code}")
            
        stats = response.json()
        if not isinstance(stats, dict):
            raise ValueError("Invalid response format")
        
        # Process metrics
        metrics = {
            "group": {
                "avg_latency": stats.get("average_latency", 0.0),
                "concurrent_users": stats.get("total_requests", 0),
                "state": stats.get("state", "UNKNOWN")
            }
        }
        
        # Add experiment-specific metrics if available
        for exp_id, exp_stats in stats.get("experiments_stats", {}).items():
            if not isinstance(exp_stats, dict):
                logger.warning(f"Invalid experiment stats format for {exp_id}")
                continue
                
            metrics[exp_id] = {
                "avg_latency": exp_stats.get("average_latency", 0.0),
                "concurrent_users": exp_stats.get("current_users", 0)
            }
        
        # Validate metrics
        if not validate_metrics(metrics):
            raise ValueError("Invalid metrics format")
            
        logger.info(f"Retrieved metrics for group {group_id}")
        return metrics
        
    except Timeout:
        logger.error(f"Timeout while getting metrics for group {group_id}")
        raise
    except RequestException as e:
        logger.error(f"Failed to get metrics: {str(e)}")
        raise 
import time
import os
import logging
import requests
from typing import Dict, List, Union, Optional
from requests.exceptions import RequestException, Timeout, HTTPError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type


def logging_config():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def validate_metrics(metrics: Dict[str, Dict[str, Union[float, int]]]) -> bool:
    """
    Validate metrics data structure and values.
    
    Args:
        metrics (Dict[str, Dict[str, Union[float, int]]]): Metrics to validate
        
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


class GroupAlreadyCompletedError(Exception):
    """Exception raised when trying to start an already completed group."""
    pass


class GroupAlreadyRunningError(Exception):
    """Exception raised when trying to start an already running group."""
    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_not_exception_type((GroupAlreadyCompletedError, GroupAlreadyRunningError))
)
def start_experiment_group(group_id: str, base_url: str = "http://localhost:8001", timeout: int = 30) -> bool:
    """
    Start an existing experiment group.
    
    Args:
        group_id (str): ID of the group to start
        base_url (str): Base URL of the test application API
        timeout (int): Request timeout in seconds
        
    Returns:
        bool: True if group was started or is already running, False if group is already completed
        
    Raises:
        GroupAlreadyCompletedError: If the group is already completed
        GroupAlreadyRunningError: If the group is already running (non-fatal, returns True)
        RequestException: If the request fails for other reasons
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
        
        # Обрабатываем успешный ответ (200)
        if response.status_code == 200:
            response_data = response.json()
            message = response_data.get("message", "")
            
            # Проверяем, не является ли это сообщением о том, что группа уже запущена
            if "already running" in message.lower():
                logger.info(f"Group {group_id} is already running. Continuing with existing group.")
                return True
            
            logger.info(f"Started experiment group {group_id}")
            return True
        
        # Обрабатываем ошибку 400 (Bad Request)
        if response.status_code == 400:
            try:
                error_detail = response.json().get("detail", "")
                
                # Группа уже завершена - это нормальная ситуация, не ошибка
                if "already completed" in error_detail.lower():
                    logger.warning(f"Group {group_id} is already completed. This is expected if group finished.")
                    raise GroupAlreadyCompletedError(f"Group {group_id} is already completed")
                
                # Группа уже запущена - это тоже нормальная ситуация
                if "already running" in error_detail.lower():
                    logger.info(f"Group {group_id} is already running. Continuing with existing group.")
                    raise GroupAlreadyRunningError(f"Group {group_id} is already running")
                
                # Другие ошибки 400 - пробрасываем как RequestException
                raise RequestException(f"Bad request (400): {error_detail}")
                
            except (ValueError, KeyError) as parse_error:
                # Не удалось распарсить JSON ответ
                logger.warning(f"Could not parse error response: {parse_error}")
                raise RequestException(f"Bad request (400): {response.text}")
        
        # Для других статусов используем стандартную обработку
        response.raise_for_status()
        return True
        
    except GroupAlreadyCompletedError:
        # Пробрасываем специальное исключение для завершенной группы
        raise
    except GroupAlreadyRunningError:
        # Группа уже запущена - это не ошибка, возвращаем True
        return True
    except HTTPError as e:
        # Обрабатываем HTTP ошибки (кроме 400, которые уже обработаны выше)
        if e.response is not None and e.response.status_code == 400:
            try:
                error_detail = e.response.json().get("detail", "")
                if "already completed" in error_detail.lower():
                    logger.warning(f"Group {group_id} is already completed.")
                    raise GroupAlreadyCompletedError(f"Group {group_id} is already completed")
                if "already running" in error_detail.lower():
                    logger.info(f"Group {group_id} is already running.")
                    return True
            except (ValueError, KeyError):
                pass
        raise RequestException(f"HTTP error {e.response.status_code if e.response else 'unknown'}: {str(e)}")
    except Timeout:
        logger.error(f"Timeout while starting experiment group {group_id}")
        raise
    except RequestException as e:
        logger.error(f"Failed to start experiment group: {str(e)}")
        raise


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_metrics(group_id: str, base_url: str = "http://localhost:8001", timeout: int = 30) -> Dict[str, Dict[str, Union[float, int]]]:
    """
    Get metrics for the experiment group.
    
    Args:
        group_id (str): ID of the group
        base_url (str): Base URL of the test application API
        timeout (int): Request timeout in seconds
        
    Returns:
        Dict[str, Dict[str, Union[float, int]]]: Metrics for the group
        
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
        
        # Read group state from API response
        group_state = stats.get("state", "unknown")
        
        # Log warning if group is completed
        if group_state == "completed":
            logger.warning(f"Group {group_id} is completed. Metrics may be stale.")
        
        # Process metrics
        metrics = {
            "group": {
                "avg_latency": stats.get("average_latency", 0.0),
                "concurrent_users": stats.get("total_requests", 0),
                "state": group_state  # Add group state to metrics
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
            
        logger.info(f"Retrieved metrics for group {group_id} (state: {group_state})")
        return metrics
        
    except Timeout:
        logger.error(f"Timeout while getting metrics for group {group_id}")
        raise
    except RequestException as e:
        logger.error(f"Failed to get metrics: {str(e)}")
        raise 
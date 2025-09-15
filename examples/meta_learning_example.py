"""
Example usage of meta-learning functionality.

This example demonstrates how to use the meta-learning capabilities
to train models that can quickly adapt to new tasks.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example task configurations for meta-learning
EXAMPLE_TASKS = [
    {
        "task_id": "task_1",
        "name": "High CPU Task",
        "description": "Task with high CPU requirements",
        "node_name": ["node1", "node2", "node3"],
        "max_hardware": {
            "cpu": 8, "ram": 16000, "tx_bandwidth": 1000, "rx_bandwidth": 1000,
            "read_disks_bandwidth": 500, "write_disks_bandwidth": 500, "avg_latency": 300
        },
        "pod_usage": {
            "cpu": 4, "ram": 4000, "tx_bandwidth": 40, "rx_bandwidth": 40,
            "read_disks_bandwidth": 200, "write_disks_bandwidth": 200
        },
        "node_info": {
            "node1": {"cpu": 8, "ram": 16000, "tx_bandwidth": 100, "rx_bandwidth": 100, 
                     "read_disks_bandwidth": 300, "write_disks_bandwidth": 300, "avg_latency": 10},
            "node2": {"cpu": 8, "ram": 16000, "tx_bandwidth": 100, "rx_bandwidth": 100,
                     "read_disks_bandwidth": 300, "write_disks_bandwidth": 300, "avg_latency": 20},
            "node3": {"cpu": 8, "ram": 16000, "tx_bandwidth": 100, "rx_bandwidth": 100,
                     "read_disks_bandwidth": 300, "write_disks_bandwidth": 300, "avg_latency": 30}
        },
        "num_nodes": 3,
        "deployments": [
            "lwmecps-testapp-server-bs1", "lwmecps-testapp-server-bs2",
            "lwmecps-testapp-server-bs3", "lwmecps-testapp-server-bs4"
        ],
        "episodes": 50
    },
    {
        "task_id": "task_2", 
        "name": "High Memory Task",
        "description": "Task with high memory requirements",
        "node_name": ["node1", "node2", "node3"],
        "max_hardware": {
            "cpu": 8, "ram": 16000, "tx_bandwidth": 1000, "rx_bandwidth": 1000,
            "read_disks_bandwidth": 500, "write_disks_bandwidth": 500, "avg_latency": 300
        },
        "pod_usage": {
            "cpu": 2, "ram": 6000, "tx_bandwidth": 20, "rx_bandwidth": 20,
            "read_disks_bandwidth": 100, "write_disks_bandwidth": 100
        },
        "node_info": {
            "node1": {"cpu": 8, "ram": 16000, "tx_bandwidth": 100, "rx_bandwidth": 100,
                     "read_disks_bandwidth": 300, "write_disks_bandwidth": 300, "avg_latency": 10},
            "node2": {"cpu": 8, "ram": 16000, "tx_bandwidth": 100, "rx_bandwidth": 100,
                     "read_disks_bandwidth": 300, "write_disks_bandwidth": 300, "avg_latency": 20},
            "node3": {"cpu": 8, "ram": 16000, "tx_bandwidth": 100, "rx_bandwidth": 100,
                     "read_disks_bandwidth": 300, "write_disks_bandwidth": 300, "avg_latency": 30}
        },
        "num_nodes": 3,
        "deployments": [
            "lwmecps-testapp-server-bs1", "lwmecps-testapp-server-bs2",
            "lwmecps-testapp-server-bs3", "lwmecps-testapp-server-bs4"
        ],
        "episodes": 50
    },
    {
        "task_id": "task_3",
        "name": "Low Latency Task", 
        "description": "Task with low latency requirements",
        "node_name": ["node1", "node2", "node3"],
        "max_hardware": {
            "cpu": 8, "ram": 16000, "tx_bandwidth": 1000, "rx_bandwidth": 1000,
            "read_disks_bandwidth": 500, "write_disks_bandwidth": 500, "avg_latency": 100
        },
        "pod_usage": {
            "cpu": 2, "ram": 2000, "tx_bandwidth": 20, "rx_bandwidth": 20,
            "read_disks_bandwidth": 100, "write_disks_bandwidth": 100
        },
        "node_info": {
            "node1": {"cpu": 8, "ram": 16000, "tx_bandwidth": 100, "rx_bandwidth": 100,
                     "read_disks_bandwidth": 300, "write_disks_bandwidth": 300, "avg_latency": 5},
            "node2": {"cpu": 8, "ram": 16000, "tx_bandwidth": 100, "rx_bandwidth": 100,
                     "read_disks_bandwidth": 300, "write_disks_bandwidth": 300, "avg_latency": 10},
            "node3": {"cpu": 8, "ram": 16000, "tx_bandwidth": 100, "rx_bandwidth": 100,
                     "read_disks_bandwidth": 300, "write_disks_bandwidth": 300, "avg_latency": 15}
        },
        "num_nodes": 3,
        "deployments": [
            "lwmecps-testapp-server-bs1", "lwmecps-testapp-server-bs2",
            "lwmecps-testapp-server-bs3", "lwmecps-testapp-server-bs4"
        ],
        "episodes": 50
    }
]

# New task for adaptation testing
NEW_TASK_CONFIG = {
    "task_id": "new_task",
    "name": "Mixed Requirements Task",
    "description": "Task with mixed CPU and memory requirements",
    "node_name": ["node1", "node2", "node3"],
    "max_hardware": {
        "cpu": 8, "ram": 16000, "tx_bandwidth": 1000, "rx_bandwidth": 1000,
        "read_disks_bandwidth": 500, "write_disks_bandwidth": 500, "avg_latency": 200
    },
    "pod_usage": {
        "cpu": 3, "ram": 3000, "tx_bandwidth": 30, "rx_bandwidth": 30,
        "read_disks_bandwidth": 150, "write_disks_bandwidth": 150
    },
    "node_info": {
        "node1": {"cpu": 8, "ram": 16000, "tx_bandwidth": 100, "rx_bandwidth": 100,
                 "read_disks_bandwidth": 300, "write_disks_bandwidth": 300, "avg_latency": 8},
        "node2": {"cpu": 8, "ram": 16000, "tx_bandwidth": 100, "rx_bandwidth": 100,
                 "read_disks_bandwidth": 300, "write_disks_bandwidth": 300, "avg_latency": 12},
        "node3": {"cpu": 8, "ram": 16000, "tx_bandwidth": 100, "rx_bandwidth": 100,
                 "read_disks_bandwidth": 300, "write_disks_bandwidth": 300, "avg_latency": 16}
    },
    "num_nodes": 3,
    "deployments": [
        "lwmecps-testapp-server-bs1", "lwmecps-testapp-server-bs2",
        "lwmecps-testapp-server-bs3", "lwmecps-testapp-server-bs4"
    ],
    "adaptation_episodes": 20
}


def create_meta_ppo_task() -> Dict[str, Any]:
    """Create a Meta-PPO training task configuration."""
    return {
        "name": "Meta-PPO Training Example",
        "description": "Example meta-learning training using PPO with MAML",
        "model_type": "meta_ppo",
        "meta_method": "maml",
        "tasks": EXAMPLE_TASKS,
        "meta_parameters": {
            "meta_lr": 0.01,
            "inner_lr": 0.01,
            "num_inner_steps": 1,
            "num_meta_epochs": 50
        },
        "parameters": {
            "hidden_size": 64,
            "learning_rate": 3e-4,
            "discount_factor": 0.99,
            "lambda": 0.95,
            "clip_epsilon": 0.2,
            "entropy_coef": 0.01,
            "value_function_coef": 0.5,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "device": "cpu",
            "max_replicas": 10
        },
        "env_config": {
            "base_url": "http://localhost:8001",
            "stabilization_time": 10
        },
        "total_episodes": 100,
        "namespace": "lwmecps-testapp",
        "max_pods": 50,
        "base_url": "http://localhost:8001",
        "stabilization_time": 10
    }


def create_meta_sac_task() -> Dict[str, Any]:
    """Create a Meta-SAC training task configuration."""
    return {
        "name": "Meta-SAC Training Example",
        "description": "Example meta-learning training using SAC with FOMAML",
        "model_type": "meta_sac",
        "meta_method": "fomaml",
        "tasks": EXAMPLE_TASKS,
        "meta_parameters": {
            "meta_lr": 0.01,
            "inner_lr": 0.01,
            "num_inner_steps": 1,
            "num_meta_epochs": 50
        },
        "parameters": {
            "hidden_size": 256,
            "learning_rate": 3e-4,
            "discount_factor": 0.99,
            "tau": 0.005,
            "alpha": 0.2,
            "auto_entropy": True,
            "target_entropy": -1.0,
            "batch_size": 256,
            "device": "cpu",
            "max_replicas": 10
        },
        "env_config": {
            "base_url": "http://localhost:8001",
            "stabilization_time": 10
        },
        "total_episodes": 100,
        "namespace": "lwmecps-testapp",
        "max_pods": 50,
        "base_url": "http://localhost:8001",
        "stabilization_time": 10
    }


def create_meta_td3_task() -> Dict[str, Any]:
    """Create a Meta-TD3 training task configuration."""
    return {
        "name": "Meta-TD3 Training Example",
        "description": "Example meta-learning training using TD3 with MAML",
        "model_type": "meta_td3",
        "meta_method": "maml",
        "tasks": EXAMPLE_TASKS,
        "meta_parameters": {
            "meta_lr": 0.01,
            "inner_lr": 0.01,
            "num_inner_steps": 1,
            "num_meta_epochs": 50
        },
        "parameters": {
            "hidden_size": 256,
            "learning_rate": 3e-4,
            "discount_factor": 0.99,
            "tau": 0.005,
            "policy_delay": 2,
            "noise_clip": 0.5,
            "noise": 0.2,
            "batch_size": 256,
            "device": "cpu",
            "max_replicas": 10
        },
        "env_config": {
            "base_url": "http://localhost:8001",
            "stabilization_time": 10
        },
        "total_episodes": 100,
        "namespace": "lwmecps-testapp",
        "max_pods": 50,
        "base_url": "http://localhost:8001",
        "stabilization_time": 10
    }


def create_meta_dqn_task() -> Dict[str, Any]:
    """Create a Meta-DQN training task configuration."""
    return {
        "name": "Meta-DQN Training Example",
        "description": "Example meta-learning training using DQN with FOMAML",
        "model_type": "meta_dqn",
        "meta_method": "fomaml",
        "tasks": EXAMPLE_TASKS,
        "meta_parameters": {
            "meta_lr": 0.01,
            "inner_lr": 0.01,
            "num_inner_steps": 1,
            "num_meta_epochs": 50
        },
        "parameters": {
            "learning_rate": 0.001,
            "discount_factor": 0.99,
            "epsilon": 0.1,
            "memory_size": 10000,
            "batch_size": 32,
            "device": "cpu",
            "max_replicas": 10
        },
        "env_config": {
            "base_url": "http://localhost:8001",
            "stabilization_time": 10
        },
        "total_episodes": 100,
        "namespace": "lwmecps-testapp",
        "max_pods": 50,
        "base_url": "http://localhost:8001",
        "stabilization_time": 10
    }


async def run_meta_learning_example():
    """Run a complete meta-learning example."""
    logger.info("Starting meta-learning example...")
    
    # This is a conceptual example - in practice, you would use the API endpoints
    # or directly instantiate the services
    
    print("=== Meta-Learning Example ===")
    print()
    
    print("1. Supported Meta-Learning Algorithms:")
    print("   - Meta-PPO (MAML/FOMAML)")
    print("   - Meta-SAC (MAML/FOMAML)")
    print("   - Meta-TD3 (MAML/FOMAML)")
    print("   - Meta-DQN (MAML/FOMAML)")
    print()
    
    print("2. Example Task Configurations:")
    for i, task in enumerate(EXAMPLE_TASKS, 1):
        print(f"   Task {i}: {task['name']} - {task['description']}")
    print()
    
    print("3. Meta-Learning Training Process:")
    print("   a) Create meta-learning task with multiple training tasks")
    print("   b) Train meta-learning algorithm on all tasks")
    print("   c) Meta-algorithm learns good initialization parameters")
    print("   d) Can quickly adapt to new tasks with few gradient steps")
    print()
    
    print("4. Example API Usage:")
    print("   POST /meta-learning/meta-tasks")
    print("   - Create meta-learning training task")
    print()
    print("   POST /meta-learning/meta-tasks/{task_id}/start")
    print("   - Start meta-learning training")
    print()
    print("   GET /meta-learning/meta-tasks/{task_id}/progress")
    print("   - Monitor training progress")
    print()
    print("   POST /meta-learning/meta-tasks/{task_id}/adapt")
    print("   - Adapt to new task")
    print()
    
    print("5. Example Task Configurations:")
    print("   Meta-PPO with MAML:")
    ppo_task = create_meta_ppo_task()
    print(f"   - Model Type: {ppo_task['model_type']}")
    print(f"   - Meta Method: {ppo_task['meta_method']}")
    print(f"   - Number of Tasks: {len(ppo_task['tasks'])}")
    print(f"   - Meta Epochs: {ppo_task['meta_parameters']['num_meta_epochs']}")
    print()
    
    print("   Meta-SAC with FOMAML:")
    sac_task = create_meta_sac_task()
    print(f"   - Model Type: {sac_task['model_type']}")
    print(f"   - Meta Method: {sac_task['meta_method']}")
    print(f"   - Number of Tasks: {len(sac_task['tasks'])}")
    print(f"   - Meta Epochs: {sac_task['meta_parameters']['num_meta_epochs']}")
    print()
    
    print("6. Benefits of Meta-Learning:")
    print("   - Fast adaptation to new tasks")
    print("   - Better sample efficiency")
    print("   - Improved generalization")
    print("   - Reduced training time for new tasks")
    print()
    
    print("7. When to Use Meta-Learning:")
    print("   - Multiple related tasks with similar structure")
    print("   - Need for quick adaptation to new environments")
    print("   - Limited data for new tasks")
    print("   - Desire for better generalization")
    print()
    
    logger.info("Meta-learning example completed!")


def print_task_configuration(task_config: Dict[str, Any], title: str):
    """Print a formatted task configuration."""
    print(f"=== {title} ===")
    print(f"Name: {task_config['name']}")
    print(f"Description: {task_config['description']}")
    print(f"Model Type: {task_config['model_type']}")
    print(f"Meta Method: {task_config['meta_method']}")
    print(f"Number of Tasks: {len(task_config['tasks'])}")
    print(f"Meta Parameters: {json.dumps(task_config['meta_parameters'], indent=2)}")
    print()


if __name__ == "__main__":
    # Run the example
    asyncio.run(run_meta_learning_example())
    
    # Print example configurations
    print_task_configuration(create_meta_ppo_task(), "Meta-PPO Task Configuration")
    print_task_configuration(create_meta_sac_task(), "Meta-SAC Task Configuration")
    print_task_configuration(create_meta_td3_task(), "Meta-TD3 Task Configuration")
    print_task_configuration(create_meta_dqn_task(), "Meta-DQN Task Configuration")
    
    print("=== New Task for Adaptation ===")
    print(f"Task Name: {NEW_TASK_CONFIG['name']}")
    print(f"Description: {NEW_TASK_CONFIG['description']}")
    print(f"Adaptation Episodes: {NEW_TASK_CONFIG['adaptation_episodes']}")
    print()
    
    print("Meta-learning setup complete! Use the API endpoints to start training.")

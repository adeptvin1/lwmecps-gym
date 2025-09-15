"""
Пример изменения размерности сети в мета-обучении.

Этот пример демонстрирует, как использовать адаптивные мета-алгоритмы
для изменения количества нод в кластере без потери изученных знаний.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any
import requests
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base URL for API
BASE_URL = "http://localhost:8000"

# Example configurations for different node counts
NODE_CONFIGURATIONS = {
    "3_nodes": {
        "node_name": ["node1", "node2", "node3"],
        "max_hardware": {
            "cpu": 8, "ram": 16000, "tx_bandwidth": 1000, "rx_bandwidth": 1000,
            "read_disks_bandwidth": 500, "write_disks_bandwidth": 500, "avg_latency": 300
        },
        "pod_usage": {
            "cpu": 2, "ram": 2000, "tx_bandwidth": 20, "rx_bandwidth": 20,
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
        "num_nodes": 3
    },
    "5_nodes": {
        "node_name": ["node1", "node2", "node3", "node4", "node5"],
        "max_hardware": {
            "cpu": 8, "ram": 16000, "tx_bandwidth": 1000, "rx_bandwidth": 1000,
            "read_disks_bandwidth": 500, "write_disks_bandwidth": 500, "avg_latency": 300
        },
        "pod_usage": {
            "cpu": 2, "ram": 2000, "tx_bandwidth": 20, "rx_bandwidth": 20,
            "read_disks_bandwidth": 100, "write_disks_bandwidth": 100
        },
        "node_info": {
            "node1": {"cpu": 8, "ram": 16000, "tx_bandwidth": 100, "rx_bandwidth": 100,
                     "read_disks_bandwidth": 300, "write_disks_bandwidth": 300, "avg_latency": 10},
            "node2": {"cpu": 8, "ram": 16000, "tx_bandwidth": 100, "rx_bandwidth": 100,
                     "read_disks_bandwidth": 300, "write_disks_bandwidth": 300, "avg_latency": 20},
            "node3": {"cpu": 8, "ram": 16000, "tx_bandwidth": 100, "rx_bandwidth": 100,
                     "read_disks_bandwidth": 300, "write_disks_bandwidth": 300, "avg_latency": 30},
            "node4": {"cpu": 8, "ram": 16000, "tx_bandwidth": 100, "rx_bandwidth": 100,
                     "read_disks_bandwidth": 300, "write_disks_bandwidth": 300, "avg_latency": 25},
            "node5": {"cpu": 8, "ram": 16000, "tx_bandwidth": 100, "rx_bandwidth": 100,
                     "read_disks_bandwidth": 300, "write_disks_bandwidth": 300, "avg_latency": 35}
        },
        "num_nodes": 5
    },
    "7_nodes": {
        "node_name": ["node1", "node2", "node3", "node4", "node5", "node6", "node7"],
        "max_hardware": {
            "cpu": 8, "ram": 16000, "tx_bandwidth": 1000, "rx_bandwidth": 1000,
            "read_disks_bandwidth": 500, "write_disks_bandwidth": 500, "avg_latency": 300
        },
        "pod_usage": {
            "cpu": 2, "ram": 2000, "tx_bandwidth": 20, "rx_bandwidth": 20,
            "read_disks_bandwidth": 100, "write_disks_bandwidth": 100
        },
        "node_info": {
            "node1": {"cpu": 8, "ram": 16000, "tx_bandwidth": 100, "rx_bandwidth": 100,
                     "read_disks_bandwidth": 300, "write_disks_bandwidth": 300, "avg_latency": 10},
            "node2": {"cpu": 8, "ram": 16000, "tx_bandwidth": 100, "rx_bandwidth": 100,
                     "read_disks_bandwidth": 300, "write_disks_bandwidth": 300, "avg_latency": 20},
            "node3": {"cpu": 8, "ram": 16000, "tx_bandwidth": 100, "rx_bandwidth": 100,
                     "read_disks_bandwidth": 300, "write_disks_bandwidth": 300, "avg_latency": 30},
            "node4": {"cpu": 8, "ram": 16000, "tx_bandwidth": 100, "rx_bandwidth": 100,
                     "read_disks_bandwidth": 300, "write_disks_bandwidth": 300, "avg_latency": 25},
            "node5": {"cpu": 8, "ram": 16000, "tx_bandwidth": 100, "rx_bandwidth": 100,
                     "read_disks_bandwidth": 300, "write_disks_bandwidth": 300, "avg_latency": 35},
            "node6": {"cpu": 8, "ram": 16000, "tx_bandwidth": 100, "rx_bandwidth": 100,
                     "read_disks_bandwidth": 300, "write_disks_bandwidth": 300, "avg_latency": 40},
            "node7": {"cpu": 8, "ram": 16000, "tx_bandwidth": 100, "rx_bandwidth": 100,
                     "read_disks_bandwidth": 300, "write_disks_bandwidth": 300, "avg_latency": 45}
        },
        "num_nodes": 7
    }
}

# Scaling strategies with their characteristics
SCALING_STRATEGIES = {
    "zero_padding": {
        "name": "Zero Padding",
        "description": "Simple strategy that adds zero weights for new nodes",
        "complexity": "low",
        "best_for": "small changes (1-2 nodes)",
        "pros": ["Fast", "Simple", "No data loss"],
        "cons": ["May not preserve learned patterns", "Limited effectiveness for large changes"]
    },
    "weight_interpolation": {
        "name": "Weight Interpolation", 
        "description": "Interpolates weights between existing nodes",
        "complexity": "medium",
        "best_for": "medium changes (3-5 nodes)",
        "pros": ["Preserves learned patterns", "Smooth transition", "Good for moderate changes"],
        "cons": ["More complex than zero padding", "May not work well for very different node types"]
    },
    "knowledge_distillation": {
        "name": "Knowledge Distillation",
        "description": "Uses knowledge distillation to transfer information",
        "complexity": "high",
        "best_for": "large changes (5+ nodes)",
        "pros": ["Preserves complex patterns", "Works well for large changes", "Maintains performance"],
        "cons": ["Computationally expensive", "Requires more time", "Complex implementation"]
    },
    "attention_based": {
        "name": "Attention-Based",
        "description": "Uses attention mechanism for adaptation",
        "complexity": "high",
        "best_for": "complex changes with varying node characteristics",
        "pros": ["Handles complex relationships", "Adaptive to node differences", "State-of-the-art performance"],
        "cons": ["Most complex", "Requires significant computation", "May be overkill for simple changes"]
    }
}


def create_meta_learning_task_with_3_nodes() -> Dict[str, Any]:
    """Create a meta-learning task starting with 3 nodes."""
    return {
        "name": "Adaptive Meta-PPO Training (3 Nodes)",
        "description": "Meta-learning training with adaptive node scaling starting from 3 nodes",
        "model_type": "meta_ppo",
        "meta_method": "maml",
        "tasks": [
            {
                "task_id": "task_3nodes_1",
                "name": "High CPU Task (3 nodes)",
                **NODE_CONFIGURATIONS["3_nodes"],
                "episodes": 30
            },
            {
                "task_id": "task_3nodes_2", 
                "name": "High Memory Task (3 nodes)",
                **NODE_CONFIGURATIONS["3_nodes"],
                "episodes": 30
            }
        ],
        "meta_parameters": {
            "meta_lr": 0.01,
            "inner_lr": 0.01,
            "num_inner_steps": 1,
            "num_meta_epochs": 20
        },
        "parameters": {
            "hidden_size": 64,
            "learning_rate": 3e-4,
            "discount_factor": 0.99,
            "batch_size": 64,
            "max_nodes": 20,  # Maximum nodes the model can handle
            "initial_nodes": 3
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


def create_node_scaling_config(target_nodes: int, strategy: str = "weight_interpolation") -> Dict[str, Any]:
    """Create node scaling configuration."""
    return {
        "new_num_nodes": target_nodes,
        "strategy": strategy,
        "node_info": NODE_CONFIGURATIONS[f"{target_nodes}_nodes"]["node_info"],
        "adaptation_episodes": 10
    }


async def run_node_scaling_example():
    """Run a complete node scaling example."""
    logger.info("Starting node scaling example...")
    
    print("=== Node Scaling Example ===")
    print()
    
    # 1. Create initial meta-learning task
    print("1. Creating meta-learning task with 3 nodes...")
    task_data = create_meta_learning_task_with_3_nodes()
    
    try:
        response = requests.post(f"{BASE_URL}/meta-learning/meta-tasks", json=task_data)
        if response.status_code == 200:
            task = response.json()
            task_id = task["id"]
            print(f"   ✓ Task created with ID: {task_id}")
        else:
            print(f"   ✗ Failed to create task: {response.text}")
            return
    except requests.exceptions.RequestException as e:
        print(f"   ✗ API request failed: {e}")
        return
    
    # 2. Start meta-learning training
    print("\n2. Starting meta-learning training...")
    try:
        response = requests.post(f"{BASE_URL}/meta-learning/meta-tasks/{task_id}/start")
        if response.status_code == 200:
            print("   ✓ Meta-learning training started")
        else:
            print(f"   ✗ Failed to start training: {response.text}")
            return
    except requests.exceptions.RequestException as e:
        print(f"   ✗ API request failed: {e}")
        return
    
    # 3. Wait for training to complete (in real scenario)
    print("\n3. Waiting for training to complete...")
    print("   (In real scenario, you would monitor progress)")
    
    # 4. Demonstrate node scaling
    print("\n4. Demonstrating node scaling...")
    
    scaling_scenarios = [
        (5, "weight_interpolation", "Medium change: 3 → 5 nodes"),
        (7, "knowledge_distillation", "Large change: 5 → 7 nodes"),
        (4, "zero_padding", "Small change: 7 → 4 nodes")
    ]
    
    for target_nodes, strategy, description in scaling_scenarios:
        print(f"\n   {description}")
        print(f"   Strategy: {SCALING_STRATEGIES[strategy]['name']}")
        print(f"   Complexity: {SCALING_STRATEGIES[strategy]['complexity']}")
        
        # Create scaling configuration
        scaling_config = create_node_scaling_config(target_nodes, strategy)
        
        try:
            response = requests.post(
                f"{BASE_URL}/meta-learning/meta-tasks/{task_id}/adapt-nodes",
                json=scaling_config
            )
            if response.status_code == 200:
                result = response.json()
                print(f"   ✓ Successfully scaled to {target_nodes} nodes")
                print(f"   Adaptation result: {result['adaptation_result']['success']}")
            else:
                print(f"   ✗ Failed to scale nodes: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"   ✗ API request failed: {e}")
    
    # 5. Show architecture history
    print("\n5. Retrieving architecture history...")
    try:
        response = requests.get(f"{BASE_URL}/meta-learning/meta-tasks/{task_id}/architecture-history")
        if response.status_code == 200:
            history = response.json()
            print(f"   ✓ Retrieved {len(history['architecture_history'])} architecture changes")
            for i, change in enumerate(history['architecture_history'], 1):
                print(f"   Change {i}: {change['old_nodes']} → {change['new_nodes']} nodes using {change['strategy']}")
        else:
            print(f"   ✗ Failed to get history: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"   ✗ API request failed: {e}")
    
    # 6. Show current node count
    print("\n6. Checking current node count...")
    try:
        response = requests.get(f"{BASE_URL}/meta-learning/meta-tasks/{task_id}/current-nodes")
        if response.status_code == 200:
            node_info = response.json()
            print(f"   ✓ Current nodes: {node_info['current_nodes']}")
            print(f"   Max nodes: {node_info['max_nodes']}")
            print(f"   Adaptation capable: {node_info['adaptation_capable']}")
        else:
            print(f"   ✗ Failed to get node info: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"   ✗ API request failed: {e}")
    
    print("\n=== Node Scaling Example Completed ===")


def print_scaling_strategies():
    """Print information about scaling strategies."""
    print("=== Node Scaling Strategies ===")
    print()
    
    for strategy, info in SCALING_STRATEGIES.items():
        print(f"Strategy: {info['name']} ({strategy})")
        print(f"Description: {info['description']}")
        print(f"Complexity: {info['complexity']}")
        print(f"Best for: {info['best_for']}")
        print(f"Pros: {', '.join(info['pros'])}")
        print(f"Cons: {', '.join(info['cons'])}")
        print()


def print_node_configurations():
    """Print information about node configurations."""
    print("=== Node Configurations ===")
    print()
    
    for config_name, config in NODE_CONFIGURATIONS.items():
        print(f"Configuration: {config_name}")
        print(f"Number of nodes: {config['num_nodes']}")
        print(f"Node names: {', '.join(config['node_name'])}")
        print(f"Hardware specs: CPU={config['max_hardware']['cpu']}, RAM={config['max_hardware']['ram']}MB")
        print(f"Pod usage: CPU={config['pod_usage']['cpu']}, RAM={config['pod_usage']['ram']}MB")
        print()


def demonstrate_api_usage():
    """Demonstrate API usage for node scaling."""
    print("=== API Usage Examples ===")
    print()
    
    print("1. Create meta-learning task:")
    print("   POST /meta-learning/meta-tasks")
    print("   Content-Type: application/json")
    print("   {")
    print('     "name": "Adaptive Meta-PPO Training",')
    print('     "model_type": "meta_ppo",')
    print('     "meta_method": "maml",')
    print('     "tasks": [...],')
    print('     "parameters": {"max_nodes": 20, "initial_nodes": 3}')
    print("   }")
    print()
    
    print("2. Start meta-learning training:")
    print("   POST /meta-learning/meta-tasks/{task_id}/start")
    print()
    
    print("3. Scale to new number of nodes:")
    print("   POST /meta-learning/meta-tasks/{task_id}/adapt-nodes")
    print("   Content-Type: application/json")
    print("   {")
    print('     "new_num_nodes": 5,')
    print('     "strategy": "weight_interpolation",')
    print('     "node_info": {...}')
    print("   }")
    print()
    
    print("4. Get architecture history:")
    print("   GET /meta-learning/meta-tasks/{task_id}/architecture-history")
    print()
    
    print("5. Get current node count:")
    print("   GET /meta-learning/meta-tasks/{task_id}/current-nodes")
    print()
    
    print("6. Get supported scaling strategies:")
    print("   GET /meta-learning/supported-scaling-strategies")
    print()


def print_benefits():
    """Print benefits of node scaling."""
    print("=== Benefits of Node Scaling ===")
    print()
    
    benefits = [
        "Dynamic Adaptation: Models can adapt to changing cluster sizes without retraining",
        "Knowledge Preservation: Learned patterns are preserved during scaling",
        "Multiple Strategies: Different strategies for different scaling scenarios",
        "Real-time Scaling: Can scale up or down based on cluster needs",
        "Performance Maintenance: Maintains performance after scaling",
        "Flexible Architecture: Supports various node configurations",
        "Cost Efficiency: No need to retrain from scratch",
        "Production Ready: Suitable for production environments with dynamic workloads"
    ]
    
    for i, benefit in enumerate(benefits, 1):
        print(f"{i}. {benefit}")
    
    print()


if __name__ == "__main__":
    # Print information about the example
    print_scaling_strategies()
    print_node_configurations()
    print_benefits()
    demonstrate_api_usage()
    
    # Run the example (uncomment to run actual API calls)
    # asyncio.run(run_node_scaling_example())
    
    print("Node scaling example setup complete!")
    print("Uncomment the last line to run the actual example with API calls.")

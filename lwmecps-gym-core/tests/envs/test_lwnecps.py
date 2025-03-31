import pytest
import numpy as np
from lwmecps_gym_core.envs import LWMECPSEnv

@pytest.fixture
def mock_node_info():
    return {
        "node1": {
            "cpu": "4",
            "memory": "8Gi",
            "tx_bandwidth": "1000",
            "rx_bandwidth": "1000",
            "read_disks_bandwidth": "500",
            "write_disks_bandwidth": "500",
            "avg_latency": "10",
            "deployments": {
                "default": {
                    "app1": {
                        "replicas": 2,
                        "cpu_request": "1",
                        "memory_request": "2Gi"
                    }
                }
            }
        }
    }

@pytest.fixture
def env_params():
    return {
        "node_name": ["node1"],
        "max_hardware": {
            "cpu": 4,
            "ram": 8192,  # 8Gi in MB
            "tx_bandwidth": 1000,
            "rx_bandwidth": 1000,
            "read_disks_bandwidth": 500,
            "write_disks_bandwidth": 500,
            "avg_latency": 10
        },
        "pod_usage": {},
        "node_info": {},
        "num_nodes": 1,
        "namespace": "default",
        "deployment_name": "app1",
        "deployments": ["app1"],
        "max_pods": 10
    }

def test_env_initialization(env_params, mock_node_info):
    """Test environment initialization"""
    env = LWMECPSEnv(**env_params)
    assert env.num_nodes == 1
    assert env.node_name == ["node1"]
    assert env.namespace == "default"
    assert env.deployment_name == "app1"
    assert env.max_pods == 10

def test_observation_space(env_params):
    """Test observation space structure"""
    env = LWMECPSEnv(**env_params)
    obs_space = env.observation_space
    assert "node1" in obs_space.spaces
    node_space = obs_space.spaces["node1"]
    assert "cpu" in node_space.spaces
    assert "ram" in node_space.spaces
    assert "tx_bandwidth" in node_space.spaces
    assert "rx_bandwidth" in node_space.spaces
    assert "read_disks_bandwidth" in node_space.spaces
    assert "write_disks_bandwidth" in node_space.spaces
    assert "avg_latency" in node_space.spaces
    assert "deployments" in node_space.spaces

def test_action_space(env_params):
    """Test action space"""
    env = LWMECPSEnv(**env_params)
    assert env.action_space.n == 1  # One node

def test_reset(env_params, mock_node_info):
    """Test environment reset"""
    env = LWMECPSEnv(**env_params)
    observation, info = env.reset()
    assert isinstance(observation, dict)
    assert "node1" in observation
    assert isinstance(info, dict)

def test_step(env_params):
    """Test environment step"""
    env = LWMECPSEnv(**env_params)
    observation, info = env.reset()
    action = 0  # Select first node
    next_observation, reward, terminated, truncated, info = env.step(action)
    assert isinstance(next_observation, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict) 
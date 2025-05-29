# LWMECPS Gym

A reinforcement learning environment for Lightweight MEC Placement Strategy (LWMECPS) in Kubernetes clusters.

## Overview

This project provides a Gymnasium environment for training reinforcement learning agents to optimize MEC (Multi-Access Edge Computing) placement in Kubernetes clusters. It supports multiple reinforcement learning algorithms including Q-Learning, DQN, PPO, TD3, and SAC.

## Key Features

- **Kubernetes Integration**: Direct interaction with Kubernetes clusters through the official Python client
- **Multiple RL Algorithms**: Support for Q-Learning, DQN, PPO, TD3, and SAC
- **Experiment Tracking**: Integration with Weights & Biases for experiment monitoring and visualization
- **MongoDB Storage**: Persistent storage for training tasks and results
- **RESTful API**: FastAPI-based endpoints for task management and monitoring
- **System Stabilization**: Configurable stabilization time after pod movements

## Environment Configuration

The environment can be configured with the following parameters:

```python
env = gym.make(
    "lwmecps-v3",
    node_name=node_name,
    max_hardware=max_hardware,
    pod_usage=pod_usage,
    node_info=node_info,
    num_nodes=len(node_name),
    namespace="default",
    deployment_name="mec-test-app",
    deployments=["mec-test-app"],
    max_pods=10000,
    group_id="test-group-1",
    env_config={
        "base_url": "http://localhost:8001",
        "stabilization_time": 10  # Time in seconds to wait after pod movement
    }
)
```

### Environment Parameters

- `node_name`: List of node names in the cluster
- `max_hardware`: Maximum hardware resources available
- `pod_usage`: Resource usage per pod
- `node_info`: Information about each node's resources and latency
- `num_nodes`: Number of nodes in the cluster
- `namespace`: Kubernetes namespace
- `deployment_name`: Name of the deployment to manage
- `deployments`: List of deployment names
- `max_pods`: Maximum number of pods allowed
- `group_id`: Unique identifier for the experiment group
- `env_config`: Additional configuration options
  - `base_url`: Base URL for the test application API
  - `stabilization_time`: Time in seconds to wait after pod movement for system stabilization (default: 10, integer)

## Project Structure

```
lwmecps_gym/
├── api/                    # FastAPI endpoints and routers
├── core/                   # Core functionality (database, models, config)
├── envs/                   # Gymnasium environment implementation
├── ml/                     # Machine learning models and training service
└── tests/                  # Test suite
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Start the services:
```bash
docker-compose up -d
```

4. Create and start a training task:

```bash
# Q-Learning Example
curl -X POST "http://localhost:8010/api/v1/training/tasks" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "Q-Learning Training",
       "description": "Training Q-Learning agent for pod placement",
       "model_type": "q_learning",
       "parameters": {
         "learning_rate": 0.1,
         "discount_factor": 0.95,
         "exploration_rate": 1.0,
         "exploration_decay": 0.995
       },
       "total_episodes": 1000
     }'

# PPO Training Example
curl -X POST "http://localhost:8010/api/v1/training/tasks" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "PPO Training",
       "description": "Training PPO agent for pod placement",
       "model_type": "ppo",
       "parameters": {
         "learning_rate": 3e-4,
         "discount_factor": 0.99,
         "lambda": 0.95,
         "clip_epsilon": 0.2,
         "entropy_coef": 0.0,
         "value_function_coef": 0.5,
         "n_steps": 2048,
         "batch_size": 64,
         "n_epochs": 10,
         "device": "cpu",
         "deployments": ["mec-test-app"]
       },
       "total_episodes": 1000
     }'

# TD3 Training Example
curl -X POST "http://localhost:8010/api/v1/training/tasks" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "TD3 Training",
       "description": "Training TD3 agent for pod placement",
       "model_type": "td3",
       "parameters": {
         "learning_rate": 3e-4,
         "discount_factor": 0.99,
         "tau": 0.005,
         "policy_delay": 2,
         "noise_clip": 0.5,
         "noise": 0.2,
         "batch_size": 256,
         "device": "cpu",
         "deployments": ["mec-test-app"]
       },
       "total_episodes": 1000
     }'

# SAC Training Example
curl -X POST "http://localhost:8010/api/v1/training/tasks" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "SAC Training",
       "description": "Training SAC agent for pod placement",
       "model_type": "sac",
       "parameters": {
         "learning_rate": 3e-4,
         "discount_factor": 0.99,
         "tau": 0.005,
         "alpha": 0.2,
         "auto_entropy": true,
         "target_entropy": -1.0,
         "batch_size": 256,
         "device": "cpu",
         "deployments": ["mec-test-app"]
       },
       "total_episodes": 1000
     }'
```

## API Endpoints

- `POST /api/v1/training/tasks`: Create a new training task
- `GET /api/v1/training/tasks/{task_id}`: Get task details
- `POST /api/v1/training/tasks/{task_id}/start`: Start training
- `POST /api/v1/training/tasks/{task_id}/pause`: Pause training
- `POST /api/v1/training/tasks/{task_id}/resume`: Resume training
- `POST /api/v1/training/tasks/{task_id}/stop`: Stop training
- `GET /api/v1/training/tasks/{task_id}/progress`: Get training progress

## Algorithm Parameters

### Q-Learning
- `learning_rate`: Learning rate for Q-value updates (default: 0.1)
- `discount_factor`: Discount factor for future rewards (default: 0.95)
- `exploration_rate`: Initial exploration rate (default: 1.0)
- `exploration_decay`: Rate at which exploration rate decays (default: 0.995)

### DQN
- `learning_rate`: Learning rate for neural network (default: 0.001)
- `discount_factor`: Discount factor for future rewards (default: 0.99)
- `epsilon`: Exploration rate (default: 0.1)
- `memory_size`: Size of replay buffer (default: 10000)
- `batch_size`: Batch size for training (default: 32)

### PPO
- `learning_rate`: Learning rate for neural network (default: 3e-4)
- `discount_factor`: Discount factor for future rewards (default: 0.99)
- `lambda`: GAE-Lambda parameter (default: 0.95)
- `clip_epsilon`: PPO clip parameter (default: 0.2)
- `entropy_coef`: Entropy coefficient (default: 0.0)
- `value_function_coef`: Value function coefficient (default: 0.5)
- `n_steps`: Number of steps per update (default: 2048)
- `batch_size`: Batch size for training (default: 64)
- `n_epochs`: Number of epochs per update (default: 10)
- `device`: Device to use for training (default: "cpu")

### TD3
- `learning_rate`: Learning rate for neural networks (default: 3e-4)
- `discount_factor`: Discount factor for future rewards (default: 0.99)
- `tau`: Target network update rate (default: 0.005)
- `policy_delay`: Policy update delay (default: 2)
- `noise_clip`: Noise clip range (default: 0.5)
- `noise`: Action noise scale (default: 0.2)
- `batch_size`: Batch size for training (default: 256)
- `device`: Device to use for training (default: "cpu")

### SAC
- `learning_rate`: Learning rate for neural networks (default: 3e-4)
- `discount_factor`: Discount factor for future rewards (default: 0.99)
- `tau`: Target network update rate (default: 0.005)
- `alpha`: Initial entropy coefficient (default: 0.2)
- `auto_entropy`: Whether to automatically tune entropy (default: true)
- `target_entropy`: Target entropy for automatic tuning (default: -1.0)
- `batch_size`: Batch size for training (default: 256)
- `device`: Device to use for training (default: "cpu")

## Helm Deployment

The project includes a Helm chart for easy deployment to Kubernetes clusters.

### Prerequisites

- Kubernetes 1.19+
- Helm 3.0+
- Docker registry access
- Weights & Biases API key

### Building the Docker Image

1. Build the Docker image:
```bash
docker build -t adeptvin4/lwmecps-gym:latest .
```

2. Push the image to your registry:
```bash
docker push adeptvin4/lwmecps-gym:latest
```

### Installing the Chart

1. Add the chart repository:
```bash
helm repo add lwmecps-gym https://adeptvin1.github.io/lwmecps-gym
helm repo update
```

2. Create a values file (e.g., `my-values.yaml`):
```yaml
image:
  repository: adeptvin4/lwmecps-gym
  tag: latest

wandb:
  apiKey: your-wandb-api-key
  projectName: lwmecps-gym
  entity: your-entity

mongodb:
  persistence:
    size: 10Gi

resources:
  limits:
    cpu: 1000m
    memory: 1Gi
  requests:
    cpu: 500m
    memory: 512Mi
```

3. Install the chart:
```bash
helm install lwmecps-gym ./helm/lwmecps-gym \
  --namespace your-namespace \
  --create-namespace \
  -f my-values.yaml
```

### Configuration

The following table lists the configurable parameters of the chart and their default values:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `image.repository` | Container image repository | `adeptvin4/lwmecps-gym` |
| `image.tag` | Container image tag | `latest` |
| `service.type` | Kubernetes service type | `ClusterIP` |
| `service.port` | Service port | `8000` |
| `mongodb.enabled` | Enable MongoDB deployment | `true` |
| `mongodb.persistence.size` | MongoDB PVC size | `10Gi` |
| `wandb.enabled` | Enable Weights & Biases integration | `true` |
| `wandb.apiKey` | Weights & Biases API key | `""` |
| `wandb.projectName` | Weights & Biases project name | `lwmecps-gym` |
| `wandb.entity` | Weights & Biases entity | `""` |
| `training.defaultParameters` | Default training parameters | See values.yaml |
| `kubernetes.server` | Kubernetes API server URL | `https://kubernetes.default.svc` |
| `kubernetes.namespace` | Target namespace for deployments | `default` |

### Upgrading

To upgrade the deployment:

```bash
helm upgrade lwmecps-gym ./helm/lwmecps-gym \
  --namespace your-namespace \
  -f my-values.yaml
```

### Uninstalling

To uninstall the deployment:

```bash
helm uninstall lwmecps-gym --namespace your-namespace
```

## Development

1. Set up development environment:
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements-dev.txt
```

2. Run tests:
```bash
pytest
```

3. Format code:
```bash
black .
isort .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Description

Данный репозиторий используется для создания окружения Kubernetes кластера и отработки использования RL в задачах опитимизации размещения вычислительных сервисов в геораспределенных узлах обработки данных (т.е k8s размазанный по условному городу, где сеть сотовой связи является транспортной).

## Installation

### From Repository

```bash
git clone https://github.com/adeptvin1/lwmecps-gym.git
cd lwmecps-gym
pip install -e .
```

### How to start work with lib

Вам необходимо сделать следующие вещи:
1. Установить minikube используя инструкцию -> [minikube install](https://minikube.sigs.k8s.io/docs/start/?arch=%2Fmacos%2Farm64%2Fstable%2Fbinary+download)
2. Установить kubectl.
3. Запустить minikube используя команду `minikube start --nodes 4 -p minikube`
4. Запустить деплой mock сервиса используя команду `kubectl apply -f deployment.yml` (вы можете поправить его при необходимости)
5. Проверить состояние подов можно используя команду `kubectl get pods`
6. Чтобы запустить тестовый сервис используйте команду `python3 ./test_service.py`

## Environment

### Action


## Resources

`https://www.comet.com/docs/v2/guides/quickstart/` - Интересная штука для визализации обучения
`https://gymnasium.farama.org/api/spaces/composite/` - Документация по Gymnasium

## Local Development

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Minikube
- Weights & Biases account

### Installation

1. Clone the repository:
```bash
git clone https://github.com/adeptvin1/lwmecps-gym.git
cd lwmecps-gym
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
cp .env.wandb.example .env.wandb
```

Edit `.env` and `.env.wandb` with your configuration.

### Running Locally

1. Start Minikube:
```bash
minikube start
```

2. Start MongoDB and the application:
```bash
docker-compose up -d
```

3. Access the API:
```bash
# Port-forward the service to localhost
kubectl port-forward svc/lwmecps-gym 8010:8010 -n default

# Now you can access the API at http://localhost:8010
```

4. Create a training task:
```bash
curl -X POST "http://localhost:8010/api/v1/training/tasks" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "Q-Learning Training",
       "description": "Training Q-Learning agent for pod placement",
       "model_type": "q_learning",
       "parameters": {
         "learning_rate": 0.1,
         "discount_factor": 0.95,
         "exploration_rate": 1.0,
         "exploration_decay": 0.995
       },
       "total_episodes": 1000
     }'
```

## API Endpoints

- `POST /api/v1/training/tasks`: Create a new training task
- `GET /api/v1/training/tasks/{task_id}`: Get task details
- `POST /api/v1/training/tasks/{task_id}/start`: Start training
- `POST /api/v1/training/tasks/{task_id}/pause`: Pause training
- `POST /api/v1/training/tasks/{task_id}/resume`: Resume training
- `POST /api/v1/training/tasks/{task_id}/stop`: Stop training
- `GET /api/v1/training/tasks/{task_id}/progress`: Get training progress

## Helm Deployment

The project includes a Helm chart for easy deployment to Kubernetes clusters.

### Prerequisites

- Kubernetes 1.19+
- Helm 3.0+
- Docker registry access
- Weights & Biases API key

### Building the Docker Image

1. Build the Docker image:
```bash
docker build -t adeptvin4/lwmecps-gym:latest .
```

2. Push the image to your registry:
```bash
docker push adeptvin4/lwmecps-gym:latest
```

### Installing the Chart

1. Add the chart repository:
```bash
helm repo add lwmecps-gym https://adeptvin1.github.io/lwmecps-gym
helm repo update
```

2. Create a values file (e.g., `my-values.yaml`):
```yaml
image:
  repository: adeptvin4/lwmecps-gym
  tag: latest

wandb:
  apiKey: your-wandb-api-key
  projectName: lwmecps-gym
  entity: your-entity

mongodb:
  persistence:
    size: 10Gi

resources:
  limits:
    cpu: 1000m
    memory: 1Gi
  requests:
    cpu: 500m
    memory: 512Mi
```

3. Install the chart:
```bash
helm install lwmecps-gym ./helm/lwmecps-gym \
  --namespace your-namespace \
  --create-namespace \
  -f my-values.yaml
```

4. Access the API:
```bash
# Port-forward the service to localhost
kubectl port-forward svc/lwmecps-gym 8000:8000 -n your-namespace

# Now you can access the API at http://localhost:8000
```

### Configuration

The following table lists the configurable parameters of the chart and their default values:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `image.repository` | Container image repository | `adeptvin4/lwmecps-gym` |
| `image.tag` | Container image tag | `latest` |
| `service.type` | Kubernetes service type | `ClusterIP` |
| `service.port` | Service port | `8000` |
| `mongodb.enabled` | Enable MongoDB deployment | `true` |
| `mongodb.persistence.size` | MongoDB PVC size | `10Gi` |
| `wandb.enabled` | Enable Weights & Biases integration | `true` |
| `wandb.apiKey` | Weights & Biases API key | `""` |
| `wandb.projectName` | Weights & Biases project name | `lwmecps-gym` |
| `wandb.entity` | Weights & Biases entity | `""` |
| `training.defaultParameters` | Default training parameters | See values.yaml |
| `kubernetes.server` | Kubernetes API server URL | `https://kubernetes.default.svc` |
| `kubernetes.namespace` | Target namespace for deployments | `default`
# LWMECPS Gym

A reinforcement learning environment for Lightweight MEC Placement Strategy (LWMECPS) in Kubernetes clusters.

## Overview

This project provides a Gymnasium environment for training reinforcement learning agents to optimize MEC (Multi-Access Edge Computing) placement in Kubernetes clusters. It uses Q-Learning to learn optimal pod placement strategies based on node resources and network conditions.

## Key Features

- **Kubernetes Integration**: Direct interaction with Kubernetes clusters through the official Python client
- **Q-Learning Implementation**: Custom Q-Learning agent with epsilon-greedy exploration strategy
- **Experiment Tracking**: Integration with Weights & Biases for experiment monitoring and visualization
- **MongoDB Storage**: Persistent storage for training tasks and results
- **RESTful API**: FastAPI-based endpoints for task management and monitoring

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
curl -X POST "http://localhost:8000/api/v1/training/tasks" \
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
kubectl port-forward svc/lwmecps-gym 8000:8000 -n default

# Now you can access the API at http://localhost:8000
```

4. Create a training task:
```bash
curl -X POST "http://localhost:8000/api/v1/training/tasks" \
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
helm uninstall lwmecps-gym -n your-namespace
```


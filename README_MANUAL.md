# LWMECPS Gym - Manual Setup Guide

This guide provides instructions for setting up and running the LWMECPS Gym service locally, while running other components in Minikube.

## Prerequisites

- Python 3.9 or higher
- Minikube
- kubectl
- Weights & Biases account (optional, for experiment tracking)

## Installation Steps

1. **Install Minikube and kubectl**:
```bash
# macOS with Homebrew
brew install minikube
brew install kubectl

# Linux
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Windows (using PowerShell as Administrator)
winget install minikube
winget install kubernetes-cli
```

2. **Clone the repository**:
```bash
git clone https://github.com/adeptvin1/lwmecps-gym.git
cd lwmecps-gym
```

3. **Create and activate virtual environment**:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

4. **Install dependencies and package**:
```bash
# Install required packages
pip install -r requirements.txt

# Install the package in development mode
# Make sure you're in the root directory of the project
pip install -e .

# Verify installation
python -c "import lwmecps_gym; print(lwmecps_gym.__file__)"
```

Если вы видите ошибку `ModuleNotFoundError: No module named 'lwmecps_gym'`, выполните следующие шаги:

1. **Проверьте структуру проекта**:
```bash
# Должна быть такая структура:
lwmecps-gym/
├── src/
│   └── lwmecps_gym/
│       ├── __init__.py
│       ├── main.py
│       └── ...
├── setup.py
└── requirements.txt
```

2. **Переустановите пакет**:
```bash
# Удалите текущую установку
pip uninstall lwmecps-gym

# Очистите кэш pip
pip cache purge

# Переустановите пакет
pip install -e .
```

3. **Проверьте PYTHONPATH**:
```bash
# Добавьте путь к src в PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Или запустите с явным указанием PYTHONPATH
PYTHONPATH=$PYTHONPATH:$(pwd)/src python -m uvicorn src.lwmecps_gym.main:app --host 0.0.0.0 --port 8010 --reload
```

## Minikube Setup

1. **Start Minikube with multiple nodes**:
```bash
# Start Minikube with 4 nodes
minikube start --nodes 4 -p minikube

# Verify nodes
kubectl get nodes
```

2. **Deploy MongoDB to Minikube**:
```bash
# Create namespace
kubectl create namespace lwmecps

# Deploy MongoDB
kubectl apply -f k8s/mongodb.yaml -n lwmecps

# Wait for MongoDB to be ready
kubectl wait --for=condition=ready pod -l app=mongodb -n lwmecps
```

3. **Deploy test application**:
```bash
# Deploy test application
kubectl apply -f deployment.yml

# Verify deployment
kubectl get pods
```

## Configuration

1. **Create environment file**:
Create a `.env` file in the root directory with the following content:

```env
# MongoDB Configuration
MONGODB_URL=mongodb://localhost:27017/lwmecps

# Weights & Biases Configuration (Optional)
WANDB_API_KEY=your_wandb_api_key
WANDB_PROJECT_NAME=lwmecps-gym
WANDB_ENTITY=your_wandb_entity
WANDB_MODE=offline  # or 'online' if you have W&B account

# Kubernetes Configuration
KUBERNETES_SERVER=https://localhost:8443
KUBERNETES_NAMESPACE=default
KUBERNETES_DEPLOYMENT_NAME=mec-test-app
KUBERNETES_MAX_PODS=10000
```

2. **Configure kubectl context and start port-forwarding**:
```bash
# Get Minikube context
kubectl config use-context minikube

# Verify connection
kubectl cluster-info

# Start port-forwarding for MongoDB (in a separate terminal)
kubectl port-forward svc/mongodb 27017:27017 -n lwmecps

# Start port-forwarding for Kubernetes API (in a separate terminal)
kubectl port-forward --address 0.0.0.0 -n kube-system svc/kubernetes 8443:443
```

## Running the Service

1. **Start the application**:
```bash
python -m uvicorn src.lwmecps_gym.main:app --host 0.0.0.0 --port 8010 --reload
```

2. **Access the API**:
- Main API: http://localhost:8010
- Swagger UI: http://localhost:8010/docs
- ReDoc: http://localhost:8010/redoc

## Development Workflow

1. **Code Structure**:
```
lwmecps_gym/
├── api/                    # FastAPI endpoints and routers
├── core/                   # Core functionality (database, models, config)
├── envs/                   # Gymnasium environment implementation
├── ml/                     # Machine learning models and training service
└── tests/                  # Test suite
```

2. **Running Tests**:
```bash
pytest
```

3. **Code Formatting**:
```bash
black .
isort .
```

## Troubleshooting

### Minikube Issues
- **Minikube Won't Start**:
  - Check system requirements: `minikube start --help`
  - Verify virtualization is enabled
  - Check logs: `minikube logs`
  - Try resetting: `minikube delete && minikube start --nodes 4`

### MongoDB Issues
- **Connection Failed**:
  - Check if port-forward is running: `kubectl get pods -n lwmecps`
  - Verify port-forward process: `ps aux | grep port-forward`
  - Check MongoDB logs: `kubectl logs -l app=mongodb -n lwmecps`
  - Test connection: `mongosh mongodb://localhost:27017/lwmecps`
  - Restart port-forward if needed:
    ```bash
    # Kill existing port-forward
    pkill -f "port-forward"
    # Start new port-forward
    kubectl port-forward svc/mongodb 27017:27017 -n lwmecps
    ```

### Kubernetes API Issues
- **Connection Failed**:
  - Check if port-forward is running: `ps aux | grep port-forward`
  - Verify kubectl config: `kubectl config view`
  - Restart port-forward: `kubectl port-forward --address 0.0.0.0 -n kube-system svc/kubernetes 8443:443`

### Weights & Biases Issues
- **Integration Failed**:
  - Verify API key
  - Check internet connection
  - Try offline mode first
  - Check W&B logs

### Application Issues
- **Service Won't Start**:
  - Check port availability: `lsof -i :8010`
  - Verify Python version: `python --version`
  - Check dependencies: `pip list`
  - Review application logs

## Common Commands

### Minikube
```bash
# Start Minikube
minikube start --nodes 4 -p minikube

# Stop Minikube
minikube stop

# Delete Minikube
minikube delete

# Get Minikube status
minikube status

# Access Minikube dashboard
minikube dashboard
```

### Kubernetes
```bash
# Get pods
kubectl get pods -A

# Get services
kubectl get svc -A

# Get logs
kubectl logs -f <pod-name> -n <namespace>

# Port forward MongoDB (run in separate terminal)
kubectl port-forward svc/mongodb 27017:27017 -n lwmecps

# Port forward Kubernetes API (run in separate terminal)
kubectl port-forward --address 0.0.0.0 -n kube-system svc/kubernetes 8443:443

# Port forward test application (if needed)
kubectl port-forward svc/mec-test-app 8080:80 -n default
```

## Support

If you encounter any issues:
1. Check the troubleshooting section
2. Review Minikube logs: `minikube logs`
3. Check Kubernetes logs: `kubectl logs`
4. Verify Kubernetes configuration
5. Create an issue on GitHub with detailed information about the problem 
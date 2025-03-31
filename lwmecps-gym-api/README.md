# LWMECPS GYM API

API service for LWMECPS GYM project that provides REST endpoints for training and managing ML models.

## Features

- Task management (create, start, pause, resume, stop)
- Training progress monitoring
- Results storage and retrieval
- Model reconciliation
- Weights & Biases integration
- MongoDB storage
- FastAPI-based RESTful API

## Installation

```bash
pip install lwmecps-gym-api
```

## Configuration

Set up environment variables:
```bash
export MONGODB_URI="mongodb://localhost:27017"
export MONGODB_DB="lwmecps_gym"
export WANDB_API_KEY="your-api-key"
export WANDB_PROJECT="lwmecps-gym"
export WANDB_ENTITY="your-entity"
```

## API Endpoints

### Training Tasks

- `POST /api/v1/training/tasks`: Create a new training task
  ```json
  {
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
  }
  ```

- `GET /api/v1/training/tasks`: List all training tasks
- `GET /api/v1/training/tasks/{task_id}`: Get task details
- `POST /api/v1/training/tasks/{task_id}/start`: Start training
- `POST /api/v1/training/tasks/{task_id}/pause`: Pause training
- `POST /api/v1/training/tasks/{task_id}/resume`: Resume training
- `POST /api/v1/training/tasks/{task_id}/stop`: Stop training
- `GET /api/v1/training/tasks/{task_id}/progress`: Get training progress
- `GET /api/v1/training/tasks/{task_id}/results`: Get training results
- `DELETE /api/v1/training/tasks/{task_id}`: Delete a training task

### Model Reconciliation

- `POST /api/v1/training/tasks/{task_id}/reconcile`: Run model reconciliation
  ```json
  {
    "sample_size": 1000
  }
  ```

## Docker Deployment

1. Build the Docker image:
```bash
docker build -t adeptvin4/lwmecps-gym-api:latest .
```

2. Run the container:
```bash
docker run -p 8000:8000 \
  -e MONGODB_URI="mongodb://localhost:27017" \
  -e MONGODB_DB="lwmecps_gym" \
  -e WANDB_API_KEY="your-api-key" \
  -e WANDB_PROJECT="lwmecps-gym" \
  -e WANDB_ENTITY="your-entity" \
  adeptvin4/lwmecps-gym-api:latest
```

## Development

1. Clone the repository
2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Run tests:
   ```bash
   pytest
   ```

## API Documentation

Once the server is running, you can access the API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## License

MIT License 
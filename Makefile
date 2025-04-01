.PHONY: install dev test clean

# Python version
PYTHON := python3
VENV := .venv
VENV_BIN := $(VENV)/bin

# Server configuration
HOST ?= 0.0.0.0
PORT ?= 8000
DEBUG ?= true

# MongoDB configuration
MONGODB_URI ?= mongodb://localhost:27017
MONGODB_DB ?= lwmecps_gym

# Weights & Biases configuration
WANDB_API_KEY ?= 
WANDB_PROJECT ?= lwmecps-gym
WANDB_ENTITY ?= 

# Install dependencies and setup development environment
install:
	$(PYTHON) -m venv $(VENV)
	$(VENV_BIN)/pip install --upgrade pip
	$(VENV_BIN)/pip install -r requirements.txt
	$(VENV_BIN)/pip install -e .

# Run development server
dev:
	WANDB_API_KEY=$(WANDB_API_KEY) \
	WANDB_PROJECT=$(WANDB_PROJECT) \
	WANDB_ENTITY=$(WANDB_ENTITY) \
	MONGODB_URI=$(MONGODB_URI) \
	MONGODB_DB=$(MONGODB_DB) \
	$(VENV_BIN)/uvicorn src.lwmecps_gym.main:app --reload --host $(HOST) --port $(PORT)

# Run tests
test:
	$(VENV_BIN)/pytest tests/

# Clean up
clean:
	rm -rf $(VENV)
	rm -rf *.egg-info
	rm -rf build/
	rm -rf dist/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Format code
format:
	$(VENV_BIN)/black src/ tests/
	$(VENV_BIN)/isort src/ tests/

# Lint code
lint:
	$(VENV_BIN)/flake8 src/ tests/
	$(VENV_BIN)/mypy src/ tests/

# Run all checks
check: format lint test

# Help command
help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies and setup development environment"
	@echo "  make dev       - Run development server"
	@echo "  make test      - Run tests"
	@echo "  make clean     - Clean up generated files"
	@echo "  make format    - Format code using black and isort"
	@echo "  make lint      - Run linters (flake8 and mypy)"
	@echo "  make check     - Run all checks (format, lint, test)"
	@echo ""
	@echo "Configuration variables:"
	@echo "  HOST           - Server host (default: 0.0.0.0)"
	@echo "  PORT           - Server port (default: 8000)"
	@echo "  DEBUG          - Debug mode (default: true)"
	@echo ""
	@echo "MongoDB configuration:"
	@echo "  MONGODB_URI    - MongoDB connection URI (default: mongodb://localhost:27017)"
	@echo "  MONGODB_DB     - MongoDB database name (default: lwmecps_gym)"
	@echo ""
	@echo "Weights & Biases configuration:"
	@echo "  WANDB_API_KEY  - Weights & Biases API key"
	@echo "  WANDB_PROJECT  - Weights & Biases project name (default: lwmecps-gym)"
	@echo "  WANDB_ENTITY   - Weights & Biases entity/username"
	@echo ""
	@echo "Example usage:"
	@echo "  make dev WANDB_API_KEY=your_key MONGODB_URI=mongodb://user:pass@host:port" 
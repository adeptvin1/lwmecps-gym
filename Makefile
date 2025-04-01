.PHONY: install dev test clean

# Python version
PYTHON := python3
VENV := .venv
VENV_BIN := $(VENV)/bin

# Install dependencies and setup development environment
install:
	$(PYTHON) -m venv $(VENV)
	$(VENV_BIN)/pip install --upgrade pip
	$(VENV_BIN)/pip install -r requirements.txt
	$(VENV_BIN)/pip install -e .

# Run development server
dev:
	$(VENV_BIN)/uvicorn src.lwmecps_gym.main:app --reload --host 0.0.0.0 --port 8000

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
	@echo "  make help      - Show this help message" 
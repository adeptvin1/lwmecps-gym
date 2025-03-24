# Use multi-stage build with cross-compilation support
FROM --platform=$BUILDPLATFORM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and setup files
COPY src/ src/
COPY setup.py .

# Install the package in development mode
RUN pip install -e .

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8010
ENV HOST=0.0.0.0

# Expose port
EXPOSE 8010

# Run the application
CMD ["uvicorn", "src.lwmecps_gym.main:app", "--host", "0.0.0.0", "--port", "8010"]

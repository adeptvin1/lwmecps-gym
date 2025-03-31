#!/bin/bash

# Build core package first
cd ../lwmecps-gym-core
docker build -t lwmecps-gym-core:latest .

# Build API package
cd ../lwmecps-gym-api
docker build -t adeptvin4/lwmecps-gym-api:latest .

# Push to Docker Hub
docker push adeptvin4/lwmecps-gym-api:latest

# Enable experimental features
export DOCKER_CLI_EXPERIMENTAL=enabled

# Create and use a new builder instance
docker buildx create --name multiarch-builder --use || true

# Build and push for both architectures with a single tag
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --tag adeptvin4/lwmecps-gym-api:latest \
  --push \
  -f Dockerfile \
  ..

# Build for local testing (will automatically use the correct architecture)
docker buildx build \
  --platform linux/arm64 \
  --tag adeptvin4/lwmecps-gym-api:latest \
  --load \
  -f Dockerfile \
  ..

echo "Build completed. Image available:"
echo "- adeptvin4/lwmecps-gym-api:latest (multi-arch)" 
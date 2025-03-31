#!/bin/bash

# Enable experimental features for Docker CLI
export DOCKER_CLI_EXPERIMENTAL=enabled

# Create a new builder instance
docker buildx create --name multiarch-builder --use || true

# Build and push for both architectures
cd .. && docker buildx build \
    --platform linux/amd64,linux/arm64 \
    -t adeptvin4/lwmecps-gym-api:latest \
    -f lwmecps-gym-api/Dockerfile \
    --push \
    .

# Build for local testing (arm64 only)
docker buildx build \
    --platform linux/arm64 \
    -t adeptvin4/lwmecps-gym-api:latest \
    -f lwmecps-gym-api/Dockerfile \
    --load \
    .

echo "Build completed. Image available:"
echo "- adeptvin4/lwmecps-gym-api:latest (multi-arch)" 
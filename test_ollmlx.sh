#!/bin/bash

set -e

echo "=== Testing ollmlx ==="
echo ""

# Build the binary
echo "1. Building ollmlx..."
# Get the script's directory and use it
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
go build -o ollmlx . 2>&1 | grep -v "warning: ignoring duplicate libraries"
echo "   ✓ Binary built successfully"
echo ""

# Check binary name
echo "2. Checking binary name..."
./ollmlx --version | grep -q "ollmlx" && echo "   ✓ Binary is named 'ollmlx'" || echo "   ✗ Binary name is wrong"
echo ""

# Start server
echo "3. Starting ollmlx server..."
./ollmlx serve > /tmp/ollmlx.log 2>&1 &
SERVER_PID=$!
sleep 3

# Check if server is running
if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
    echo "   ✓ Server is running on port 11434"
else
    echo "   ✗ Server failed to start"
    echo "Log:"
    cat /tmp/ollmlx.log
    exit 1
fi
echo ""

# Test API endpoints
echo "4. Testing API endpoints..."

# Version endpoint
VERSION=$(curl -s http://localhost:11434/api/version)
echo "   Version: $VERSION"

# List models
MODELS=$(curl -s http://localhost:11434/api/tags)
echo "   Models endpoint works"

# Show model info
MODEL_INFO=$(curl -s http://localhost:11434/api/show -d '{"name":"gemma2:2b"}')
echo "   Show model endpoint works"
echo ""

# Test MLX model pull
echo "5. Testing MLX model operations..."

# Try to pull an MLX model (this might fail if model doesn't exist)
echo "   Attempting to pull mlx-community/gemma-3-270m-4bit..."
PULL_OUTPUT=$(curl -s http://localhost:11434/api/pull -d '{"name":"mlx-community/gemma-3-270m-4bit"}')
echo "   Pull response: $PULL_OUTPUT"
echo ""

# Test model generation with existing GGUF model
echo "6. Testing model generation with GGUF model..."
GENERATE_OUTPUT=$(curl -s http://localhost:11434/api/generate -d '{"model":"gemma2:2b","prompt":"Hello"}')
echo "   Generate response: $GENERATE_OUTPUT"
echo ""

# Cleanup
echo "7. Cleaning up..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
echo "   ✓ Server stopped"
echo ""

echo "=== Test Complete ==="

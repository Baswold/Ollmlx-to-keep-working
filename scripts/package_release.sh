#!/bin/bash
set -e

# Configuration
APP_NAME="Ollmlx"
BINARY_NAME="ollmlx"
BUNDLE_ID="ai.ollmlx.ollmlx"
VERSION=$(git describe --tags --first-parent --abbrev=7 --long --dirty --always | sed -e "s/^v//g")
DIST_DIR="dist"
APP_BUNDLE="${DIST_DIR}/${APP_NAME}.app"
DMG_NAME="${APP_NAME}.dmg"

echo "Packaging ${APP_NAME} version ${VERSION}..."

# Ensure we're in the root directory
cd "$(dirname "$0")/.."

# 0. Clean
echo "Cleaning dist..."
rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}"

# 0. Build Frontend
echo "Building frontend..."
if command -v npm &> /dev/null; then
    cd app/ui/app
    npm install
    npm run build
    cd ../../..
else
    echo "Error: npm is not installed. Please install Node.js and npm to build the frontend."
    exit 1
fi

# 1. Build binaries
echo "Building binaries..."
# Main server CLI
GOOS=darwin CGO_ENABLED=1 go build -o "${DIST_DIR}/${BINARY_NAME}" .
# MLX Runner
GOOS=darwin CGO_ENABLED=1 go build -o "${DIST_DIR}/ollama-runner" ./cmd/runner
# GUI App Wrapper
GOOS=darwin CGO_ENABLED=1 go build -o "${DIST_DIR}/${APP_NAME}_bin" ./app/cmd/app

# 2. Assemble App Bundle
echo "Creating App Bundle..."
# Copy template
mkdir -p "${APP_BUNDLE}"
cp -R "app/darwin/Ollama.app/" "${APP_BUNDLE}/"

# MacOS dir
mkdir -p "${APP_BUNDLE}/Contents/MacOS"
mv "${DIST_DIR}/${APP_NAME}_bin" "${APP_BUNDLE}/Contents/MacOS/${APP_NAME}"

# Resources dir
mkdir -p "${APP_BUNDLE}/Contents/Resources"
# Install binaries
cp "${DIST_DIR}/${BINARY_NAME}" "${APP_BUNDLE}/Contents/Resources/${BINARY_NAME}"
chmod +x "${APP_BUNDLE}/Contents/Resources/${BINARY_NAME}"
cp "${DIST_DIR}/ollama-runner" "${APP_BUNDLE}/Contents/Resources/ollama-runner"
chmod +x "${APP_BUNDLE}/Contents/Resources/ollama-runner"

# Install Backend Source (for internal venv creation)
echo "Bundling MLX Backend..."
mkdir -p "${APP_BUNDLE}/Contents/Resources/mlx_backend"
cp -r mlx_backend/* "${APP_BUNDLE}/Contents/Resources/mlx_backend/"
# Clean up potential pycache or incidental files
find "${APP_BUNDLE}/Contents/Resources/mlx_backend" -name "__pycache__" -exec rm -rf {} +
find "${APP_BUNDLE}/Contents/Resources/mlx_backend" -name "*.pyc" -delete

# 3. Create DMG
echo "Creating DMG..."
# We use the existing create-dmg script
# We need to ensure the script is executable
chmod +x scripts/create-dmg.sh

# Create DMG
./scripts/create-dmg.sh \
  --volname "${APP_NAME}" \
  --volicon "app/darwin/Ollama.app/Contents/Resources/icon.icns" \
  --window-pos 200 120 \
  --window-size 800 400 \
  --icon-size 100 \
  --icon "${APP_NAME}.app" 200 190 \
  --hide-extension "${APP_NAME}.app" \
  --app-drop-link 600 185 \
  "${DIST_DIR}/${DMG_NAME}" \
  "${APP_BUNDLE}"

echo "Done! Package created at ${DIST_DIR}/${DMG_NAME}"

#!/usr/bin/env bash
set -euo pipefail

# ollmlx Installer
# 1. Checks prerequisites (Go, Python 3.10+)
# 2. Sets up a dedicated Python virtual environment (~/.ollmlx/venv)
# 3. Installs MLX dependencies
# 4. Builds the Go binary
# 5. Optional: Installs to system path

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLLMLX_DIR="$HOME/.ollmlx"
VENV_DIR="$OLLMLX_DIR/venv"

# --- Colors ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log() { echo -e "${BLUE}[ollmlx]${NC} $1"; }
ok() { echo -e "${GREEN}[ok]${NC} $1"; }
err() { echo -e "${RED}[error]${NC} $1" >&2; exit 1; }

# --- 1. Prerequisites ---
log "Checking prerequisites..."

# Check Go
if ! command -v go >/dev/null; then
  err "Go is required but not found. Please install Go 1.21+"
fi
# Simple go version check (heuristic)
GO_VER=$(go version | awk '{print $3}' | sed 's/go//')
ok "Found Go $GO_VER"

# Check Python 3.10+
if ! command -v python3 >/dev/null; then
  err "Python 3 is required but not found."
fi

PY_CHECK=$(python3 -c "import sys; print(1) if sys.version_info >= (3, 10) else print(0)")
if [ "$PY_CHECK" != "1" ]; then
  err "Python 3.10 or higher is required."
fi
ok "Found Python $(python3 --version)"

# --- 2. Virtual Environment ---
log "Setting up Python environment..."
if [ ! -d "$VENV_DIR" ]; then
  log "Creating virtual environment at $VENV_DIR"
  mkdir -p "$OLLMLX_DIR"
  python3 -m venv "$VENV_DIR"
else
  ok "Using existing virtual environment at $VENV_DIR"
fi

# Activate venv for dependency installation
source "$VENV_DIR/bin/activate"

# --- 3. Dependencies ---
REQ_FILE="$ROOT/mlx_backend/requirements.txt"
if [ -f "$REQ_FILE" ]; then
  log "Installing/Updating Python dependencies..."
  pip install --upgrade pip -q
  pip install -r "$REQ_FILE" -q
  ok "Python dependencies installed"
else
  err "requirements.txt not found at $REQ_FILE"
fi

# --- 4. Build ---
log "Building ollmlx binary..."
cd "$ROOT"
go build -o ollmlx .
ok "Build complete: ./ollmlx"

log "Building ollama-runner binary..."
go build -o ollama-runner ./cmd/runner
ok "Build complete: ./ollama-runner"

# --- 5. Installation ---
if [[ $# -gt 0 && "$1" == "--install" ]]; then
  DEST_DIR="/usr/local/bin"
  DEST="$DEST_DIR/ollmlx"
  RUNNER_DEST="$DEST_DIR/ollama-runner"
  
  log "Installing to $DEST_DIR (may require password)"
  sudo cp ollmlx "$DEST"
  sudo cp ollama-runner "$RUNNER_DEST"
  
  ok "Installed $DEST"
  ok "Installed $RUNNER_DEST"
  
  echo ""
  echo -e "${GREEN}Success! You can now run:${NC} ollmlx serve"
else
  echo ""
  echo "--------------------------------------------------"
  echo -e "Build successful!"
  echo -e "  Main binary:   ${GREEN}$ROOT/ollmlx${NC}"
  echo -e "  Runner binary: ${GREEN}$ROOT/ollama-runner${NC}"
  echo ""
  echo "To install system-wide, run:"
  echo -e "  ${BLUE}sudo cp ollmlx ollama-runner /usr/local/bin/${NC}"
  echo ""
  echo "Or run this script with --install:"
  echo -e "  ${BLUE}./scripts/install_ollmlx.sh --install${NC}"
  echo "--------------------------------------------------"
fi


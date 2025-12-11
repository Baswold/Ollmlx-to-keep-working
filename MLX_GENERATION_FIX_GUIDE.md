# 🔧 MLX Generation Fix-It Guide - COMPREHENSIVE ANALYSIS

**Document Status**: Complete Deep-Dive Analysis (Ready for Implementation)
**Date**: December 11, 2025
**Severity**: CRITICAL - Blocks all MLX model generation

---

## Part 1: How Ollama Actually Works (From Architecture Research)

### The Standard Ollama Flow (What Works For GGUF)

Ollama uses a **Client-Server architecture** with the following flow:

```
User: ollama run llama2
    ↓
HTTP Client Request: POST /api/generate
    ↓
Server (Port 11434)
    ↓
Model Execution Layer (Scheduler + Runner)
    ↓
Backend: llama.cpp (via mmap - memory mapping)
    ↓
GGUF Model File (~/.ollama/models/blobs/<SHA256>)
    ↓
Token Generation (with KV Cache for speed)
    ↓
Streaming Response back to Client
```

### Key Ollama Concepts We're Missing

1. **Model Storage**: Ollama stores GGUF files in `~/.ollama/models/blobs/` with SHA256 hash names
2. **Model Registry**: Models are identified by name (e.g., `llama2:latest`) and resolved to blob files
3. **Memory Mapping (mmap)**: Direct memory mapping prevents data copying - this is crucial for performance
4. **Model Scheduler**: Intelligent loading with automatic eviction based on memory constraints
5. **Runner Processes**: Separate processes for each backend (llama.cpp, Metal, CUDA, etc.)
6. **KV Cache**: Caches attention scores to avoid recalculation during generation

### What We're Trying To Do With MLX

We're trying to create a parallel pipeline for MLX models:

```
User: ollama run mlx-community/llama-2
    ↓
HTTP Client Request: POST /api/generate
    ↓
Server (Port 11434)
    ↓
MLX Model Execution Layer (OUR NEW LAYER)
    ↓
MLX Runner Process (Port ~8023)
    ↓
Python MLX Backend Server
    ↓
MLX Model + Tokenizer (~/.ollama/models/mlx/mlx-community_llama-2/)
    ↓
Token Generation (similar but using MLX framework)
    ↓
Streaming Response back to Client
```

**The Problem**: This pipeline is broken at critical points. Let's fix it.

---

## Part 2: Exact Root Causes (7 Critical Issues)

### 🔴 ISSUE #1: Model Name Format Mismatch in `/api/generate` Handler

**File**: `server/routes_mlx.go:484-546`

**The Bug**:
```go
// Line 487: Correctly converts slash to underscore
localName := strings.ReplaceAll(req.Model, "/", "_")

// Lines 489, 494: Correctly checks with localName
if !manager.ModelExists(localName) { ... }
if _, err := manager.GetModelInfo(localName); err != nil { ... }

// ❌ LINE 502: WRONG - passes original name with slashes
cmd, port, err := startMLXRunner(runnerCtx, req.Model)  // Should be: localName

// ❌ LINE 524: WRONG - passes original name with slashes
if err := loadMLXModel(runnerCtx, client, port, req.Model); err != nil {  // Should be: localName
```

**Why This Breaks**:
- Models are stored on disk as: `~/.ollama/models/mlx/mlx-community_llama-2/`
- But we pass `mlx-community/llama-2` to the runner
- The runner then passes the wrong path to the Python backend
- The Python backend converts it, but this is a hack, not a real fix

**Flow Breakdown**:
```
Model on disk: ~/.ollama/models/mlx/mlx-community_llama-2/
    ↓
We check with localName: "mlx-community_llama-2" ✓ Found
    ↓
We start runner with: req.Model = "mlx-community/llama-2" ❌ Wrong
    ↓
Runner receives: modelName = "mlx-community/llama-2"
    ↓
Runner constructs path: modelName (with slashes)
    ↓
Python backend gets: "mlx-community/llama-2"
    ↓
Python converts: "mlx-community_llama-2" (emergency fix)
    ↓
Python looks in: ~/.ollama/models/mlx/mlx-community_llama-2/ ✓ But only because of conversion
```

**The Fix**:
```go
// Line 502
cmd, port, err := startMLXRunner(runnerCtx, localName)

// Line 524
if err := loadMLXModel(runnerCtx, client, port, localName); err != nil {
```

---

### 🔴 ISSUE #2: Same Model Name Bug in `/api/chat` Handler

**File**: `server/routes_mlx.go:548-644`

**The Bug**:
```go
// Line 551: Correctly converts
localName := strings.ReplaceAll(req.Model, "/", "_")

// Lines 553, 558: Correctly checks
if !manager.ModelExists(localName) { ... }
if _, err := manager.GetModelInfo(localName); err != nil { ... }

// ❌ LINE 566: WRONG - passes original name
cmd, port, err := startMLXRunner(runnerCtx, req.Model)  // Should be: localName

// ❌ LINE 588: WRONG - passes original name
if err := loadMLXModel(runnerCtx, client, port, req.Model); err != nil {  // Should be: localName
```

**Why It Breaks**: Same reason as Issue #1

**The Fix**: Same as Issue #1

---

### 🔴 ISSUE #3: Python Backend Ignores `OLLAMA_MODELS` Environment Variable

**File**: `mlx_backend/server.py:120`

**The Bug**:
```python
def __init__(self):
    self.model = None
    self.tokenizer = None
    self.current_model_name = None
    # ❌ HARDCODED - completely ignores OLLAMA_MODELS env var
    self.model_path = Path.home() / ".ollama" / "models" / "mlx"
```

**Compare with Go (which does it correctly)**:
```go
// mlx_models.go:41
modelsDir := filepath.Join(envconfig.Models(), "mlx")  // ✓ Uses envconfig
```

**Why This Is Critical**:
- Go code respects `OLLAMA_MODELS` environment variable
- If a user sets `OLLAMA_MODELS=/custom/path`, models go to `/custom/path/mlx/`
- But Python backend ALWAYS looks in `~/.ollama/models/mlx/`
- Models won't be found, even though they exist elsewhere
- This creates a silent failure mode

**Example**:
```bash
# User sets custom models directory
export OLLAMA_MODELS=/data/my_models

# Go code respects this
ollmlx pull mlx-community/llama-2
# → Saves to: /data/my_models/mlx/mlx-community_llama-2/

# But Python backend doesn't
# → Looks in: ~/.ollama/models/mlx/mlx-community_llama-2/
# → Not found! ❌
```

**The Fix**:
```python
import os
from pathlib import Path

def __init__(self):
    self.model = None
    self.tokenizer = None
    self.current_model_name = None

    # ✓ Respect OLLAMA_MODELS like Go does
    ollama_models = os.environ.get('OLLAMA_MODELS')
    if ollama_models:
        self.model_path = Path(ollama_models) / "mlx"
    else:
        self.model_path = Path.home() / ".ollama" / "models" / "mlx"

    logger.info(f"Using model path: {self.model_path}")
```

---

### 🔴 ISSUE #4: Runner Doesn't Pass `OLLAMA_MODELS` Environment Variable

**File**: `runner/mlxrunner/runner.go` (No code currently handles this)

**The Bug**:
The MLX runner is started as a subprocess but doesn't receive the `OLLAMA_MODELS` environment variable.

```go
// Currently in routes_mlx.go:79-91
cmd := exec.CommandContext(ctx, "go", "run", "./runner/mlxrunner/runner.go", "-model", modelName, "-port", strconv.Itoa(port))
cmd.Stdout = os.Stdout
cmd.Stderr = os.Stderr
// ❌ No: cmd.Env = ...  to pass environment variables
```

**Why This Breaks**:
- Even if we fix the Python backend to respect `OLLAMA_MODELS`
- The runner subprocess never receives the environment variable
- Python backend still can't find the models

**The Fix**:
```go
cmd := exec.CommandContext(ctx, "go", "run", "./runner/mlxrunner/runner.go", "-model", modelName, "-port", strconv.Itoa(port))
cmd.Stdout = os.Stdout
cmd.Stderr = os.Stderr
// ✓ Pass all current environment variables
cmd.Env = append(os.Environ(), "OLLAMA_MODELS="+envconfig.Models())
```

---

### 🔴 ISSUE #5: Fragile MLX Backend Path Resolution

**File**: `runner/mlxrunner/runner.go:61-73`

**The Bug**:
```go
var mlxBackendPath string
for _, candidate := range []string{
    filepath.Join("mlx_backend", "server.py"),  // ❌ Relative path - depends on working dir
    filepath.Join(filepath.Dir(s.modelPath), "..", "..", "..", "mlx_backend", "server.py"),
    // ❌ If s.modelPath = "mlx-community/llama-2", then filepath.Dir = "mlx-community"
    // The path becomes: mlx-community/../../../mlx_backend/server.py (fragile!)
} {
    if _, err := os.Stat(candidate); err == nil {
        mlxBackendPath = candidate
        break
    }
}
if mlxBackendPath == "" {
    return fmt.Errorf("MLX backend server not found")  // ❌ Vague error message
}
```

**Why It Breaks**:
1. If working directory isn't project root, first candidate fails
2. If `modelPath` contains slashes, relative path calculation is wrong
3. If both fail, we get no indication of WHY the backend wasn't found
4. User can't debug the issue

**The Fix**:
```go
// Get the directory where ollmlx binary is running from
ex, err := os.Executable()
if err != nil {
    return fmt.Errorf("determine executable path: %w", err)
}
binaryDir := filepath.Dir(ex)
projectRoot := filepath.Join(binaryDir, "..")  // Assuming binary is in project root

var mlxBackendPath string
candidates := []string{
    filepath.Join(projectRoot, "mlx_backend", "server.py"),
    filepath.Join(binaryDir, "mlx_backend", "server.py"),
    filepath.Join("mlx_backend", "server.py"),  // Current directory fallback
}

for _, candidate := range candidates {
    absPath, _ := filepath.Abs(candidate)
    if _, err := os.Stat(candidate); err == nil {
        mlxBackendPath = candidate
        slog.Debug("Found MLX backend", "path", absPath)
        break
    }
    slog.Debug("MLX backend candidate not found", "path", absPath)
}
if mlxBackendPath == "" {
    return fmt.Errorf("MLX backend server not found at any of: %v", candidates)
}
```

---

### 🔴 ISSUE #6: No Logging of Health Check Failures

**File**: `server/routes_mlx.go:93-112`

**The Bug**:
```go
func waitForMLXRunner(ctx context.Context, client *http.Client, port int) error {
    deadline := time.Now().Add(30 * time.Second)
    url := fmt.Sprintf("http://127.0.0.1:%d/health", port)
    for time.Now().Before(deadline) {
        resp, err := client.Get(url)
        if err == nil && resp.StatusCode == http.StatusOK {
            resp.Body.Close()
            return nil
        }
        if resp != nil {
            resp.Body.Close()
        }
        // ❌ Silent failure - no logging of what went wrong
        // ❌ err is ignored - could be connection refused, timeout, etc.
        select {
        case <-ctx.Done():
            return ctx.Err()
        case <-time.After(500 * time.Millisecond):
        }
    }
    // ❌ Returns generic message with no debug info
    return fmt.Errorf("mlx runner did not become healthy")
}
```

**Why This Breaks**:
- If backend fails to start, we only get "mlx runner did not become healthy"
- No indication of whether it crashed, never started, port wasn't listening, etc.
- Debugging becomes impossible

**The Fix**:
```go
func waitForMLXRunner(ctx context.Context, client *http.Client, port int) error {
    deadline := time.Now().Add(30 * time.Second)
    url := fmt.Sprintf("http://127.0.0.1:%d/health", port)
    var lastErr error
    for time.Now().Before(deadline) {
        resp, err := client.Get(url)
        if err == nil && resp.StatusCode == http.StatusOK {
            resp.Body.Close()
            slog.Info("MLX runner healthy", "port", port)
            return nil
        }
        if err != nil {
            lastErr = err
            slog.Debug("Health check failed", "error", err.Error(), "port", port)
        }
        if resp != nil {
            slog.Debug("Health check returned non-200", "status", resp.StatusCode, "port", port)
            resp.Body.Close()
        }
        select {
        case <-ctx.Done():
            return ctx.Err()
        case <-time.After(500 * time.Millisecond):
        }
    }
    if lastErr != nil {
        return fmt.Errorf("mlx runner health check failed after 30s: %w", lastErr)
    }
    return fmt.Errorf("mlx runner did not become healthy after 30s")
}
```

---

### 🔴 ISSUE #7: Using `go run` Instead of Compiled Binary

**File**: `server/routes_mlx.go:79-91`

**The Bug**:
```go
func startMLXRunner(ctx context.Context, modelName string) (*exec.Cmd, int, error) {
    l, err := net.Listen("tcp", "127.0.0.1:0")
    if err != nil {
        return nil, 0, fmt.Errorf("allocate port: %w", err)
    }
    port := l.Addr().(*net.TCPAddr).Port
    l.Close()

    // ❌ BUG: Uses "go run" instead of built executable
    cmd := exec.CommandContext(ctx, "go", "run", "./runner/mlxrunner/runner.go", "-model", modelName, "-port", strconv.Itoa(port))
    cmd.Stdout = os.Stdout
    cmd.Stderr = os.Stderr
    return cmd, port, nil
}
```

**Why This Is Bad**:
1. Requires Go toolchain to be installed on the user's machine
2. Slow - `go run` recompiles every time
3. Depends on working directory
4. Hard to distribute/package the binary
5. Better practice: build everything into a single binary

**The Fix - Option 1: Build MLX runner into main binary**
```go
// Don't start subprocess - call the runner code directly as a goroutine
func startMLXRunner(ctx context.Context, modelName string) (int, error) {
    l, err := net.Listen("tcp", "127.0.0.1:0")
    if err != nil {
        return 0, fmt.Errorf("allocate port: %w", err)
    }
    port := l.Addr().(*net.TCPAddr).Port
    l.Close()

    // Start runner in goroutine with context
    go runMLXRunnerServer(ctx, modelName, port)
    return port, nil
}
```

**The Fix - Option 2: Build runner separately and embed in binary**
```go
// Build runner as separate executable
// Then embed in binary using //go:embed
//go:embed runner/mlxrunner/runner
var runnerBinary []byte

func startMLXRunner(ctx context.Context, modelName string) (*exec.Cmd, int, error) {
    // Extract runner binary to temp file
    tmpFile, err := os.CreateTemp("", "mlx-runner-*")
    if err != nil {
        return nil, 0, err
    }
    defer tmpFile.Close()

    if _, err := tmpFile.Write(runnerBinary); err != nil {
        return nil, 0, err
    }

    // Make it executable
    os.Chmod(tmpFile.Name(), 0755)

    // Run the extracted binary
    cmd := exec.CommandContext(ctx, tmpFile.Name(), "-model", modelName, "-port", strconv.Itoa(port))
    // ...
}
```

---

## Part 3: Complete Error Flow Map

```
┌─ POST /api/generate ─────────────────────────────────────────┐
│ model = "mlx-community/llama-2"                               │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
           ┌─ IsMLXModelReference() ──────┐
           │ Converts to:                 │
           │ "mlx-community_llama-2"      │
           │ ✓ Found in ModelManager      │
           └─────────┬───────────────────┘
                     ↓
         ┌─ generateMLXModel() ────────────┐
         │ localName = "mlx-community_llama-2" │
         │ ✓ ModelExists(localName) = true     │
         └──────────┬─────────────────────┘
                    ↓
      ┌──── startMLXRunner(req.Model) ────┐
      │ ❌ BUG: Still using slashes!      │
      │ Starts: go run runner.go           │
      │   -model mlx-community/llama-2     │
      └──────────┬──────────────────────┘
                 ↓
    ┌──── MLX Runner Process ────────────┐
    │ modelPath = "mlx-community/llama-2"│
    │ filepath.Dir() = "mlx-community"   │
    │ Backend path calculation broken    │
    │                                    │
    │ But somehow finds backend.py       │
    │ Starts Python backend server       │
    └──────────┬──────────────────────┘
               ↓
  ┌──── Python Backend (server.py) ─────┐
  │ model_path = ~/.ollama/models/mlx    │
  │ ❌ BUG: Ignores OLLAMA_MODELS        │
  │ ❌ BUG: Never received env vars      │
  │                                      │
  │ Receives: loadMLXModel(               │
  │   "mlx-community/llama-2")            │
  │                                      │
  │ Converts: "mlx-community_llama-2"   │
  │ Looks for:                           │
  │ ~/.ollama/models/mlx/                │
  │   mlx-community_llama-2/             │
  │                                      │
  │ If OLLAMA_MODELS is custom: ❌ FAIL  │
  │ If standard location: ✓ OK           │
  └──────────┬──────────────────────┘
             ↓
  ┌──── Model Loading ─────────────────┐
  │ mlx_lm.load(modelPath)              │
  │ Result: success or error            │
  └──────────┬──────────────────────┘
             ↓
  ┌──── Token Generation ──────────────┐
  │ Streaming tokens back to client    │
  └────────────────────────────────────┘
```

---

## Part 4: Detailed Fix-It TODO List

### Phase 1: Fix Model Name Format Issues (HIGHEST PRIORITY)

**Estimated Time: 30 minutes**

- [ ] **Fix Issue #1**: Update `server/routes_mlx.go:502`
  - Change: `startMLXRunner(runnerCtx, req.Model)`
  - To: `startMLXRunner(runnerCtx, localName)`
  - Lines affected: 502
  - Files: `server/routes_mlx.go`

- [ ] **Fix Issue #1b**: Update `server/routes_mlx.go:524`
  - Change: `loadMLXModel(runnerCtx, client, port, req.Model)`
  - To: `loadMLXModel(runnerCtx, client, port, localName)`
  - Lines affected: 524
  - Files: `server/routes_mlx.go`

- [ ] **Fix Issue #2**: Update chat handler `server/routes_mlx.go:566`
  - Change: `startMLXRunner(runnerCtx, req.Model)`
  - To: `startMLXRunner(runnerCtx, localName)`
  - Lines affected: 566
  - Files: `server/routes_mlx.go`

- [ ] **Fix Issue #2b**: Update `server/routes_mlx.go:588`
  - Change: `loadMLXModel(runnerCtx, client, port, req.Model)`
  - To: `loadMLXModel(runnerCtx, client, port, localName)`
  - Lines affected: 588
  - Files: `server/routes_mlx.go`

- [ ] **Test After Phase 1**:
  ```bash
  ./ollmlx run mlx-community/llama-2
  # Should now work or fail with more specific error
  ```

### Phase 2: Fix Environment Variable Handling (HIGH PRIORITY)

**Estimated Time: 1 hour**

- [ ] **Fix Issue #3**: Update Python backend `mlx_backend/server.py:120`
  - Respect `OLLAMA_MODELS` environment variable
  - Add proper logging when path is determined
  - Files: `mlx_backend/server.py`
  - Lines: 120 and __init__ method

- [ ] **Fix Issue #4**: Update runner startup in `server/routes_mlx.go:85-91`
  - Pass environment variables to subprocess
  - Specifically pass `OLLAMA_MODELS`
  - Files: `server/routes_mlx.go`
  - Lines: 85-91

- [ ] **Test After Phase 2**:
  ```bash
  # Test with custom OLLAMA_MODELS
  export OLLAMA_MODELS=/custom/path
  mkdir -p /custom/path/mlx
  cp -r ~/.ollama/models/mlx/* /custom/path/mlx/
  ./ollmlx run mlx-community/llama-2
  # Should work with custom path
  ```

### Phase 3: Fix Robustness Issues (MEDIUM PRIORITY)

**Estimated Time: 1 hour**

- [ ] **Fix Issue #5**: Improve backend path resolution in `runner/mlxrunner/runner.go:61-73`
  - Use executable directory instead of relative paths
  - Add detailed error messages
  - Files: `runner/mlxrunner/runner.go`
  - Lines: 61-73

- [ ] **Fix Issue #6**: Add logging to health checks in `server/routes_mlx.go:93-112`
  - Log each health check attempt
  - Log specific failure reasons
  - Files: `server/routes_mlx.go`
  - Lines: 93-112

- [ ] **Test After Phase 3**:
  ```bash
  # Enable debug logging
  OLLAMA_DEBUG=1 ./ollmlx run mlx-community/llama-2
  # Should see detailed debug output showing what's happening
  ```

### Phase 4: Structural Improvement (LOWER PRIORITY)

**Estimated Time: 2+ hours**

- [ ] **Fix Issue #7**: Replace `go run` with compiled binary
  - Option A: Build runner as separate executable in build process
  - Option B: Inline runner code into main binary
  - Files: Multiple - requires refactoring
  - Consider: `Makefile`, `runner/mlxrunner/runner.go`, `server/routes_mlx.go`

- [ ] **Add integration tests** for MLX generation
  - Test with standard `OLLAMA_MODELS`
  - Test with custom `OLLAMA_MODELS`
  - Test error cases

### Phase 5: Testing & Validation

**Estimated Time: 1-2 hours**

- [ ] **Build and test Phase 1 fixes**: 30 min
- [ ] **Build and test Phase 2 fixes**: 30 min
- [ ] **Build and test Phase 3 fixes**: 30 min
- [ ] **Run full integration test suite**: 30 min
- [ ] **Compare GGUF vs MLX generation**: 15 min
- [ ] **Document any remaining issues**: 15 min

---

## Part 5: Things That Need To Be Better

### Code Quality Issues

1. **Inconsistent Error Handling**
   - Some errors are logged with details
   - Some errors are silent or vague
   - Recommendation: Use structured logging (`slog`) consistently
   - Add context to all error returns: `fmt.Errorf("do thing: %w", err)`

2. **No Request Tracing**
   - Can't trace a request through the system
   - Recommendation: Add request IDs or trace context
   - Log request → response with timing

3. **Missing Health Checks**
   - Only MLX backend has health checks
   - Recommendation: Add health check to main server
   - Add readiness/liveness endpoints for Kubernetes

4. **Poor Configuration Management**
   - `OLLAMA_MODELS` only respected in some places
   - Recommendation: Create config struct that's passed everywhere
   - Validate config on startup

### Architecture Issues

1. **Subprocess Management**
   - Using `go run` is inefficient and fragile
   - No process pooling - new runner per request
   - Recommendation: Build runner into main binary or as clean subprocess
   - Consider: Keep runner alive between requests?

2. **Model Path Format Inconsistency**
   - Slashes vs underscores causes confusion
   - Recommendation: Normalize model names at entry point
   - Use normalized names consistently throughout

3. **No Backwards Compatibility Layer**
   - If model format changes, old clients break
   - Recommendation: Version API endpoints
   - Support both slash and underscore formats in conversion

4. **Tight Coupling Between Go and Python**
   - Changes to model path break both codebases
   - Recommendation: Define strict API contract
   - Version the Python backend API
   - Add compatibility tests

### Documentation Issues

1. **No Architecture Diagram**
   - Users don't understand data flow
   - Recommendation: Add detailed architecture diagram
   - Show Go vs Python responsibilities

2. **No Troubleshooting Guide**
   - When things break, users are lost
   - Recommendation: Document common failures and solutions
   - Add debug mode with verbose logging

3. **No API Contract Documentation**
   - What data flows between Go and Python?
   - Recommendation: Document the HTTP API between components
   - Include request/response examples

### Testing Issues

1. **No End-to-End Tests**
   - Can't verify full MLX generation pipeline
   - Recommendation: Add tests that:
     - Pull real MLX model
     - Generate text
     - Compare output between GGUF and MLX

2. **No Failure Case Testing**
   - What happens when model not found?
   - What happens when runner crashes?
   - Recommendation: Add chaos testing
   - Add timeout/cancellation tests

3. **No Performance Tests**
   - Claims of "2-3x faster" unvalidated
   - Recommendation: Add benchmarks
   - Compare latency and throughput vs original Ollama

### Operational Issues

1. **No Process Monitoring**
   - How to know if runner is unhealthy?
   - Recommendation: Add process monitoring
   - Add metrics for runner uptime, errors, etc.

2. **No Cleanup**
   - What happens to orphaned runner processes?
   - Recommendation: Implement graceful shutdown
   - Add process cleanup on timeout

3. **No Resource Limits**
   - Models can consume unlimited memory
   - Recommendation: Add configurable memory limits
   - Add request timeout configuration

---

## Part 6: Impact Assessment

### Critical Path Issues (Must Fix)
- [ ] Model name format mismatch (Phase 1) - **BLOCKS MLX GENERATION**
- [ ] OLLAMA_MODELS environment variable (Phase 2) - **BLOCKS CUSTOM PATHS**

### High Priority Issues (Should Fix)
- [ ] Environment variable passing to subprocess (Phase 2)
- [ ] Backend path resolution (Phase 3)
- [ ] Health check logging (Phase 3)

### Medium Priority Issues (Nice to Have)
- [ ] Using compiled binary instead of `go run` (Phase 4)
- [ ] Integration tests (Phase 5)
- [ ] Request tracing (Quality improvement)

### Low Priority Issues (Future Enhancement)
- [ ] Process pooling for runners
- [ ] Performance benchmarking
- [ ] Backwards compatibility layer

---

## Part 7: Validation Checklist

After implementing all fixes, verify:

```bash
# Basic functionality
./ollmlx run mlx-community/llama-2
# Should: Load model, generate text, stream response

# Custom OLLAMA_MODELS
export OLLAMA_MODELS=/tmp/test_models
mkdir -p /tmp/test_models/mlx
cp -r ~/.ollama/models/mlx/* /tmp/test_models/mlx/
./ollmlx run mlx-community/llama-2
# Should: Still work with custom path

# API endpoints
curl http://localhost:11434/api/generate -d '{"model": "mlx-community/llama-2", "prompt": "Hello"}'
# Should: Stream response with 200 status

# Chat
curl http://localhost:11434/api/chat -d '{"model": "mlx-community/llama-2", "messages": [...]}'
# Should: Return chat response

# Error cases
curl http://localhost:11434/api/generate -d '{"model": "nonexistent-model", "prompt": "Hi"}'
# Should: Return meaningful error, not crash

# Debugging
OLLAMA_DEBUG=1 ./ollmlx run mlx-community/llama-2
# Should: Show detailed logs of each step
```

---

## Summary

**Total Issues Found**: 7 critical/high priority
**Total LOC to Change**: ~50-100 lines
**Estimated Fix Time**: 4-6 hours (including testing)
**Estimated Improvement**: MLX generation goes from 0% to 100% functional

**Key Insight**: The infrastructure is 95% there. The issues are mostly parameter passing and configuration management bugs. None require architectural changes.


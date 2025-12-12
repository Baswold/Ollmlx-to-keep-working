//go:build mlxfixture

package mlxfixture

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/runner/mlxrunner"
)

type stubBackendOptions struct {
	allowLoad         bool
	expectModel       bool
	delay             time.Duration
	errorOnCompletion bool
}

type stubBackend struct {
	server *httptest.Server
	opts   stubBackendOptions
	loaded sync.WaitGroup
}

func newStubBackend(t *testing.T, opts stubBackendOptions) *stubBackend {
	t.Helper()

	backend := &stubBackend{opts: opts}
	backend.loaded.Add(1)

	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "ok")
	})
	mux.HandleFunc("/load", func(w http.ResponseWriter, r *http.Request) {
		if !backend.opts.allowLoad {
			http.Error(w, "blocked", http.StatusBadRequest)
			return
		}

		var req map[string]string
		_ = json.NewDecoder(r.Body).Decode(&req)
		if backend.opts.expectModel {
			if _, err := os.Stat(req["model"]); err != nil {
				http.Error(w, "missing model", http.StatusNotFound)
				return
			}
		}

		backend.loaded.Done()
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"status":"ok"}`)
	})
	mux.HandleFunc("/completion", func(w http.ResponseWriter, r *http.Request) {
		backend.loaded.Wait()
		if backend.opts.delay > 0 {
			time.Sleep(backend.opts.delay)
		}
		if backend.opts.errorOnCompletion {
			http.Error(w, "backend failure", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/x-ndjson")
		enc := json.NewEncoder(w)
		_ = enc.Encode(map[string]any{"content": "stub response", "done": false, "model": r.URL.Query().Get("model")})
		_ = enc.Encode(map[string]any{"done": true})
	})

	server := httptest.NewServer(mux)
	backend.server = server
	return backend
}

func (s *stubBackend) Close() { s.server.Close() }

func backendPort(t *testing.T, srv *httptest.Server) int {
	t.Helper()

	addr := srv.Listener.Addr().(*net.TCPAddr)
	return addr.Port
}

func startRunner(t *testing.T, modelPath string, backend *httptest.Server) (string, func()) {
	t.Helper()

	runnerServer := &mlxrunner.Server{
		modelPath: modelPath,
		mlxPort:   backendPort(t, backend),
		mlxClient: backend.Client(),
	}
	runnerServer.ready.Add(1)
	runnerServer.cond = sync.NewCond(&runnerServer.mu)

	mux := http.NewServeMux()
	mux.HandleFunc("POST /load", runnerServer.load)
	mux.HandleFunc("/completion", runnerServer.completion)
	mux.HandleFunc("/health", runnerServer.health)

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}

	go http.Serve(listener, mux)

	return "http://" + listener.Addr().String(), func() { listener.Close() }
}

func serveFixtureModel(t *testing.T, modelID string) *httptest.Server {
	t.Helper()

	fixtureDir := filepath.Join("integration", "testdata", "mlx-fixture")
	handler := http.NewServeMux()
	handler.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		// Expected path: /<modelID>/resolve/main/<filename>
		parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/"), "/")
		if len(parts) < 4 {
			http.NotFound(w, r)
			return
		}
		filename := parts[len(parts)-1]
		data, err := os.ReadFile(filepath.Join(fixtureDir, filename))
		if err != nil {
			http.NotFound(w, r)
			return
		}
		w.Write(data)
	})

	return httptest.NewServer(handler)
}

func TestMLXFixturePullAndGenerate(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	fixtureModel := "local/fixture"
	server := serveFixtureModel(t, fixtureModel)
	defer server.Close()

	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("OLLAMA_MLX_BASE_URL", server.URL)

	manager := llm.NewMLXModelManager()
	if err := manager.DownloadMLXModel(ctx, fixtureModel, nil); err != nil {
		t.Fatalf("failed to download fixture model: %v", err)
	}
	if !manager.ModelExists(fixtureModel) {
		t.Fatalf("expected fixture model to be cached")
	}

	backend := newStubBackend(t, stubBackendOptions{allowLoad: true, expectModel: true})
	defer backend.Close()

	runnerURL, closeRunner := startRunner(t, manager.GetModelPath(fixtureModel), backend.server)
	defer closeRunner()

	loadReq := map[string]string{"model": manager.GetModelPath(fixtureModel)}
	loadBody, _ := json.Marshal(loadReq)
	loadResp, err := http.Post(runnerURL+"/load", "application/json", bytes.NewReader(loadBody))
	if err != nil {
		t.Fatalf("load request failed: %v", err)
	}
	if loadResp.StatusCode != http.StatusOK {
		t.Fatalf("expected load success, got %d", loadResp.StatusCode)
	}

	genReq := map[string]any{"prompt": "hello"}
	genBody, _ := json.Marshal(genReq)
	resp, err := http.Post(runnerURL+"/completion?model="+fixtureModel, "application/json", bytes.NewReader(genBody))
	if err != nil {
		t.Fatalf("completion request failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 from runner, got %d", resp.StatusCode)
	}

	var lines int
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		if strings.TrimSpace(scanner.Text()) == "" {
			continue
		}
		lines++
	}

	if lines < 2 {
		t.Fatalf("expected streaming completion response, saw %d lines", lines)
	}
}

func TestMLXMissingModelFails(t *testing.T) {
	backend := newStubBackend(t, stubBackendOptions{allowLoad: true, expectModel: true})
	defer backend.Close()

	runnerURL, closeRunner := startRunner(t, filepath.Join(t.TempDir(), "missing"), backend.server)
	defer closeRunner()

	loadReq := map[string]string{"model": "does-not-exist"}
	body, _ := json.Marshal(loadReq)
	resp, err := http.Post(runnerURL+"/load", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("load request failed: %v", err)
	}
	if resp.StatusCode == http.StatusOK {
		t.Fatalf("expected load failure for missing model")
	}
}

func TestMLXRunnerHandlesBackendFailure(t *testing.T) {
	backend := newStubBackend(t, stubBackendOptions{allowLoad: true, errorOnCompletion: true})
	defer backend.Close()

	runnerURL, closeRunner := startRunner(t, "", backend.server)
	defer closeRunner()

	loadReq := map[string]string{"model": "noop"}
	body, _ := json.Marshal(loadReq)
	_, _ = http.Post(runnerURL+"/load", "application/json", bytes.NewReader(body))

	resp, err := http.Post(runnerURL+"/completion", "application/json", bytes.NewReader([]byte(`{"prompt":"test"}`)))
	if err != nil {
		t.Fatalf("completion request failed: %v", err)
	}
	if resp.StatusCode == http.StatusOK {
		t.Fatalf("expected runner to surface backend failure")
	}
}

func TestMLXCompletionPerformance(t *testing.T) {
	backend := newStubBackend(t, stubBackendOptions{allowLoad: true, delay: 50 * time.Millisecond})
	defer backend.Close()

	runnerURL, closeRunner := startRunner(t, "", backend.server)
	defer closeRunner()

	_, _ = http.Post(runnerURL+"/load", "application/json", bytes.NewReader([]byte(`{"model":"noop"}`)))

	start := time.Now()
	resp, err := http.Post(runnerURL+"/completion", "application/json", bytes.NewReader([]byte(`{"prompt":"time"}`)))
	if err != nil {
		t.Fatalf("completion request failed: %v", err)
	}
	defer resp.Body.Close()

	duration := time.Since(start)
	if duration > time.Second {
		t.Fatalf("completion exceeded timeout budget: %v", duration)
	}
}

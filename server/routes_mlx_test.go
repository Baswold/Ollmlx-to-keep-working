package server

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

func TestStartMLXRunnerPropagatesModelsEnv(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	cmd, _, err := startMLXRunner(context.Background(), "test-model")
	if err != nil {
		t.Fatalf("startMLXRunner() error = %v", err)
	}

	expected := fmt.Sprintf("OLLAMA_MODELS=%s", envconfig.Models())
	found := false
	for _, env := range cmd.Env {
		if env == expected {
			found = true
			break
		}
	}

	if !found {
		t.Fatalf("expected runner environment to include %q", expected)
	}
}

func TestGenerateMLXModelUsesLocalName(t *testing.T) {
	gin.SetMode(gin.TestMode)

	modelName := "mlx-community/llama-2"
	localName := strings.ReplaceAll(modelName, "/", "_")

	modelsRoot := t.TempDir()
	t.Setenv("OLLAMA_MODELS", modelsRoot)

	// Create model at the path MLXModelManager expects (modelsRoot/localName)
	modelDir := filepath.Join(modelsRoot, localName)
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatalf("failed to create model directory: %v", err)
	}

	if err := os.WriteFile(filepath.Join(modelDir, "config.json"), []byte("{}"), 0o644); err != nil {
		t.Fatalf("failed to write config: %v", err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "model.safetensors"), []byte{}, 0o644); err != nil {
		t.Fatalf("failed to write weights: %v", err)
	}

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to start listener: %v", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/completion", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		fmt.Fprintf(w, `{"content":"ok","done":true,"done_reason":"stop"}\n`)
	})

	srv := &http.Server{Handler: mux}
	go srv.Serve(listener)
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		srv.Shutdown(ctx)
	})

	port := listener.Addr().(*net.TCPAddr).Port

	var startedModel string
	startMLXRunnerFunc = func(ctx context.Context, modelName string) (*exec.Cmd, int, error) {
		startedModel = modelName
		return exec.CommandContext(ctx, "true"), port, nil
	}
	defer func() { startMLXRunnerFunc = startMLXRunner }()

	var loadedModel string
	loadMLXModelFunc = func(_ context.Context, _ *http.Client, p int, modelName string) error {
		if p != port {
			t.Fatalf("unexpected port: got %d want %d", p, port)
		}
		loadedModel = modelName
		return nil
	}
	defer func() { loadMLXModelFunc = loadMLXModel }()

	stream := false
	req := &api.GenerateRequest{Model: modelName, Prompt: "Hello", Stream: &stream}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/api/generate", nil)

	srvInstance := &Server{}
	srvInstance.generateMLXModel(c, req)

	if w.Code != http.StatusOK {
		t.Fatalf("unexpected status: got %d body %s", w.Code, w.Body.String())
	}

	if startedModel != localName {
		t.Fatalf("runner received %q, want %q", startedModel, localName)
	}

	if loadedModel != localName {
		t.Fatalf("loader received %q, want %q", loadedModel, localName)
	}
}

func TestWaitForMLXRunnerPropagatesHealthError(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to create listener: %v", err)
	}

	server := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "backend unhealthy", http.StatusServiceUnavailable)
		}),
	}

	go server.Serve(listener)
	defer server.Shutdown(context.Background())

	ctx, cancel := context.WithTimeout(context.Background(), 1200*time.Millisecond)
	defer cancel()

	client := &http.Client{Timeout: 500 * time.Millisecond}
	port := listener.Addr().(*net.TCPAddr).Port

	err = waitForMLXRunner(ctx, client, port)
	if err == nil {
		t.Fatalf("expected waitForMLXRunner to fail")
	}

	if !strings.Contains(err.Error(), "backend unhealthy") {
		t.Fatalf("expected error message to include backend response, got: %v", err)
	}
}

// TestParseParameterCount tests the parameter count parsing function
func TestParseParameterCount(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected int64
	}{
		// Standard formats
		{"7b", "7b", 7_000_000_000},
		{"7B uppercase", "7B", 7_000_000_000},
		{"1.5b", "1.5b", 1_500_000_000},
		{"135m", "135m", 135_000_000},
		{"135M uppercase", "135M", 135_000_000},
		{"1.7b", "1.7b", 1_700_000_000},
		{"3b", "3b", 3_000_000_000},
		{"70b", "70b", 70_000_000_000},

		// Word formats
		{"7 billion", "7 billion", 7_000_000_000},
		{"7billion", "7billion", 7_000_000_000},
		{"135 million", "135 million", 135_000_000},
		{"1.5 billion", "1.5 billion", 1_500_000_000},

		// With commas
		{"7,000,000,000", "7,000,000,000", 7_000_000_000},
		{"135,000,000", "135,000,000", 135_000_000},

		// Edge cases
		{"empty", "", 0},
		{"just spaces", "   ", 0},
		{"0b", "0b", 0},
		{"0.5b", "0.5b", 500_000_000},

		// Trillion (rare)
		{"1t", "1t", 1_000_000_000_000},

		// Thousand
		{"500k", "500k", 500_000},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := parseParameterCount(tt.input)
			if result != tt.expected {
				t.Errorf("parseParameterCount(%q) = %d, want %d", tt.input, result, tt.expected)
			}
		})
	}
}

// TestDetectMLXChatTemplate tests chat template detection for MLX models
func TestDetectMLXChatTemplate(t *testing.T) {
	tests := []struct {
		modelName string
		expected  ChatTemplateType
	}{
		// Qwen family
		{"mlx-community/Qwen2.5-3B-Instruct-4bit", TemplateQwen},
		{"Qwen/Qwen2-VL-7B-Instruct", TemplateQwen},

		// Llama family
		{"mlx-community/Llama-3.2-1B-Instruct-4bit", TemplateLlama},
		{"meta-llama/Llama-2-7b-chat", TemplateLlama},
		{"mlx-community/Meta-Llama3-8B-Instruct", TemplateLlama},

		// Mistral family
		{"mlx-community/Mistral-7B-Instruct-v0.3-4bit", TemplateMistral},
		{"mistralai/Mixtral-8x7B-Instruct", TemplateMistral},

		// Phi family
		{"microsoft/Phi-3.5-mini-instruct", TemplatePhi},
		{"mlx-community/Phi-2", TemplatePhi},

		// Gemma family
		{"google/gemma-2-9b-it", TemplateGemma},
		{"mlx-community/gemma-7b-instruct", TemplateGemma},

		// SmolLM family
		{"mlx-community/SmolLM2-135M-Instruct-4bit", TemplateSmolLM},

		// Unknown (defaults to ChatML)
		{"some-random-model", TemplateChatML},
		{"custom/unknown-model", TemplateChatML},
	}

	for _, tt := range tests {
		t.Run(tt.modelName, func(t *testing.T) {
			result := detectMLXChatTemplate(tt.modelName)
			if result != tt.expected {
				t.Errorf("detectMLXChatTemplate(%q) = %q, want %q", tt.modelName, result, tt.expected)
			}
		})
	}
}

// TestGetImageToken tests image token generation
func TestGetImageToken(t *testing.T) {
	tests := []struct {
		modelName  string
		imageIndex int
		expected   string
	}{
		// Qwen2-VL uses numbered tokens
		{"Qwen/Qwen2-VL-7B-Instruct", 0, "<image_1>"},
		{"Qwen/Qwen2-VL-7B-Instruct", 1, "<image_2>"},
		{"mlx-community/Qwen2-VL-2B-Instruct-4bit", 2, "<image_3>"},

		// LLaVA uses simple <image>
		{"llava-hf/llava-1.5-7b-hf", 0, "<image>"},
		{"mlx-community/llava-1.5-7b-4bit", 1, "<image>"},

		// Pixtral uses simple <image>
		{"mistral/pixtral-12b-2409", 0, "<image>"},

		// Paligemma uses simple <image>
		{"google/paligemma-3b-mix-224", 0, "<image>"},

		// Unknown defaults to <image>
		{"some-random-vision-model", 0, "<image>"},
		{"custom-vlm", 1, "<image>"},
	}

	for _, tt := range tests {
		t.Run(tt.modelName, func(t *testing.T) {
			result := getImageToken(tt.modelName, tt.imageIndex)
			if result != tt.expected {
				t.Errorf("getImageToken(%q, %d) = %q, want %q", tt.modelName, tt.imageIndex, result, tt.expected)
			}
		})
	}
}

// TestParseToolCallsFromText tests tool call parsing from text
func TestParseToolCallsFromText(t *testing.T) {
	tests := []struct {
		name          string
		input         string
		expectCalls   bool
		expectedCount int
		expectedName  string
	}{
		{
			name:          "OpenAI format with function wrapper",
			input:         `{"tool_calls":[{"function":{"name":"get_weather","arguments":{"location":"SF"}}}]}`,
			expectCalls:   true,
			expectedCount: 1,
			expectedName:  "get_weather",
		},
		{
			name:          "Simple format with name and arguments",
			input:         `{"tool_calls":[{"name":"calculate","arguments":{"expression":"2+2"}}]}`,
			expectCalls:   true,
			expectedCount: 1,
			expectedName:  "calculate",
		},
		{
			name:          "Direct single call",
			input:         `{"name":"single_func","arguments":{"key":"value"}}`,
			expectCalls:   true,
			expectedCount: 1,
			expectedName:  "single_func",
		},
		{
			name:          "Tool name as key",
			input:         `{"get_weather":{"location":"NYC"}}`,
			expectCalls:   true,
			expectedCount: 1,
			expectedName:  "get_weather",
		},
		{
			name:          "JSON in text",
			input:         `I'll call the function. {"tool_calls":[{"name":"test","arguments":{}}]} Done.`,
			expectCalls:   true,
			expectedCount: 1,
			expectedName:  "test",
		},
		{
			name:          "Multiple tool calls",
			input:         `{"tool_calls":[{"name":"func1","arguments":{}},{"name":"func2","arguments":{}}]}`,
			expectCalls:   true,
			expectedCount: 2,
			expectedName:  "func1",
		},
		{
			name:        "No tool calls - plain text",
			input:       "This is just regular text without any JSON.",
			expectCalls: false,
		},
		{
			name:        "No tool calls - invalid JSON",
			input:       `{"tool_calls": [broken`,
			expectCalls: false,
		},
		{
			name:        "Empty tool calls array",
			input:       `{"tool_calls":[]}`,
			expectCalls: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls, ok := parseToolCallsFromText(tt.input)

			if tt.expectCalls {
				if !ok {
					t.Errorf("parseToolCallsFromText(%q) returned false, want true", tt.input)
					return
				}
				if len(calls) != tt.expectedCount {
					t.Errorf("parseToolCallsFromText(%q) returned %d calls, want %d", tt.input, len(calls), tt.expectedCount)
					return
				}
				if calls[0].Function.Name != tt.expectedName {
					t.Errorf("parseToolCallsFromText(%q) first call name = %q, want %q", tt.input, calls[0].Function.Name, tt.expectedName)
				}
			} else {
				if ok && len(calls) > 0 {
					t.Errorf("parseToolCallsFromText(%q) returned calls, want none", tt.input)
				}
			}
		})
	}
}

// TestFormatChatPromptWithModel tests model-aware chat formatting
func TestFormatChatPromptWithModel(t *testing.T) {
	tests := []struct {
		modelName      string
		expectedMarker string // A marker that should appear in the output
	}{
		{"mlx-community/Qwen2.5-3B", "<|im_start|>"},            // Qwen format
		{"meta-llama/Llama-3.2-1B", "<|begin_of_text|>"},        // Llama 3 format
		{"mlx-community/Mistral-7B", "<s>"},                     // Mistral format
		{"microsoft/Phi-3.5-mini", "<|system|>"},                // Phi format
		{"google/gemma-2-9b", "<start_of_turn>"},                // Gemma format
		{"mlx-community/SmolLM2-135M", "<|im_start|>"},          // SmolLM format
		{"unknown-model", "<|im_start|>"},                       // Default ChatML
	}

	for _, tt := range tests {
		t.Run(tt.modelName, func(t *testing.T) {
			result := formatChatPromptWithModel(nil, nil, tt.modelName)
			if result == "" {
				t.Errorf("formatChatPromptWithModel(nil, nil, %q) = empty, want content", tt.modelName)
				return
			}

			if !strings.Contains(result, tt.expectedMarker) {
				t.Errorf("formatChatPromptWithModel(nil, nil, %q) = %q, want to contain %q", tt.modelName, result, tt.expectedMarker)
			}
		})
	}
}

// TestToolPromptBlock tests tool prompt generation
func TestToolPromptBlock(t *testing.T) {
	// Test with nil tools
	result := toolPromptBlock(nil)
	if result != "" {
		t.Errorf("toolPromptBlock(nil) = %q, want empty string", result)
	}

	// Test with empty tools
	result = toolPromptBlock(api.Tools{})
	if result != "" {
		t.Errorf("toolPromptBlock([]) = %q, want empty string", result)
	}

	// Test with actual tools
	tools := api.Tools{
		{
			Function: api.ToolFunction{
				Name:        "get_weather",
				Description: "Get weather for a location",
				Parameters: api.ToolFunctionParameters{
					Type:     "object",
					Required: []string{"location"},
					Properties: map[string]api.ToolProperty{
						"location": {Type: api.PropertyType{"string"}, Description: "City name"},
					},
				},
			},
		},
	}

	result = toolPromptBlock(tools)
	if result == "" {
		t.Error("toolPromptBlock(tools) = empty, want content")
	}
	if !strings.Contains(result, "get_weather") {
		t.Errorf("toolPromptBlock(tools) = %q, want to contain 'get_weather'", result)
	}
	if !strings.Contains(result, "tool_calls") {
		t.Errorf("toolPromptBlock(tools) = %q, want to contain 'tool_calls'", result)
	}
}

// BenchmarkParseParameterCount benchmarks the parameter parsing
func BenchmarkParseParameterCount(b *testing.B) {
	inputs := []string{"7b", "1.5b", "135m", "7 billion", "7,000,000,000", ""}

	for i := 0; i < b.N; i++ {
		for _, input := range inputs {
			parseParameterCount(input)
		}
	}
}

// BenchmarkDetectMLXChatTemplate benchmarks template detection
func BenchmarkDetectMLXChatTemplate(b *testing.B) {
	models := []string{
		"mlx-community/Qwen2.5-3B-Instruct",
		"meta-llama/Llama-3.2-1B",
		"mistralai/Mistral-7B",
		"unknown-model",
	}

	for i := 0; i < b.N; i++ {
		for _, model := range models {
			detectMLXChatTemplate(model)
		}
	}
}

// BenchmarkParseToolCallsFromText benchmarks tool call parsing
func BenchmarkParseToolCallsFromText(b *testing.B) {
	inputs := []string{
		`{"tool_calls":[{"name":"get_weather","arguments":{"location":"SF"}}]}`,
		`I'll call a function. {"name":"test","arguments":{}} Done.`,
		"No JSON here at all",
	}

	for i := 0; i < b.N; i++ {
		for _, input := range inputs {
			parseToolCallsFromText(input)
		}
	}
}

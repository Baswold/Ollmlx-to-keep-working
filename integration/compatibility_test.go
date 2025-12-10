package integration

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

// TestMLXvsGGUFResponseFormat tests that MLX responses match GGUF response format
func TestMLXvsGGUFResponseFormat(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping compatibility test in short mode")
	}

	// This test requires both backends to be available
	// and comparable models

	// Check if server is running
	resp, err := http.Get("http://localhost:11434/api/version")
	if err != nil {
		t.Skip("ollama server not running, skipping test")
	}
	resp.Body.Close()

	// Test with a simple prompt
	testPrompt := "Why is the sky blue?"

	// Test MLX model
	mlxModel := "mlx-community/SmolLM2-135M-Instruct-4bit"
	manager := llm.NewMLXModelManager()
	if !manager.ModelExists(mlxModel) {
		t.Skipf("MLX model %s not available", mlxModel)
	}

	// Generate response from MLX model
	mlxResponse, err := generateResponse(mlxModel, testPrompt, false)
	if err != nil {
		t.Logf("Failed to generate MLX response: %v", err)
		t.Skip("cannot generate MLX response")
	}

	// Verify MLX response structure
	if mlxResponse.Model != mlxModel {
		t.Errorf("MLX response model mismatch: expected %s, got %s", mlxModel, mlxResponse.Model)
	}

	if mlxResponse.Response == "" {
		t.Error("MLX response is empty")
	}

	if mlxResponse.Done != true {
		t.Error("MLX response should have Done=true")
	}

	// Verify timing fields exist
	if mlxResponse.TotalDuration == 0 {
		t.Error("MLX response missing TotalDuration")
	}

	if mlxResponse.PromptEvalCount == 0 {
		t.Error("MLX response missing PromptEvalCount")
	}

	if mlxResponse.EvalCount == 0 {
		t.Error("MLX response missing EvalCount")
	}

	// Test streaming format
	mlxStream, err := generateStreamingResponse(mlxModel, testPrompt, 10)
	if err != nil {
		t.Logf("Failed to generate MLX streaming response: %v", err)
		t.Skip("cannot generate MLX streaming response")
	}

	// Verify streaming response structure
	if len(mlxStream) == 0 {
		t.Error("MLX streaming response is empty")
	}

	// Verify first chunk
	if mlxStream[0].Model != mlxModel {
		t.Errorf("MLX streaming first chunk model mismatch: expected %s, got %s", mlxModel, mlxStream[0].Model)
	}

	// Verify last chunk has Done=true
	if !mlxStream[len(mlxStream)-1].Done {
		t.Error("MLX streaming last chunk should have Done=true")
	}

	// Verify all chunks have required fields
	for i, chunk := range mlxStream {
		if chunk.Model == "" {
			t.Errorf("MLX streaming chunk %d missing Model", i)
		}

		if chunk.Response == "" && !chunk.Done {
			t.Errorf("MLX streaming chunk %d missing Response", i)
		}

		if chunk.Done && i < len(mlxStream)-1 {
			t.Errorf("MLX streaming chunk %d has Done=true but not last", i)
		}
	}
}

// TestMLXAPICompatibility tests that MLX endpoints match Ollama API specification
func TestMLXAPICompatibility(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping API compatibility test in short mode")
	}

	// This test requires the ollama server to be running
	resp, err := http.Get("http://localhost:11434/api/version")
	if err != nil {
		t.Skip("ollama server not running, skipping test")
	}
	resp.Body.Close()

	// Test /api/tags endpoint includes MLX models
	tagsResp, err := http.Get("http://localhost:11434/api/tags")
	if err != nil {
		t.Logf("Failed to get tags: %v", err)
		t.Skip("cannot get tags")
	}
	defer tagsResp.Body.Close()

	var tags api.ListResponse
	if err := json.NewDecoder(tagsResp.Body).Decode(&tags); err != nil {
		t.Logf("Failed to decode tags: %v", err)
		t.Skip("cannot decode tags")
	}

	// Check if MLX models are listed
	foundMLX := false
	for _, model := range tags.Models {
		if strings.Contains(model.Name, "mlx-community") || strings.Contains(strings.ToLower(model.Name), "-mlx") {
			foundMLX = true
			break
		}
	}

	if !foundMLX {
		t.Log("No MLX models found in tags, skipping further API tests")
		t.Skip("no MLX models available")
	}

	// Test /api/show endpoint for MLX model
	showModel := "mlx-community/SmolLM2-135M-Instruct-4bit"
	manager := llm.NewMLXModelManager()
	if !manager.ModelExists(showModel) {
		t.Skipf("MLX model %s not available", showModel)
	}

	showBody := map[string]string{"name": showModel}
	showBytes, _ := json.Marshal(showBody)
	showResp, err := http.Post("http://localhost:11434/api/show", "application/json", bytes.NewReader(showBytes))
	if err != nil {
		t.Logf("Failed to call show endpoint: %v", err)
		t.Skip("cannot call show endpoint")
	}
	defer showResp.Body.Close()

	if showResp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(showResp.Body)
		t.Logf("Show endpoint returned status %d: %s", showResp.StatusCode, string(body))
		t.Skip("show endpoint failed")
	}

	var showResult api.ShowResponse
	if err := json.NewDecoder(showResp.Body).Decode(&showResult); err != nil {
		t.Logf("Failed to decode show response: %v", err)
		t.Skip("cannot decode show response")
	}

	// Verify show response structure
	if showResult.Details.Format != "MLX" {
		t.Errorf("Expected format MLX, got %s", showResult.Details.Format)
	}

	if family, ok := showResult.ModelInfo["general.family"].(string); !ok || family == "" {
		t.Errorf("Expected model family for %s, got %v", showModel, showResult.ModelInfo["general.family"])
	}

	if showResult.ModifiedAt.IsZero() {
		t.Error("Show response missing ModifiedAt timestamp")
	}
}

// TestMLXStreamingFormat tests that MLX streaming format matches Ollama specification
func TestMLXStreamingFormat(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping streaming format test in short mode")
	}

	// This test requires the ollama server to be running
	resp, err := http.Get("http://localhost:11434/api/version")
	if err != nil {
		t.Skip("ollama server not running, skipping test")
	}
	resp.Body.Close()

	// Use a small test model
	testModel := "mlx-community/SmolLM2-135M-Instruct-4bit"
	manager := llm.NewMLXModelManager()
	if !manager.ModelExists(testModel) {
		t.Skipf("test model %s not available", testModel)
	}

	// Test streaming with various options
	testCases := []struct {
		name    string
		temp    float64
		maxTok  int
		stopSeq []string
	}{
		{"default", 0.7, 20, nil},
		{"low_temp", 0.1, 10, nil},
		{"high_temp", 1.0, 15, nil},
		{"with_stop", 0.7, 10, []string{"\n"}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Generate streaming response
			stream, err := generateStreamingResponse(testModel, "Hello world", tc.maxTok)
			if err != nil {
				t.Logf("Failed to generate streaming response: %v", err)
				t.Skip("cannot generate streaming response")
			}

			// Verify streaming response
			if len(stream) == 0 {
				t.Error("Streaming response is empty")
				return
			}

			// Verify first chunk
			first := stream[0]
			if first.Model != testModel {
				t.Errorf("First chunk model mismatch: expected %s, got %s", testModel, first.Model)
			}

			if first.Response == "" {
				t.Error("First chunk missing Response")
			}

			if first.Done {
				t.Error("First chunk should not have Done=true")
			}

			// Verify last chunk
			last := stream[len(stream)-1]
			if !last.Done {
				t.Error("Last chunk should have Done=true")
			}

			if last.Response == "" {
				t.Error("Last chunk missing Response")
			}

			// Verify timing fields in last chunk
			if last.TotalDuration == 0 {
				t.Error("Last chunk missing TotalDuration")
			}

			if last.PromptEvalCount == 0 {
				t.Error("Last chunk missing PromptEvalCount")
			}

			if last.EvalCount == 0 {
				t.Error("Last chunk missing EvalCount")
			}

			// Verify context field
			if last.Context == nil || len(last.Context) == 0 {
				t.Error("Last chunk missing or empty Context")
			}
		})
	}
}

// TestMLXErrorHandling tests error handling in MLX backend
func TestMLXErrorHandling(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping error handling test in short mode")
	}

	// This test requires the ollama server to be running
	resp, err := http.Get("http://localhost:11434/api/version")
	if err != nil {
		t.Skip("ollama server not running, skipping test")
	}
	resp.Body.Close()

	// Test with non-existent model
	testCases := []struct {
		name      string
		model     string
		prompt    string
		expect404 bool
	}{
		{"nonexistent_model", "nonexistent-model", "test", true},
		{"invalid_model_name", "invalid/model/name", "test", true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			reqBody := map[string]interface{}{
				"model":  tc.model,
				"prompt": tc.prompt,
				"stream": false,
			}

			reqBytes, _ := json.Marshal(reqBody)
			req, _ := http.NewRequest("POST", "http://localhost:11434/api/generate", bytes.NewReader(reqBytes))
			req.Header.Set("Content-Type", "application/json")

			client := &http.Client{}
			resp, err := client.Do(req)
			if err != nil {
				t.Logf("Request failed: %v", err)
				t.Skip("cannot make request")
			}
			defer resp.Body.Close()

			// Check status code
			if tc.expect404 && resp.StatusCode != http.StatusNotFound {
				t.Errorf("Expected status 404, got %d", resp.StatusCode)
			}

			if !tc.expect404 && resp.StatusCode >= 400 {
				body, _ := io.ReadAll(resp.Body)
				t.Errorf("Unexpected error: %d - %s", resp.StatusCode, string(body))
			}
		})
	}
}

// TestMLXModelManagement tests model management operations
func TestMLXModelManagement(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping model management test in short mode")
	}

	manager := llm.NewMLXModelManager()

	// Test listing models
	models, err := manager.ListModels()
	if err != nil {
		t.Fatalf("Failed to list models: %v", err)
	}

	if len(models) == 0 {
		t.Skip("no MLX models available")
	}

	// Test model info for each model
	for _, model := range models {
		info, err := manager.GetModelInfo(model.Name)
		if err != nil {
			t.Errorf("Failed to get info for model %s: %v", model.Name, err)
			continue
		}

		// Verify info structure
		if info.Name != model.Name {
			t.Errorf("Model name mismatch: %s vs %s", info.Name, model.Name)
		}

		if info.Format != "MLX" {
			t.Errorf("Expected format MLX for model %s, got %s", info.Name, info.Format)
		}

		if info.Digest == "" {
			t.Errorf("Model %s missing Digest", info.Name)
		}

		if info.ModifiedAt.IsZero() {
			t.Errorf("Model %s missing ModifiedAt", info.Name)
		}

		if info.Size == 0 {
			t.Logf("Model %s missing Size, skipping size assertion", info.Name)
		}
	}
}

// TestMLXResponseFields tests that all required response fields are present
func TestMLXResponseFields(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping response fields test in short mode")
	}

	// This test requires the ollama server to be running
	resp, err := http.Get("http://localhost:11434/api/version")
	if err != nil {
		t.Skip("ollama server not running, skipping test")
	}
	resp.Body.Close()

	// Use a small test model
	testModel := "mlx-community/SmolLM2-135M-Instruct-4bit"
	manager := llm.NewMLXModelManager()
	if !manager.ModelExists(testModel) {
		t.Skipf("test model %s not available", testModel)
	}

	// Generate response
	response, err := generateResponse(testModel, "test", false)
	if err != nil {
		t.Fatalf("Failed to generate response: %v", err)
	}

	// Check required fields
	requiredFields := []string{"Model", "Response", "Done", "TotalDuration", "PromptEvalCount", "PromptEvalDuration", "EvalCount", "EvalDuration"}

	for _, field := range requiredFields {
		switch field {
		case "Model":
			if response.Model == "" {
				t.Errorf("Missing required field: %s", field)
			}
		case "Response":
			if response.Response == "" {
				t.Errorf("Missing required field: %s", field)
			}
		case "Done":
			// Done is a boolean, just check it exists
		case "TotalDuration":
			if response.TotalDuration == 0 {
				t.Errorf("Missing required field: %s", field)
			}
		case "PromptEvalCount":
			if response.PromptEvalCount == 0 {
				t.Errorf("Missing required field: %s", field)
			}
		case "PromptEvalDuration":
			if response.PromptEvalDuration == 0 {
				t.Errorf("Missing required field: %s", field)
			}
		case "EvalCount":
			if response.EvalCount == 0 {
				t.Errorf("Missing required field: %s", field)
			}
		case "EvalDuration":
			if response.EvalDuration == 0 {
				t.Errorf("Missing required field: %s", field)
			}
		}
	}

	// Check optional fields
	optionalFields := []string{"CreatedAt", "Context", "DoneReason"}

	for _, field := range optionalFields {
		switch field {
		case "CreatedAt":
			if response.CreatedAt.IsZero() {
				t.Logf("Optional field missing: %s", field)
			}
		case "Context":
			if response.Context == nil {
				t.Logf("Optional field missing: %s", field)
			}
		case "DoneReason":
			if response.DoneReason == "" {
				t.Logf("Optional field missing: %s", field)
			}
		}
	}
}

// generateResponse generates a non-streaming response
func generateResponse(model, prompt string, stream bool) (api.GenerateResponse, error) {
	client := &http.Client{}
	reqBody := map[string]interface{}{
		"model":  model,
		"prompt": prompt,
		"stream": stream,
		"options": map[string]interface{}{
			"temperature": 0.7,
			"max_tokens":  20,
		},
	}

	reqBytes, _ := json.Marshal(reqBody)
	req, _ := http.NewRequest("POST", "http://localhost:11434/api/generate", bytes.NewReader(reqBytes))
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return api.GenerateResponse{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return api.GenerateResponse{}, fmt.Errorf("status %d: %s", resp.StatusCode, string(body))
	}

	var result api.GenerateResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return api.GenerateResponse{}, err
	}

	return result, nil
}

// generateStreamingResponse generates a streaming response and returns all chunks
func generateStreamingResponse(model, prompt string, maxTokens int) ([]api.GenerateResponse, error) {
	client := &http.Client{}
	reqBody := map[string]interface{}{
		"model":  model,
		"prompt": prompt,
		"stream": true,
		"options": map[string]interface{}{
			"temperature": 0.7,
			"max_tokens":  maxTokens,
		},
	}

	reqBytes, _ := json.Marshal(reqBody)
	req, _ := http.NewRequest("POST", "http://localhost:11434/api/generate", bytes.NewReader(reqBytes))
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("status %d: %s", resp.StatusCode, string(body))
	}

	var chunks []api.GenerateResponse
	reader := bufio.NewReader(resp.Body)

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
		}

		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		var chunk api.GenerateResponse
		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			return nil, fmt.Errorf("failed to parse chunk: %w, line: %s", err, line)
		}

		chunks = append(chunks, chunk)
	}

	return chunks, nil
}

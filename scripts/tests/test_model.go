//go:build ignore
// +build ignore

package main

import (
	"fmt"
	"github.com/ollama/ollama/llm"
)

func main() {
	manager := llm.NewMLXModelManager()

	// Test with the model name
	modelName := "mlx-community/gemma-3-270m-4bit"
	exists := manager.ModelExists(modelName)
	fmt.Printf("Model exists: %v\n", exists)

	// Check the model path
	modelPath := manager.GetModelPath(modelName)
	fmt.Printf("Model path: %s\n", modelPath)

	// Check if the path exists
	fmt.Printf("Path exists: %v\n", manager.ModelExists(modelName))
}

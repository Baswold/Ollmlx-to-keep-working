//go:build ignore
// +build ignore

package main

import (
	"fmt"
	"github.com/ollama/ollama/llm"
)

func main() {
	manager := llm.NewMLXModelManager()

	modelName := "mlx-community/gemma-3-270m-4bit"
	exists := manager.ModelExists(modelName)
	fmt.Printf("Model exists: %v\n", exists)

	if !exists {
		fmt.Println("Model does not exist!")
		return
	}

	modelPath := manager.GetModelPath(modelName)
	fmt.Printf("Model path: %s\n", modelPath)
}

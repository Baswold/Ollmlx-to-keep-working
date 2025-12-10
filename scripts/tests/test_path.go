//go:build ignore
// +build ignore

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

func main() {
	modelName := "mlx-community/gemma-3-270m-4bit"

	// Convert HuggingFace URL format to local directory name
	localName := strings.ReplaceAll(modelName, "/", "_")
	fmt.Printf("Local name: %s\n", localName)

	// Get the models directory
	homeDir, _ := os.UserHomeDir()
	modelsDir := filepath.Join(homeDir, ".ollama", "models", "mlx")
	fmt.Printf("Models dir: %s\n", modelsDir)

	// Get the full path
	modelPath := filepath.Join(modelsDir, localName)
	fmt.Printf("Model path: %s\n", modelPath)

	// Check if it exists
	if _, err := os.Stat(modelPath); err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("Path exists!\n")
	}
}

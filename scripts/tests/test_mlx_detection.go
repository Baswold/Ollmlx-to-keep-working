//go:build ignore
// +build ignore

package main

import (
	"fmt"
	"strings"
)

func IsMLXModelReference(modelName string) bool {
	if strings.HasPrefix(modelName, "mlx-community/") {
		return true
	}

	if strings.Contains(strings.ToLower(modelName), "-mlx") {
		return true
	}

	return false
}

func main() {
	modelName := "mlx-community/gemma-3-270m-4bit"
	isMLX := IsMLXModelReference(modelName)
	fmt.Printf("Is MLX model: %v\n", isMLX)
}

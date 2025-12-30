//go:build !windows && !darwin

package cmd

import (
	"context"
	"errors"

	"github.com/ollama/ollama/api"
)

func startApp(ctx context.Context, client *api.Client) error {
	return errors.New("could not connect to ollmlx server, run 'ollmlx serve' to start it")
}

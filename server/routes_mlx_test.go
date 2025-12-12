package server

import (
	"context"
	"net"
	"net/http"
	"strings"
	"testing"
	"time"
)

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

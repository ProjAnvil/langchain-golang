package middleware

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestShellSessionPersistsCwdAndEnv(t *testing.T) {
	dir, err := os.MkdirTemp("", "lgshell-")
	if err != nil {
		t.Fatalf("MkdirTemp: %v", err)
	}
	defer os.RemoveAll(dir)
	// macOS: /var is a symlink to /private/var, so resolve it for a stable pwd.
	dir, _ = filepath.EvalSymlinks(dir)

	s := NewShellSession(dir, []string{"/bin/sh"}, map[string]string{})
	if err := s.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer s.Stop(5 * time.Second)

	// Initial cwd is the workspace.
	r, err := s.Execute(context.Background(), "pwd", 10*time.Second)
	if err != nil {
		t.Fatalf("Execute pwd: %v", err)
	}
	if got := strings.TrimSpace(r.Output); got != dir {
		t.Fatalf("initial pwd = %q, want %q", got, dir)
	}

	// cd persists across commands.
	if _, err := s.Execute(context.Background(), "cd /tmp && pwd", 10*time.Second); err != nil {
		t.Fatalf("Execute cd: %v", err)
	}
	r, err = s.Execute(context.Background(), "pwd", 10*time.Second)
	if err != nil {
		t.Fatalf("Execute pwd after cd: %v", err)
	}
	if got := strings.TrimSpace(r.Output); got != "/tmp" {
		t.Fatalf("pwd after cd = %q, want /tmp", got)
	}

	// Env var persists across commands.
	if _, err := s.Execute(context.Background(), "export FOO=bar", 10*time.Second); err != nil {
		t.Fatalf("Execute export: %v", err)
	}
	r, err = s.Execute(context.Background(), "echo $FOO", 10*time.Second)
	if err != nil {
		t.Fatalf("Execute echo: %v", err)
	}
	if got := strings.TrimSpace(r.Output); got != "bar" {
		t.Fatalf("echo FOO = %q, want bar", got)
	}
}

func TestShellSessionExitCode(t *testing.T) {
	dir, err := os.MkdirTemp("", "lgshell-")
	if err != nil {
		t.Fatalf("MkdirTemp: %v", err)
	}
	defer os.RemoveAll(dir)
	// macOS: /var is a symlink to /private/var, so resolve it for a stable pwd.
	dir, _ = filepath.EvalSymlinks(dir)

	s := NewShellSession(dir, []string{"/bin/sh"}, map[string]string{})
	if err := s.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer s.Stop(5 * time.Second)

	r, err := s.Execute(context.Background(), "false", 10*time.Second)
	if err != nil {
		t.Fatalf("Execute false: %v", err)
	}
	if r.ExitCode != 1 {
		t.Fatalf("exit code = %d, want 1", r.ExitCode)
	}
}

func TestShellSessionNotStarted(t *testing.T) {
	s := NewShellSession("/tmp", []string{"/bin/sh"}, nil)
	if _, err := s.Execute(context.Background(), "pwd", time.Second); err == nil {
		t.Fatal("expected error when session not started")
	}
}

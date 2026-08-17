package middleware

import (
	"context"
	"strings"
	"testing"
	"time"
)

func newStartedSession(t *testing.T) *ShellSession {
	t.Helper()
	s := NewShellSession(t.TempDir(), []string{"/bin/sh"}, map[string]string{})
	if err := s.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() { _ = s.Stop(5 * time.Second) })
	return s
}

func TestNewShellSessionArgumentHandling(t *testing.T) {
	// A trailing "-c" is stripped so the shell reads from stdin.
	s := NewShellSession(t.TempDir(), []string{"/bin/sh", "-c"}, nil)
	if len(s.shell) != 1 || s.shell[0] != "/bin/sh" {
		t.Fatalf("shell argv mismatch: %#v", s.shell)
	}
	// An empty argv falls back to /bin/sh.
	s = NewShellSession(t.TempDir(), nil, nil)
	if len(s.shell) != 1 || s.shell[0] != "/bin/sh" {
		t.Fatalf("default shell argv mismatch: %#v", s.shell)
	}
	// Env is copied.
	env := map[string]string{"A": "1"}
	s = NewShellSession(t.TempDir(), []string{"/bin/sh"}, env)
	env["A"] = "mutated"
	if s.env["A"] != "1" {
		t.Fatalf("env should be copied: %#v", s.env)
	}
}

func TestShellSessionStartTwiceIsNoop(t *testing.T) {
	s := newStartedSession(t)
	if err := s.Start(context.Background()); err != nil {
		t.Fatalf("second Start should be a no-op: %v", err)
	}
}

func TestShellSessionStartFailure(t *testing.T) {
	s := NewShellSession(t.TempDir(), []string{"/nonexistent-lg-shell"}, nil)
	if err := s.Start(context.Background()); err == nil {
		t.Fatal("expected start failure for missing shell binary")
	}
}

func TestShellSessionEnvPassthrough(t *testing.T) {
	s := NewShellSession(t.TempDir(), []string{"/bin/sh"}, map[string]string{"LG_SESSION_VAR": "session-value"})
	if err := s.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() { _ = s.Stop(5 * time.Second) })
	r, err := s.Execute(context.Background(), "echo \"$LG_SESSION_VAR\"", 10*time.Second)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if r.Output != "session-value\n" {
		t.Fatalf("env mismatch: %q", r.Output)
	}
}

func TestShellSessionExecuteTimeout(t *testing.T) {
	s := newStartedSession(t)
	r, err := s.Execute(context.Background(), "sleep 2", 200*time.Millisecond)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !r.TimedOut || r.Output != "Command timed out." {
		t.Fatalf("timeout mismatch: %#v", r)
	}
}

func TestShellSessionExecuteContextCanceled(t *testing.T) {
	s := newStartedSession(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	r, err := s.Execute(ctx, "sleep 2", 30*time.Second)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !r.TimedOut {
		t.Fatalf("expected TimedOut on canceled context: %#v", r)
	}
}

func TestShellSessionExecuteAfterShellExit(t *testing.T) {
	s := newStartedSession(t)
	// "exit" terminates the shell; the reader hits EOF before any marker.
	if _, err := s.Execute(context.Background(), "exit", 10*time.Second); err == nil ||
		!strings.Contains(err.Error(), "read shell output") {
		t.Fatalf("expected read error after shell exit, got %v", err)
	}
}

func TestShellSessionRestart(t *testing.T) {
	s := newStartedSession(t)
	if _, err := s.Execute(context.Background(), "export LG_RESTART_VAR=before", 10*time.Second); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if err := s.Restart(context.Background(), 5*time.Second); err != nil {
		t.Fatalf("Restart: %v", err)
	}
	// After a restart the env from the previous shell is gone.
	r, err := s.Execute(context.Background(), "echo \"[$LG_RESTART_VAR]\"", 10*time.Second)
	if err != nil {
		t.Fatalf("Execute after restart: %v", err)
	}
	if r.Output != "[]\n" {
		t.Fatalf("env should reset after restart: %q", r.Output)
	}
}

func TestShellSessionStopWhenNotStarted(t *testing.T) {
	s := NewShellSession(t.TempDir(), []string{"/bin/sh"}, nil)
	if err := s.Stop(time.Second); err != nil {
		t.Fatalf("stop on unstarted session should be nil: %v", err)
	}
}

func TestShellSessionExecuteAfterStop(t *testing.T) {
	s := newStartedSession(t)
	if err := s.Stop(5 * time.Second); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	if _, err := s.Execute(context.Background(), "printf hi", time.Second); err == nil {
		t.Fatal("expected error executing on a stopped session")
	}
	// Stopping again is a no-op.
	if err := s.Stop(time.Second); err != nil {
		t.Fatalf("second Stop should be nil: %v", err)
	}
}

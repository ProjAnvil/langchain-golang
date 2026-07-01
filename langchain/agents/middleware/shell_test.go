package middleware

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestShellToolMiddlewareRunCommand(t *testing.T) {
	middleware, err := NewShellToolMiddleware(t.TempDir())
	if err != nil {
		t.Fatalf("new shell middleware: %v", err)
	}
	result, err := middleware.Run(context.Background(), "printf hello", false)
	if err != nil {
		t.Fatalf("run command: %v", err)
	}
	if result.Output != "hello" || result.ExitCode != 0 || result.TimedOut {
		t.Fatalf("result mismatch: %#v", result)
	}
}

func TestShellToolMiddlewareValidatesPayload(t *testing.T) {
	middleware, err := NewShellToolMiddleware(t.TempDir())
	if err != nil {
		t.Fatalf("new shell middleware: %v", err)
	}
	if _, err := middleware.Run(context.Background(), "", false); err == nil {
		t.Fatal("expected empty payload error")
	}
	if _, err := middleware.Run(context.Background(), "echo hi", true); err == nil {
		t.Fatal("expected command plus restart error")
	}
	result, err := middleware.Run(context.Background(), "", true)
	if err != nil {
		t.Fatalf("restart: %v", err)
	}
	if !strings.Contains(result.Output, "restarted") {
		t.Fatalf("restart output mismatch: %#v", result)
	}
}

func TestShellToolMiddlewareTimeoutAndTruncation(t *testing.T) {
	policy := DefaultShellExecutionPolicy()
	policy.CommandTimeout = 10 * time.Millisecond
	policy.MaxOutputLines = 1
	policy.MaxOutputBytes = 5
	middleware, err := NewShellToolMiddleware(t.TempDir(), WithShellExecutionPolicy(policy))
	if err != nil {
		t.Fatalf("new shell middleware: %v", err)
	}
	result, err := middleware.Run(context.Background(), "printf 'abcdef\\nsecond\\n'", false)
	if err != nil {
		t.Fatalf("run command: %v", err)
	}
	if result.Output != "abcde" || !result.TruncatedByBytes || !result.TruncatedByLines {
		t.Fatalf("truncation mismatch: %#v", result)
	}

	result, err = middleware.Run(context.Background(), "sleep 1", false)
	if err != nil {
		t.Fatalf("run timeout command: %v", err)
	}
	if !result.TimedOut {
		t.Fatalf("expected timeout: %#v", result)
	}
}

func TestShellToolMiddlewareRedactsOutput(t *testing.T) {
	rule, err := (RedactionRule{PIIType: "email", Strategy: RedactionRedact}).Resolve()
	if err != nil {
		t.Fatalf("resolve rule: %v", err)
	}
	middleware, err := NewShellToolMiddleware(t.TempDir(), WithShellRedactionRules(rule))
	if err != nil {
		t.Fatalf("new shell middleware: %v", err)
	}
	result, err := middleware.Run(context.Background(), "printf user@example.com", false)
	if err != nil {
		t.Fatalf("run command: %v", err)
	}
	if result.Output != "[REDACTED_EMAIL]" {
		t.Fatalf("redacted output mismatch: %#v", result)
	}
}

func TestShellExecutionPolicyValidation(t *testing.T) {
	policy := DefaultShellExecutionPolicy()
	policy.MaxOutputLines = 0
	if _, err := NewShellToolMiddleware(t.TempDir(), WithShellExecutionPolicy(policy)); err == nil {
		t.Fatal("expected invalid policy error")
	}
	if _, err := NewShellToolMiddleware(t.TempDir(), WithShellCommand()); err == nil {
		t.Fatal("expected empty shell command error")
	}
}

func TestShellLifecycleStartupResourcesAndShutdown(t *testing.T) {
	root := t.TempDir()
	middleware, err := NewShellToolMiddleware(
		root,
		WithShellStartupCommands("printf start >> lifecycle.log"),
		WithShellShutdownCommands("printf stop >> lifecycle.log"),
	)
	if err != nil {
		t.Fatalf("new shell middleware: %v", err)
	}
	state := map[string]any{}

	update, err := middleware.BeforeAgent(context.Background(), state)
	if err != nil {
		t.Fatalf("before agent: %v", err)
	}
	resources := update[ShellSessionResourcesKey].(*ShellSessionResources)
	if !resources.StartupRan || resources.WorkspaceRoot != middleware.WorkspaceRoot {
		t.Fatalf("resources mismatch: %#v", resources)
	}

	second, err := middleware.BeforeAgent(context.Background(), state)
	if err != nil {
		t.Fatalf("second before agent: %v", err)
	}
	if second[ShellSessionResourcesKey] != resources {
		t.Fatal("expected resumable resources to be reused")
	}

	data, err := os.ReadFile(filepath.Join(root, "lifecycle.log"))
	if err != nil {
		t.Fatalf("read lifecycle log: %v", err)
	}
	if string(data) != "start" {
		t.Fatalf("startup should run once, got %q", string(data))
	}

	result, err := middleware.RunWithState(context.Background(), state, "pwd", false)
	if err != nil {
		t.Fatalf("run with state: %v", err)
	}
	wantRoot, err := filepath.EvalSymlinks(root)
	if err != nil {
		t.Fatalf("eval root: %v", err)
	}
	gotRoot, err := filepath.EvalSymlinks(strings.TrimSpace(result.Output))
	if err != nil {
		t.Fatalf("eval pwd: %v", err)
	}
	if gotRoot != wantRoot {
		t.Fatalf("run workspace mismatch: %#v", result)
	}

	if err := middleware.AfterAgent(context.Background(), state); err != nil {
		t.Fatalf("after agent: %v", err)
	}
	if !resources.ShutdownRan || !resources.Closed {
		t.Fatalf("shutdown resources mismatch: %#v", resources)
	}
	data, err = os.ReadFile(filepath.Join(root, "lifecycle.log"))
	if err != nil {
		t.Fatalf("read lifecycle log after shutdown: %v", err)
	}
	if string(data) != "startstop" {
		t.Fatalf("shutdown did not run: %q", string(data))
	}
}

func TestShellLifecycleRestartRerunsStartup(t *testing.T) {
	root := t.TempDir()
	middleware, err := NewShellToolMiddleware(root, WithShellStartupCommands("printf start >> restart.log"))
	if err != nil {
		t.Fatalf("new shell middleware: %v", err)
	}
	state := map[string]any{}
	if _, err := middleware.BeforeAgent(context.Background(), state); err != nil {
		t.Fatalf("before agent: %v", err)
	}
	if _, err := middleware.RunWithState(context.Background(), state, "", true); err != nil {
		t.Fatalf("restart: %v", err)
	}
	data, err := os.ReadFile(filepath.Join(root, "restart.log"))
	if err != nil {
		t.Fatalf("read restart log: %v", err)
	}
	if string(data) != "startstart" {
		t.Fatalf("startup should rerun on restart, got %q", string(data))
	}
}

func TestShellLifecycleStartupFailure(t *testing.T) {
	middleware, err := NewShellToolMiddleware(t.TempDir(), WithShellStartupCommands("exit 7"))
	if err != nil {
		t.Fatalf("new shell middleware: %v", err)
	}
	_, err = middleware.BeforeAgent(context.Background(), map[string]any{})
	if err == nil || !strings.Contains(err.Error(), "startup command") {
		t.Fatalf("expected startup failure, got %v", err)
	}
}

func TestShellLifecycleOwnedWorkspaceCleanup(t *testing.T) {
	middleware, err := NewShellToolMiddleware("")
	if err != nil {
		t.Fatalf("new shell middleware: %v", err)
	}
	workspace := middleware.WorkspaceRoot
	state := map[string]any{}
	if _, err := middleware.BeforeAgent(context.Background(), state); err != nil {
		t.Fatalf("before agent: %v", err)
	}
	if _, err := os.Stat(workspace); err != nil {
		t.Fatalf("workspace should exist before cleanup: %v", err)
	}
	if err := middleware.AfterAgent(context.Background(), state); err != nil {
		t.Fatalf("after agent: %v", err)
	}
	if _, err := os.Stat(workspace); !os.IsNotExist(err) {
		t.Fatalf("expected owned workspace cleanup, stat err=%v", err)
	}
}

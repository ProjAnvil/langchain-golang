package middleware

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestShellExecutionPolicyValidate(t *testing.T) {
	valid := DefaultShellExecutionPolicy()
	if err := valid.Validate(); err != nil {
		t.Fatalf("default policy should be valid: %v", err)
	}

	policy := DefaultShellExecutionPolicy()
	policy.CommandTimeout = -time.Second
	if err := policy.Validate(); err == nil || !strings.Contains(err.Error(), "non-negative") {
		t.Fatalf("expected negative timeout error, got %v", err)
	}

	policy = DefaultShellExecutionPolicy()
	policy.MaxOutputLines = 0
	if err := policy.Validate(); err == nil || !strings.Contains(err.Error(), "max_output_lines") {
		t.Fatalf("expected max output lines error, got %v", err)
	}

	policy = DefaultShellExecutionPolicy()
	policy.MaxOutputBytes = -1
	if err := policy.Validate(); err == nil || !strings.Contains(err.Error(), "max_output_bytes") {
		t.Fatalf("expected max output bytes error, got %v", err)
	}
}

func TestShellToolMiddlewareToolNameDefaultsWhenEmpty(t *testing.T) {
	middleware, err := NewShellToolMiddleware(t.TempDir(), func(m *ShellToolMiddleware) {
		m.ToolName = ""
	})
	if err != nil {
		t.Fatalf("new shell middleware: %v", err)
	}
	if middleware.ToolName != ShellToolName {
		t.Fatalf("tool name should default to %q, got %q", ShellToolName, middleware.ToolName)
	}
}

func TestShellToolMiddlewareWorkspaceUnderFileFails(t *testing.T) {
	file := filepath.Join(t.TempDir(), "afile")
	if err := os.WriteFile(file, []byte("x"), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}
	if _, err := NewShellToolMiddleware(filepath.Join(file, "sub")); err == nil {
		t.Fatal("expected workspace creation failure under a regular file")
	}
}

func TestShellToolMiddlewareToolInvocation(t *testing.T) {
	middleware, err := NewShellToolMiddleware(t.TempDir())
	if err != nil {
		t.Fatalf("new shell middleware: %v", err)
	}
	if len(middleware.Tools) != 1 {
		t.Fatalf("tools mismatch: %#v", middleware.Tools)
	}

	result, err := middleware.Tools[0].Invoke(context.Background(), map[string]any{"command": "printf hi"})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if result.Content != "hi" {
		t.Fatalf("tool result mismatch: %#v", result)
	}
	if result.Metadata["exit_code"] != 0 || result.Metadata["timed_out"] != false {
		t.Fatalf("tool metadata mismatch: %#v", result.Metadata)
	}

	// A failing run surfaces the error.
	if _, err := middleware.Tools[0].Invoke(context.Background(), map[string]any{}); err == nil {
		t.Fatal("expected error for empty command")
	}

	// Restart via the tool.
	result, err = middleware.Tools[0].Invoke(context.Background(), map[string]any{"restart": true})
	if err != nil || !strings.Contains(result.Content, "restarted") {
		t.Fatalf("restart result mismatch: %#v %v", result, err)
	}
}

func TestShellToolMiddlewareEnvAndSingleArgShell(t *testing.T) {
	middleware, err := NewShellToolMiddleware(
		t.TempDir(),
		WithShellCommand("/bin/sh"), // single arg: "-c" is appended automatically
		WithShellEnv(map[string]string{"LG_TEST_VAR": "lg-value"}),
	)
	if err != nil {
		t.Fatalf("new shell middleware: %v", err)
	}
	result, err := middleware.Run(context.Background(), "printf %s \"$LG_TEST_VAR\"", false)
	if err != nil {
		t.Fatalf("run command: %v", err)
	}
	if result.Output != "lg-value" {
		t.Fatalf("env var not propagated: %#v", result)
	}
}

func TestShellToolMiddlewareZeroTimeoutDefaults(t *testing.T) {
	policy := DefaultShellExecutionPolicy()
	policy.CommandTimeout = 0 // triggers the 30s default inside runInWorkspace
	middleware, err := NewShellToolMiddleware(t.TempDir(), WithShellExecutionPolicy(policy))
	if err != nil {
		t.Fatalf("new shell middleware: %v", err)
	}
	result, err := middleware.Run(context.Background(), "printf ok", false)
	if err != nil || result.Output != "ok" {
		t.Fatalf("run mismatch: %#v %v", result, err)
	}
}

func TestShellToolMiddlewareMissingShellBinary(t *testing.T) {
	middleware, err := NewShellToolMiddleware(t.TempDir(), WithShellCommand("/nonexistent-lg-shell", "-c"))
	if err != nil {
		t.Fatalf("new shell middleware: %v", err)
	}
	if _, err := middleware.Run(context.Background(), "printf hi", false); err == nil {
		t.Fatal("expected spawn error for missing shell binary")
	}
}

func TestShellToolMiddlewareMergesStderr(t *testing.T) {
	middleware, err := NewShellToolMiddleware(t.TempDir())
	if err != nil {
		t.Fatalf("new shell middleware: %v", err)
	}
	result, err := middleware.Run(context.Background(), "printf out; printf err >&2", false)
	if err != nil {
		t.Fatalf("run command: %v", err)
	}
	if result.Output != "out\nerr" {
		t.Fatalf("stderr merge mismatch: %#v", result)
	}
}

func TestShellToolMiddlewareEmptyOutput(t *testing.T) {
	middleware, err := NewShellToolMiddleware(t.TempDir())
	if err != nil {
		t.Fatalf("new shell middleware: %v", err)
	}
	result, err := middleware.Run(context.Background(), "true", false)
	if err != nil {
		t.Fatalf("run command: %v", err)
	}
	if result.Output != "" || result.TotalLines != 0 || result.TotalBytes != 0 {
		t.Fatalf("empty output mismatch: %#v", result)
	}
}

func TestShellToolMiddlewareRedactionBlockError(t *testing.T) {
	rule, err := (RedactionRule{PIIType: "email", Strategy: RedactionBlock}).Resolve()
	if err != nil {
		t.Fatalf("resolve rule: %v", err)
	}
	middleware, err := NewShellToolMiddleware(t.TempDir(), WithShellRedactionRules(rule))
	if err != nil {
		t.Fatalf("new shell middleware: %v", err)
	}
	if _, err := middleware.Run(context.Background(), "printf user@example.com", false); err == nil {
		t.Fatal("expected redaction block error")
	}
}

func TestShellAfterAgentWithoutResources(t *testing.T) {
	middleware, err := NewShellToolMiddleware(t.TempDir())
	if err != nil {
		t.Fatalf("new shell middleware: %v", err)
	}
	// No resources in state (and nil state): AfterAgent is a no-op.
	if err := middleware.AfterAgent(context.Background(), nil); err != nil {
		t.Fatalf("after agent: %v", err)
	}
	if err := middleware.AfterAgent(context.Background(), map[string]any{}); err != nil {
		t.Fatalf("after agent: %v", err)
	}
}

func TestShellStartupCommandSpawnError(t *testing.T) {
	middleware, err := NewShellToolMiddleware(
		t.TempDir(),
		WithShellCommand("/nonexistent-lg-shell", "-c"),
		WithShellStartupCommands("printf hi"),
	)
	if err != nil {
		t.Fatalf("new shell middleware: %v", err)
	}
	_, err = middleware.BeforeAgent(context.Background(), map[string]any{})
	if err == nil || !strings.Contains(err.Error(), "startup command") {
		t.Fatalf("expected startup spawn error, got %v", err)
	}

	// RunWithState surfaces the same resource-creation failure.
	if _, err := middleware.RunWithState(context.Background(), map[string]any{}, "printf hi", false); err == nil {
		t.Fatal("expected RunWithState to fail when resources cannot be created")
	}
}

func TestShellRunWithStateRestartFailure(t *testing.T) {
	root := t.TempDir()
	middleware, err := NewShellToolMiddleware(root, WithShellStartupCommands("exit 3"))
	if err != nil {
		t.Fatalf("new shell middleware: %v", err)
	}
	state := map[string]any{}
	// Seed resources manually so the restart path runs its startup commands.
	state[ShellSessionResourcesKey] = &ShellSessionResources{WorkspaceRoot: middleware.WorkspaceRoot}
	if _, err := middleware.RunWithState(context.Background(), state, "", true); err == nil ||
		!strings.Contains(err.Error(), "exit code 3") {
		t.Fatalf("expected restart startup failure, got %v", err)
	}
}

func TestShellPersistentSessionLifecycle(t *testing.T) {
	root := t.TempDir()
	resolved, err := filepath.EvalSymlinks(root)
	if err != nil {
		t.Fatalf("eval root: %v", err)
	}
	middleware, err := NewShellToolMiddleware(root, WithShellPersistentSession())
	if err != nil {
		t.Fatalf("new shell middleware: %v", err)
	}

	// cwd persists across commands in the persistent session.
	if _, err := middleware.Run(context.Background(), "cd /tmp", false); err != nil {
		t.Fatalf("run cd: %v", err)
	}
	result, err := middleware.Run(context.Background(), "pwd", false)
	if err != nil {
		t.Fatalf("run pwd: %v", err)
	}
	if got := strings.TrimSpace(result.Output); got != "/tmp" {
		t.Fatalf("persistent cwd mismatch: %q (workspace %q)", got, resolved)
	}

	// AfterAgent stops the persistent session.
	state := map[string]any{}
	if _, err := middleware.BeforeAgent(context.Background(), state); err != nil {
		t.Fatalf("before agent: %v", err)
	}
	if err := middleware.AfterAgent(context.Background(), state); err != nil {
		t.Fatalf("after agent: %v", err)
	}
	middleware.sessionMu.Lock()
	stopped := middleware.session == nil
	middleware.sessionMu.Unlock()
	if !stopped {
		t.Fatal("persistent session should be stopped after AfterAgent")
	}
}

func TestShellPersistentSessionShellExitError(t *testing.T) {
	middleware, err := NewShellToolMiddleware(t.TempDir(), WithShellPersistentSession())
	if err != nil {
		t.Fatalf("new shell middleware: %v", err)
	}
	// "exit" terminates the persistent shell; reading its output fails.
	if _, err := middleware.Run(context.Background(), "exit", false); err == nil {
		t.Fatal("expected error after the persistent shell exits")
	}
	middleware.stopPersistentSession(time.Second)
}

func TestShellPersistentSessionTimeout(t *testing.T) {
	policy := DefaultShellExecutionPolicy()
	policy.CommandTimeout = 200 * time.Millisecond
	middleware, err := NewShellToolMiddleware(t.TempDir(), WithShellPersistentSession(), WithShellExecutionPolicy(policy))
	if err != nil {
		t.Fatalf("new shell middleware: %v", err)
	}
	result, err := middleware.Run(context.Background(), "sleep 2", false)
	if err != nil {
		t.Fatalf("run command: %v", err)
	}
	if !result.TimedOut || result.Output != "Command timed out." {
		t.Fatalf("persistent timeout mismatch: %#v", result)
	}
	// Stop the stuck session so the test does not leak a sleeping shell. The
	// timeout must outlast the shell's `sleep 2` so Stop takes the clean
	// Wait path (a timeout kill would race with the Wait goroutine).
	middleware.stopPersistentSession(5 * time.Second)
	middleware.sessionMu.Lock()
	stopped := middleware.session == nil
	middleware.sessionMu.Unlock()
	if !stopped {
		t.Fatal("session should be cleared after stopPersistentSession")
	}
}

func TestShellStopPersistentSessionWithoutSession(t *testing.T) {
	middleware, err := NewShellToolMiddleware(t.TempDir(), WithShellPersistentSession())
	if err != nil {
		t.Fatalf("new shell middleware: %v", err)
	}
	// No session started: stop is a no-op.
	middleware.stopPersistentSession(time.Second)
}

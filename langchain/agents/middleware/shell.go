package middleware

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
)

const ShellToolName = "shell"

const ShellTempPrefix = "langchain-shell-"

const DefaultShellToolDescription = "Execute a shell command inside a controlled session. Outputs may be truncated when they become very large, and long running commands will be terminated once their configured timeout elapses."

type ShellExecutionPolicy struct {
	CommandTimeout     time.Duration
	StartupTimeout     time.Duration
	TerminationTimeout time.Duration
	MaxOutputLines     int
	MaxOutputBytes     int
}

func DefaultShellExecutionPolicy() ShellExecutionPolicy {
	return ShellExecutionPolicy{
		CommandTimeout:     30 * time.Second,
		StartupTimeout:     30 * time.Second,
		TerminationTimeout: 10 * time.Second,
		MaxOutputLines:     100,
	}
}

func (p ShellExecutionPolicy) Validate() error {
	if p.CommandTimeout < 0 || p.StartupTimeout < 0 || p.TerminationTimeout < 0 {
		return fmt.Errorf("shell timeouts must be non-negative")
	}
	if p.MaxOutputLines <= 0 {
		return fmt.Errorf("max_output_lines must be positive")
	}
	if p.MaxOutputBytes < 0 {
		return fmt.Errorf("max_output_bytes must be non-negative")
	}
	return nil
}

type CommandExecutionResult struct {
	Output           string
	ExitCode         int
	TimedOut         bool
	TruncatedByLines bool
	TruncatedByBytes bool
	TotalLines       int
	TotalBytes       int
}

const ShellSessionResourcesKey = "shell_session_resources"

type ShellSessionResources struct {
	WorkspaceRoot string
	OwnsWorkspace bool
	StartupRan    bool
	ShutdownRan   bool
	Closed        bool
}

type ShellToolMiddleware struct {
	WorkspaceRoot    string
	StartupCommands  []string
	ShutdownCommands []string
	Policy           ShellExecutionPolicy
	RedactionRules   []ResolvedRedactionRule
	ToolName         string
	ShellCommand     []string
	Env              map[string]string
	Tools            []tools.Tool
	OwnsWorkspace    bool
}

func NewShellToolMiddleware(workspaceRoot string, opts ...ShellToolOption) (*ShellToolMiddleware, error) {
	policy := DefaultShellExecutionPolicy()
	m := &ShellToolMiddleware{
		WorkspaceRoot: workspaceRoot,
		Policy:        policy,
		ToolName:      ShellToolName,
		ShellCommand:  []string{"/bin/sh", "-c"},
		Env:           map[string]string{},
	}
	for _, opt := range opts {
		opt(m)
	}
	if m.ToolName == "" {
		m.ToolName = ShellToolName
	}
	if len(m.ShellCommand) == 0 {
		return nil, fmt.Errorf("shell command must contain at least one argument")
	}
	if err := m.Policy.Validate(); err != nil {
		return nil, err
	}
	if m.WorkspaceRoot == "" {
		temp, err := os.MkdirTemp("", ShellTempPrefix)
		if err != nil {
			return nil, err
		}
		m.WorkspaceRoot = temp
		m.OwnsWorkspace = true
	}
	abs, err := filepath.Abs(m.WorkspaceRoot)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(abs, 0o755); err != nil {
		return nil, err
	}
	m.WorkspaceRoot = abs
	tool, err := tools.NewFunc(
		m.ToolName,
		DefaultShellToolDescription,
		schema.Object(map[string]schema.Schema{
			"command": schema.String("The shell command to execute."),
			"restart": schema.Boolean("Whether to restart the shell session."),
		}),
		func(ctx context.Context, input map[string]any) (tools.Result, error) {
			command, _ := input["command"].(string)
			restart, _ := input["restart"].(bool)
			result, err := m.Run(ctx, command, restart)
			if err != nil {
				return tools.Result{}, err
			}
			return tools.Result{Content: result.Output, Metadata: map[string]any{
				"exit_code":          result.ExitCode,
				"timed_out":          result.TimedOut,
				"truncated_by_lines": result.TruncatedByLines,
				"truncated_by_bytes": result.TruncatedByBytes,
				"total_lines":        result.TotalLines,
				"total_bytes":        result.TotalBytes,
			}}, nil
		},
	)
	if err != nil {
		return nil, err
	}
	m.Tools = []tools.Tool{tool}
	return m, nil
}

type ShellToolOption func(*ShellToolMiddleware)

func WithShellExecutionPolicy(policy ShellExecutionPolicy) ShellToolOption {
	return func(m *ShellToolMiddleware) {
		m.Policy = policy
	}
}

func WithShellRedactionRules(rules ...ResolvedRedactionRule) ShellToolOption {
	return func(m *ShellToolMiddleware) {
		m.RedactionRules = append([]ResolvedRedactionRule(nil), rules...)
	}
}

func WithShellEnv(env map[string]string) ShellToolOption {
	return func(m *ShellToolMiddleware) {
		m.Env = map[string]string{}
		for key, value := range env {
			m.Env[key] = value
		}
	}
}

func WithShellCommand(command ...string) ShellToolOption {
	return func(m *ShellToolMiddleware) {
		m.ShellCommand = append([]string(nil), command...)
	}
}

func WithShellStartupCommands(commands ...string) ShellToolOption {
	return func(m *ShellToolMiddleware) {
		m.StartupCommands = append([]string(nil), commands...)
	}
}

func WithShellShutdownCommands(commands ...string) ShellToolOption {
	return func(m *ShellToolMiddleware) {
		m.ShutdownCommands = append([]string(nil), commands...)
	}
}

func (m *ShellToolMiddleware) BeforeAgent(ctx context.Context, state map[string]any) (map[string]any, error) {
	resources, err := m.GetOrCreateResources(ctx, state)
	if err != nil {
		return nil, err
	}
	return map[string]any{ShellSessionResourcesKey: resources}, nil
}

func (m *ShellToolMiddleware) AfterAgent(ctx context.Context, state map[string]any) error {
	resources, ok := shellResourcesFromState(state)
	if !ok || resources.Closed {
		return nil
	}
	for _, command := range m.ShutdownCommands {
		_, _ = m.runInWorkspace(ctx, resources.WorkspaceRoot, command, m.Policy.CommandTimeout)
	}
	resources.ShutdownRan = len(m.ShutdownCommands) > 0
	resources.Closed = true
	if resources.OwnsWorkspace {
		_ = os.RemoveAll(resources.WorkspaceRoot)
	}
	return nil
}

func (m *ShellToolMiddleware) GetOrCreateResources(ctx context.Context, state map[string]any) (*ShellSessionResources, error) {
	if resources, ok := shellResourcesFromState(state); ok && !resources.Closed {
		return resources, nil
	}
	resources := &ShellSessionResources{
		WorkspaceRoot: m.WorkspaceRoot,
		OwnsWorkspace: m.OwnsWorkspace,
	}
	for _, command := range m.StartupCommands {
		result, err := m.runInWorkspace(ctx, resources.WorkspaceRoot, command, m.Policy.StartupTimeout)
		if err != nil {
			return nil, fmt.Errorf("startup command %q failed: %w", command, err)
		}
		if result.TimedOut || result.ExitCode != 0 {
			return nil, fmt.Errorf("startup command %q failed with exit code %d", command, result.ExitCode)
		}
	}
	resources.StartupRan = len(m.StartupCommands) > 0
	if state != nil {
		state[ShellSessionResourcesKey] = resources
	}
	return resources, nil
}

func (m *ShellToolMiddleware) RunWithState(ctx context.Context, state map[string]any, command string, restart bool) (CommandExecutionResult, error) {
	resources, err := m.GetOrCreateResources(ctx, state)
	if err != nil {
		return CommandExecutionResult{}, err
	}
	if restart {
		resources.StartupRan = false
		for _, command := range m.StartupCommands {
			result, err := m.runInWorkspace(ctx, resources.WorkspaceRoot, command, m.Policy.StartupTimeout)
			if err != nil {
				return CommandExecutionResult{}, fmt.Errorf("startup command %q failed: %w", command, err)
			}
			if result.TimedOut || result.ExitCode != 0 {
				return CommandExecutionResult{}, fmt.Errorf("startup command %q failed with exit code %d", command, result.ExitCode)
			}
		}
		resources.StartupRan = len(m.StartupCommands) > 0
		return CommandExecutionResult{Output: "Shell session restarted.", ExitCode: 0}, nil
	}
	return m.run(ctx, resources.WorkspaceRoot, command, false)
}

func (m *ShellToolMiddleware) Run(ctx context.Context, command string, restart bool) (CommandExecutionResult, error) {
	return m.run(ctx, m.WorkspaceRoot, command, restart)
}

func (m *ShellToolMiddleware) run(ctx context.Context, workspace string, command string, restart bool) (CommandExecutionResult, error) {
	if command == "" && !restart {
		return CommandExecutionResult{}, fmt.Errorf("shell tool requires either 'command' or 'restart'")
	}
	if command != "" && restart {
		return CommandExecutionResult{}, fmt.Errorf("specify only one of 'command' or 'restart'")
	}
	if restart {
		return CommandExecutionResult{Output: "Shell session restarted.", ExitCode: 0}, nil
	}
	return m.runInWorkspace(ctx, workspace, command, m.Policy.CommandTimeout)
}

func (m *ShellToolMiddleware) runInWorkspace(ctx context.Context, workspace string, command string, timeout time.Duration) (CommandExecutionResult, error) {
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	args := append([]string(nil), m.ShellCommand...)
	if len(args) == 1 {
		args = append(args, "-c")
	}
	args = append(args, command)
	cmd := exec.CommandContext(runCtx, args[0], args[1:]...)
	cmd.Dir = workspace
	cmd.Env = os.Environ()
	for key, value := range m.Env {
		cmd.Env = append(cmd.Env, key+"="+value)
	}
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	timedOut := runCtx.Err() == context.DeadlineExceeded
	exitCode := 0
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			exitCode = exitErr.ExitCode()
		} else if !timedOut {
			return CommandExecutionResult{}, err
		}
	}
	output := stdout.String()
	if stderr.Len() > 0 {
		if output != "" && !strings.HasSuffix(output, "\n") {
			output += "\n"
		}
		output += stderr.String()
	}
	output, err = m.applyRedaction(output)
	if err != nil {
		return CommandExecutionResult{}, err
	}
	result := truncateShellOutput(output, m.Policy)
	result.ExitCode = exitCode
	result.TimedOut = timedOut
	if timedOut && result.Output == "" {
		result.Output = "Command timed out."
	}
	return result, nil
}

func shellResourcesFromState(state map[string]any) (*ShellSessionResources, bool) {
	if state == nil {
		return nil, false
	}
	resources, ok := state[ShellSessionResourcesKey].(*ShellSessionResources)
	return resources, ok
}

func (m *ShellToolMiddleware) applyRedaction(output string) (string, error) {
	var err error
	for _, rule := range m.RedactionRules {
		updated, _, applyErr := rule.Apply(output)
		if applyErr != nil {
			err = applyErr
			break
		}
		output = updated
	}
	return output, err
}

func truncateShellOutput(output string, policy ShellExecutionPolicy) CommandExecutionResult {
	totalBytes := len([]byte(output))
	lines := strings.SplitAfter(output, "\n")
	totalLines := len(lines)
	if output == "" {
		totalLines = 0
		lines = nil
	}
	truncatedByLines := false
	if policy.MaxOutputLines > 0 && len(lines) > policy.MaxOutputLines {
		lines = lines[:policy.MaxOutputLines]
		truncatedByLines = true
		output = strings.Join(lines, "")
	}
	truncatedByBytes := false
	if policy.MaxOutputBytes > 0 && len([]byte(output)) > policy.MaxOutputBytes {
		output = string([]byte(output)[:policy.MaxOutputBytes])
		truncatedByBytes = true
	}
	return CommandExecutionResult{
		Output:           output,
		TruncatedByLines: truncatedByLines,
		TruncatedByBytes: truncatedByBytes,
		TotalLines:       totalLines,
		TotalBytes:       totalBytes,
	}
}

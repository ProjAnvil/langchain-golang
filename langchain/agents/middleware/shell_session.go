package middleware

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"
)

// ShellSession is a long-lived shell subprocess with persistent working
// directory and environment, mirroring Python's `ShellSession`
// (langchain.agents.middleware.shell_tool.ShellSession). Commands are sent to
// the shell's stdin; a unique done-marker printed to stdout delimits each
// command's output and carries its exit status.
//
// The persistent session merges the shell's stdout and stderr into a single
// pipe (the equivalent of `2>&1`), so a single ordered reader captures both
// streams and the done-marker protocol delimits each command's full output.
type ShellSession struct {
	mu sync.Mutex

	cmd        *exec.Cmd
	stdin      io.WriteCloser
	stdout     *bufio.Reader
	stdoutPipe *os.File

	workspace    string
	shell        []string
	env          map[string]string
	started      bool
	needsRestart bool
	markerSeq    int
}

// NewShellSession builds a session around a long-lived shell argv (e.g.
// ["/bin/sh"]) WITHOUT a trailing "-c", so the shell reads commands from
// stdin and keeps cwd/env state between calls.
func NewShellSession(workspace string, shell []string, env map[string]string) *ShellSession {
	// Strip a trailing "-c" if the caller reused the per-command shell spec.
	argv := append([]string(nil), shell...)
	for i, a := range argv {
		if a == "-c" {
			argv = append(argv[:i], argv[i+1:]...)
			break
		}
	}
	if len(argv) == 0 {
		argv = []string{"/bin/sh"}
	}
	envCopy := map[string]string{}
	for k, v := range env {
		envCopy[k] = v
	}
	return &ShellSession{workspace: workspace, shell: argv, env: envCopy}
}

// Start launches the shell subprocess if it is not already running.
func (s *ShellSession) Start(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.startLocked(ctx)
}

func (s *ShellSession) startLocked(ctx context.Context) error {
	if s.started {
		return nil
	}
	_ = ctx // ctx is reserved for future cancellation plumbing.

	cmd := exec.Command(s.shell[0], s.shell[1:]...)
	cmd.Dir = s.workspace
	cmd.Env = os.Environ()
	for k, v := range s.env {
		cmd.Env = append(cmd.Env, k+"="+v)
	}

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	// Merge stdout and stderr into one pipe (2>&1) so a single ordered reader
	// captures both streams and the marker always delimits the full output.
	pr, pw, err := os.Pipe()
	if err != nil {
		return err
	}
	cmd.Stdout = pw
	cmd.Stderr = pw
	if err := cmd.Start(); err != nil {
		_ = pw.Close()
		_ = pr.Close()
		return err
	}
	// The child holds a copy of pw; close ours so the reader sees EOF when the
	// child exits.
	_ = pw.Close()

	s.cmd = cmd
	s.stdin = stdin
	s.stdoutPipe = pr
	s.stdout = bufio.NewReader(pr)
	s.started = true
	s.needsRestart = false
	return nil
}

// Execute runs command in the persistent shell and returns its combined
// output. The shell keeps its working directory and environment across calls.
// A timed-out or canceled command kills and restarts the session (mirroring
// Python's restart-on-timeout): the marker reader goroutine may still be
// blocked on the shared pipe, so the session must be torn down before
// another Execute can safely read again.
func (s *ShellSession) Execute(ctx context.Context, command string, timeout time.Duration) (CommandExecutionResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.started || s.cmd == nil {
		if !s.needsRestart {
			return CommandExecutionResult{}, fmt.Errorf("shell session is not running")
		}
		// A previous timeout killed the session; restart it transparently so
		// callers can keep issuing commands.
		if err := s.startLocked(ctx); err != nil {
			return CommandExecutionResult{}, fmt.Errorf("restart shell session: %w", err)
		}
	}
	if timeout <= 0 {
		timeout = 30 * time.Second
	}

	marker := fmt.Sprintf("__LGSHELL_DONE_%d__", s.markerSeq)
	s.markerSeq++

	payload := command
	if !strings.HasSuffix(payload, "\n") {
		payload += "\n"
	}
	if _, err := io.WriteString(s.stdin, payload); err != nil {
		return CommandExecutionResult{}, fmt.Errorf("write command to shell: %w", err)
	}
	if _, err := fmt.Fprintf(s.stdin, "printf '%s %%s\\n' $?\n", marker); err != nil {
		return CommandExecutionResult{}, fmt.Errorf("write marker to shell: %w", err)
	}

	type outcome struct {
		output   string
		exitCode int
	}
	ch := make(chan outcome, 1)
	errCh := make(chan error, 1)
	// Capture the reader in a local: this goroutine must not touch session
	// fields after spawn, because timeoutLocked (on another goroutine) nils
	// them while tearing the session down. Closing s.stdoutPipe unblocks a
	// blocked ReadString; os.File supports concurrent Close and Read.
	reader := s.stdout
	go func() {
		var buf bytes.Buffer
		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				if buf.Len() > 0 {
					ch <- outcome{output: buf.String()}
				} else {
					errCh <- err
				}
				return
			}
			if strings.HasPrefix(line, marker) {
				exitCode := 0
				fields := strings.Fields(line)
				if len(fields) >= 2 {
					// A malformed status field degrades to 0, matching
					// Python's _safe_int default.
					_, _ = fmt.Sscanf(fields[1], "%d", &exitCode)
				}
				ch <- outcome{output: buf.String(), exitCode: exitCode}
				return
			}
			buf.WriteString(line)
		}
	}()

	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case r := <-ch:
		return CommandExecutionResult{Output: r.output, ExitCode: r.exitCode}, nil
	case err := <-errCh:
		return CommandExecutionResult{}, fmt.Errorf("read shell output: %w", err)
	case <-ctx.Done():
		s.timeoutLocked()
		return CommandExecutionResult{Output: "Command timed out.", TimedOut: true}, nil
	case <-timer.C:
		s.timeoutLocked()
		return CommandExecutionResult{Output: "Command timed out.", TimedOut: true}, nil
	}
}

// timeoutLocked tears down the shell after a timed-out or canceled command.
// The session may have consumed arbitrary output before its marker arrived,
// so its state is unsalvageable. Killing the process closes the pipe, which
// unblocks the marker-reader goroutine (os.File supports concurrent Close
// and Read); closing the read end as well guarantees it. Execute restarts
// the session lazily on the next call, like Python's restart-on-timeout.
func (s *ShellSession) timeoutLocked() {
	if s.cmd == nil {
		return
	}
	cmd := s.cmd
	if s.stdin != nil {
		_ = s.stdin.Close()
	}
	_ = cmd.Process.Kill()
	go func() { _ = cmd.Wait() }() // reap the killed process
	if s.stdoutPipe != nil {
		_ = s.stdoutPipe.Close()
	}
	s.cmd = nil
	s.stdin = nil
	s.stdout = nil
	s.stdoutPipe = nil
	s.started = false
	s.needsRestart = true
}

// Stop terminates the shell subprocess.
func (s *ShellSession) Stop(timeout time.Duration) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.started || s.cmd == nil {
		return nil
	}
	if s.stdin != nil {
		_, _ = io.WriteString(s.stdin, "exit\n")
	}
	done := make(chan struct{})
	cmd := s.cmd
	go func() {
		_ = cmd.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(timeout):
		_ = cmd.Process.Kill()
	}
	if s.stdin != nil {
		_ = s.stdin.Close()
	}
	if s.stdoutPipe != nil {
		_ = s.stdoutPipe.Close()
	}
	s.started = false
	s.cmd = nil
	s.stdin = nil
	s.stdout = nil
	s.stdoutPipe = nil
	s.needsRestart = false
	return nil
}

// Restart stops and restarts the shell subprocess.
func (s *ShellSession) Restart(ctx context.Context, timeout time.Duration) error {
	if err := s.Stop(timeout); err != nil {
		return err
	}
	return s.Start(ctx)
}

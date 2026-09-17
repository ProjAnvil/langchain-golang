package mcp

// Config-construction failure tests: the MCPConfig / ClientGroup guard
// clauses, backend-spec validation inside connectServer, and the member
// lifecycle errors (start / initialize / close) exercised through a stub
// Client so the failure injection is deterministic.

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	mcpclient "github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"
)

// fakeGroupClient is a ClientGroup member whose lifecycle calls fail on
// demand. Every other MCPClient method is inherited from the embedded nil
// interface and panics if called, keeping the stub honest about what the
// group actually touches.
type fakeGroupClient struct {
	mcpclient.MCPClient
	startErr error
	initErr  error
	listErr  error
	tools    []mcp.Tool
	closeErr error
}

func (f *fakeGroupClient) Start(context.Context) error { return f.startErr }

func (f *fakeGroupClient) Initialize(context.Context, mcp.InitializeRequest) (*mcp.InitializeResult, error) {
	if f.initErr != nil {
		return nil, f.initErr
	}
	return &mcp.InitializeResult{}, nil
}

func (f *fakeGroupClient) ListTools(context.Context, mcp.ListToolsRequest) (*mcp.ListToolsResult, error) {
	if f.listErr != nil {
		return nil, f.listErr
	}
	return &mcp.ListToolsResult{Tools: f.tools}, nil
}

func (f *fakeGroupClient) Close() error { return f.closeErr }

// bogusServerSpec is a Server spec this package does not know how to connect,
// proving the sealed default branch of connectServer rejects outsiders.
type bogusServerSpec struct{}

func (bogusServerSpec) isMCPServer() {}

// TestNewRejectsEmptyConfig: an MCPConfig without servers aborts.
func TestNewRejectsEmptyConfig(t *testing.T) {
	_, err := New(t.Context(), MCPConfig{})
	if err == nil || !strings.Contains(err.Error(), "no servers") {
		t.Fatalf("New(empty) err = %v, want no-servers error", err)
	}
}

// TestNewRejectsEmptyServerKey: a fleet entry keyed "" aborts construction.
func TestNewRejectsEmptyServerKey(t *testing.T) {
	_, err := New(t.Context(), MCPConfig{Servers: map[string]Server{
		"": &InProcessServer{Server: newTestMCPServer("x")},
	}})
	if err == nil || !strings.Contains(err.Error(), "key must not be empty") {
		t.Fatalf("New(empty key) err = %v, want empty-key error", err)
	}
}

// TestNewRejectsUnsupportedServerSpec: an unknown Server implementation is
// refused by the sealed interface switch.
func TestNewRejectsUnsupportedServerSpec(t *testing.T) {
	_, err := New(t.Context(), MCPConfig{Servers: map[string]Server{
		"weird": bogusServerSpec{},
	}})
	if err == nil || !strings.Contains(err.Error(), "unsupported server spec") {
		t.Fatalf("New(bogus spec) err = %v, want unsupported-spec error", err)
	}
}

// TestNewRejectsEmptyHTTPURL: an HTTP backend without a URL aborts before any
// connection attempt.
func TestNewRejectsEmptyHTTPURL(t *testing.T) {
	_, err := New(t.Context(), MCPConfig{Servers: map[string]Server{
		"web": &HTTPServer{URL: ""},
	}})
	if err == nil || !strings.Contains(err.Error(), "HTTP URL must not be empty") {
		t.Fatalf("New(empty URL) err = %v, want empty-URL error", err)
	}
}

// TestNewRejectsMalformedHTTPURL: a URL the HTTP transport cannot parse fails
// at client construction.
func TestNewRejectsMalformedHTTPURL(t *testing.T) {
	_, err := New(t.Context(), MCPConfig{Servers: map[string]Server{
		"web": &HTTPServer{URL: "http://\x7finvalid"}, // control character: url.Parse rejects
	}})
	if err == nil || !strings.Contains(err.Error(), "http client") {
		t.Fatalf("New(malformed URL) err = %v, want http client error", err)
	}
}

// TestNewRejectsNilInProcessServer: the in-process backend requires the
// server value.
func TestNewRejectsNilInProcessServer(t *testing.T) {
	_, err := New(t.Context(), MCPConfig{Servers: map[string]Server{
		"local": &InProcessServer{},
	}})
	if err == nil || !strings.Contains(err.Error(), "in-process server must not be nil") {
		t.Fatalf("New(nil in-process) err = %v, want nil-server error", err)
	}
}

// TestNewHTTPInitializeFailure: a streamable-HTTP backend whose endpoint went
// away fails the initialize handshake (Start is lazy for HTTP transports).
func TestNewHTTPInitializeFailure(t *testing.T) {
	ts := httptest.NewServer(http.NotFoundHandler())
	url := ts.URL + "/mcp"
	ts.Close() // the endpoint is gone before the fleet connects

	_, err := New(t.Context(), MCPConfig{Servers: map[string]Server{
		"dead": &HTTPServer{URL: url},
	}})
	if err == nil || !strings.Contains(err.Error(), "initialize server") {
		t.Fatalf("New(dead endpoint) err = %v, want initialize error", err)
	}
}

// TestNewClientGroupRequiresClients: a nil member map aborts.
func TestNewClientGroupRequiresClients(t *testing.T) {
	_, err := NewClientGroup(t.Context(), nil)
	if err == nil || !strings.Contains(err.Error(), "requires clients") {
		t.Fatalf("NewClientGroup(nil) err = %v, want requires-clients error", err)
	}
}

// TestNewClientGroupRejectsEmptyKey: a member keyed "" aborts before any
// member starts.
func TestNewClientGroupRejectsEmptyKey(t *testing.T) {
	_, err := NewClientGroup(t.Context(), map[string]Client{
		"": &fakeGroupClient{},
	})
	if err == nil || !strings.Contains(err.Error(), "key must not be empty") {
		t.Fatalf("NewClientGroup(empty key) err = %v, want empty-key error", err)
	}
}

// TestNewClientGroupRejectsNilMember: a nil member aborts with its key named.
func TestNewClientGroupRejectsNilMember(t *testing.T) {
	_, err := NewClientGroup(t.Context(), map[string]Client{
		"hole": nil,
	})
	if err == nil || !strings.Contains(err.Error(), `member "hole" is nil`) {
		t.Fatalf("NewClientGroup(nil member) err = %v, want nil-member error", err)
	}
}

// TestNewClientGroupStartFailure: a member that fails Start aborts the group
// with the start error wrapping the transport failure.
func TestNewClientGroupStartFailure(t *testing.T) {
	boom := errors.New("transport exploded")
	_, err := NewClientGroup(t.Context(), map[string]Client{
		"bad": &fakeGroupClient{startErr: boom},
	})
	if err == nil || !strings.Contains(err.Error(), `start server "bad"`) || !errors.Is(err, boom) {
		t.Fatalf("NewClientGroup(start failure) err = %v, want wrapped start error", err)
	}
}

// TestNewClientGroupInitializeFailure: a member that starts but fails the
// initialize handshake aborts the group; the already-started member is closed
// on the way out (closeMembers).
func TestNewClientGroupInitializeFailure(t *testing.T) {
	boom := errors.New("handshake refused")
	_, err := NewClientGroup(t.Context(), map[string]Client{
		"bad": &fakeGroupClient{initErr: boom},
	})
	if err == nil || !strings.Contains(err.Error(), `initialize server "bad"`) || !errors.Is(err, boom) {
		t.Fatalf("NewClientGroup(init failure) err = %v, want wrapped initialize error", err)
	}
}

// TestNewClientGroupOptions: nil options are skipped and WithoutElicitation
// turns the elicitation bridge off — the group still lists tools, and the
// initialize handshake advertises no elicitation capability.
func TestNewClientGroupOptions(t *testing.T) {
	var nilOpt ClientGroupOption
	group, err := NewClientGroup(t.Context(), map[string]Client{
		"srv": &fakeGroupClient{tools: []mcp.Tool{mcp.NewTool("plain")}},
	}, nilOpt, WithoutElicitation())
	if err != nil {
		t.Fatalf("NewClientGroup() error = %v", err)
	}
	defer group.Close()

	if group.elicit {
		t.Fatal("WithoutElicitation() must clear the elicitation flag")
	}
	tools, err := group.LoadTools(t.Context())
	if err != nil {
		t.Fatalf("LoadTools() error = %v", err)
	}
	if names := toolNamesOf(tools); len(names) != 1 || names[0] != "srv_plain" {
		t.Fatalf("LoadTools() names = %v, want [srv_plain]", names)
	}
}

// TestClientGroupCloseError: the first member Close error is surfaced (with
// its key), later members still close.
func TestClientGroupCloseError(t *testing.T) {
	boom := errors.New("close refused")
	group, err := NewClientGroup(t.Context(), map[string]Client{
		"graceful": &fakeGroupClient{},
		"refuses":  &fakeGroupClient{closeErr: boom},
	})
	if err != nil {
		t.Fatalf("NewClientGroup() error = %v", err)
	}
	err = group.Close()
	if err == nil || !strings.Contains(err.Error(), `close server "refuses"`) || !errors.Is(err, boom) {
		t.Fatalf("Close() err = %v, want wrapped close error", err)
	}
}

// TestClientNamesSorted: the member keys come back sorted regardless of map
// iteration order.
func TestClientNamesSorted(t *testing.T) {
	group, err := NewClientGroup(t.Context(), map[string]Client{
		"zeta":  &fakeGroupClient{},
		"alpha": &fakeGroupClient{},
		"mid":   &fakeGroupClient{},
	})
	if err != nil {
		t.Fatalf("NewClientGroup() error = %v", err)
	}
	defer group.Close()
	if got := group.ClientNames(); fmt.Sprint(got) != "[alpha mid zeta]" {
		t.Fatalf("ClientNames() = %v, want [alpha mid zeta]", got)
	}
}

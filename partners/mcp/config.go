package mcp

import (
	"context"
	"fmt"
	"maps"
	"slices"

	mcpclient "github.com/mark3labs/mcp-go/client"
	mcptransport "github.com/mark3labs/mcp-go/client/transport"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
	coretools "github.com/projanvil/langchain-golang/core/tools"
)

// Server is one backend of an MCPConfig fleet. Implemented by StdioServer,
// HTTPServer, and InProcessServer; the sealed method keeps the set open only
// to specs this package knows how to connect.
type Server interface {
	isMCPServer()
}

// StdioServer launches an MCP server as a subprocess speaking MCP over
// stdio, mirroring the {"command", "args"} config entries of the upstream
// MCPConfig dict.
type StdioServer struct {
	Command string
	Args    []string
	Env     []string
}

func (*StdioServer) isMCPServer() {}

// HTTPServer connects to a streamable-HTTP MCP endpoint, mirroring the
// {"url"} config entries of the upstream MCPConfig dict.
type HTTPServer struct {
	URL     string
	Headers map[string]string
}

func (*HTTPServer) isMCPServer() {}

// InProcessServer connects straight to an in-memory mcp-go server, the Go
// equivalent of passing an in-process FastMCP instance to the upstream
// MCPAdapter. It is also the connection used by the offline test suite.
type InProcessServer struct {
	Server *server.MCPServer
}

func (*InProcessServer) isMCPServer() {}

// MCPConfig is the aggregated multi-server configuration ("mcpServers" dict
// upstream): New connects every backend and LoadTools namespaces each tool
// with its config key. The fleet mixes transports freely (stdio next to
// HTTP next to in-process).
type MCPConfig struct {
	Servers map[string]Server
}

// Client is a prebuilt, independently configured mcp-go connection for
// ClientGroup membership: the full client surface plus the Start lifecycle
// method, satisfied by *mcpclient.Client. Members keep their own protocol
// negotiation, auth, and handlers (upstream ClientGroup semantics).
type Client interface {
	mcpclient.MCPClient
	Start(ctx context.Context) error
}

// Adapter is the aggregated connection built from an MCPConfig. It owns the
// clients it constructed and closes them on Close.
type Adapter struct {
	group *ClientGroup
}

// New connects every server in cfg, initializing each connection, and
// returns the aggregate adapter. A backend that fails to start or complete
// its initialize handshake aborts the construction with that error.
//
// Upstream's aggregated fleet shares a single negotiated protocol era
// across every backend; the mcp-go clients underneath each negotiate
// independently, so this Go port documents per-connection negotiation
// instead (use one Adapter per server and concatenate tool lists when era
// pinning matters).
func New(ctx context.Context, cfg MCPConfig) (*Adapter, error) {
	if len(cfg.Servers) == 0 {
		return nil, fmt.Errorf("mcp: MCPConfig has no servers")
	}
	members := make(map[string]Client, len(cfg.Servers))
	for key, srv := range cfg.Servers {
		if key == "" {
			closeMembers(members)
			return nil, fmt.Errorf("mcp: MCPConfig server key must not be empty")
		}
		cli, err := connectServer(ctx, key, srv)
		if err != nil {
			// A fleet whose later backend fails must not leak the ones
			// already connected (stdio subprocesses, HTTP sessions).
			closeMembers(members)
			return nil, err
		}
		members[key] = cli
	}
	return &Adapter{group: &ClientGroup{members: members, elicit: true}}, nil
}

// LoadTools lists tools across every backend, namespaced {key}_{tool}.
func (a *Adapter) LoadTools(ctx context.Context, opts ...LoadOption) ([]coretools.Tool, error) {
	return a.group.LoadTools(ctx, opts...)
}

// Close shuts down every backend connection the adapter built.
func (a *Adapter) Close() error {
	return a.group.Close()
}

// ClientGroup manages prebuilt, independent client connections. Each member
// keeps its own protocol negotiation, authentication, and handlers; the
// group routes each tool call back to the client that advertised the tool.
// LoadTools namespaces every tool {server}_{tool}, identical to the
// MCPConfig key-prefix scheme. Close closes every member.
type ClientGroup struct {
	members map[string]Client
	cache   toolsCache
	elicit  bool
}

// ClientGroupOption configures NewClientGroup.
type ClientGroupOption func(*ClientGroup)

// WithoutElicitation skips installing the interrupt-bridging elicitation
// handler on group members. By default (upstream parity) elicitation is on:
// the group arms every *mcpclient.Client member with the bridge, replacing
// any handler already set — mcp-go exposes no way to introspect an existing
// handler, so callers wanting their own elicitation handling build the group
// with this option and install it themselves.
func WithoutElicitation() ClientGroupOption {
	return func(g *ClientGroup) { g.elicit = false }
}

// NewClientGroup starts and initializes every member connection. Members
// must not already be initialized; the group owns their lifecycle from here.
func NewClientGroup(ctx context.Context, clients map[string]Client, opts ...ClientGroupOption) (*ClientGroup, error) {
	if clients == nil {
		return nil, fmt.Errorf("mcp: ClientGroup requires clients")
	}
	g := &ClientGroup{members: maps.Clone(clients), elicit: true}
	for _, opt := range opts {
		if opt != nil {
			opt(g)
		}
	}
	for key, cli := range g.members {
		if key == "" {
			return nil, fmt.Errorf("mcp: ClientGroup server key must not be empty")
		}
		if cli == nil {
			return nil, fmt.Errorf("mcp: ClientGroup member %q is nil", key)
		}
	}
	started := make(map[string]Client, len(g.members))
	for key, cli := range g.members {
		started[key] = cli
		if g.elicit {
			armElicitation(cli)
		}
		if err := cli.Start(ctx); err != nil {
			closeMembers(started)
			return nil, fmt.Errorf("mcp: start server %q: %w", key, err)
		}
		if _, err := cli.Initialize(ctx, initializeRequest(g.elicit)); err != nil {
			closeMembers(started)
			return nil, fmt.Errorf("mcp: initialize server %q: %w", key, err)
		}
	}
	return g, nil
}

// closeMembers closes every member connection, ignoring individual errors:
// it backs the constructor failure paths, where surfacing the original
// construction error matters more than the shutdown errors of the backends
// being torn down.
func closeMembers(members map[string]Client) {
	for _, cli := range members {
		_ = cli.Close()
	}
}

// ClientNames returns the member keys, sorted.
func (g *ClientGroup) ClientNames() []string {
	return slices.Sorted(maps.Keys(g.members))
}

// Close closes every member connection.
func (g *ClientGroup) Close() error {
	var firstErr error
	for key, cli := range g.members {
		if err := cli.Close(); err != nil && firstErr == nil {
			firstErr = fmt.Errorf("mcp: close server %q: %w", key, err)
		}
	}
	return firstErr
}

// connectServer builds, arms, starts, and initializes a client for one
// server spec, with the elicitation bridge installed (upstream default).
// The returned Client is owned by the caller.
func connectServer(ctx context.Context, key string, srv Server) (Client, error) {
	var (
		cli *mcpclient.Client
		err error
	)
	switch s := srv.(type) {
	case *StdioServer:
		if s.Command == "" {
			return nil, fmt.Errorf("mcp: server %q: stdio command must not be empty", key)
		}
		cli, err = mcpclient.NewStdioMCPClient(s.Command, s.Env, s.Args...)
		if err != nil {
			return nil, fmt.Errorf("mcp: server %q: stdio client: %w", key, err)
		}
	case *HTTPServer:
		if s.URL == "" {
			return nil, fmt.Errorf("mcp: server %q: HTTP URL must not be empty", key)
		}
		headers := maps.Clone(s.Headers)
		cli, err = mcpclient.NewStreamableHttpClient(s.URL, mcptransport.WithHTTPHeaderFunc(
			func(context.Context) map[string]string { return headers }))
		if err != nil {
			return nil, fmt.Errorf("mcp: server %q: http client: %w", key, err)
		}
	case *InProcessServer:
		if s.Server == nil {
			return nil, fmt.Errorf("mcp: server %q: in-process server must not be nil", key)
		}
		cli, err = mcpclient.NewInProcessClient(s.Server)
		if err != nil {
			return nil, fmt.Errorf("mcp: server %q: in-process client: %w", key, err)
		}
	default:
		return nil, fmt.Errorf("mcp: server %q: unsupported server spec %T", key, srv)
	}
	armElicitation(cli)
	if err := cli.Start(ctx); err != nil {
		return nil, fmt.Errorf("mcp: start server %q: %w", key, err)
	}
	if _, err := cli.Initialize(ctx, initializeRequest(true)); err != nil {
		return nil, fmt.Errorf("mcp: initialize server %q: %w", key, err)
	}
	return cli, nil
}

// armElicitation installs the interrupt-bridging elicitation handler on a
// concrete client. Only the in-process constructor accepts ClientOptions
// directly; for stdio/HTTP clients the option (a plain func(*Client)) is
// applied post-construction.
func armElicitation(cli Client) {
	if c, ok := cli.(*mcpclient.Client); ok {
		mcpclient.WithElicitationHandler(elicitationBridge{})(c)
	}
}

// initializeRequest builds the MCP initialize handshake, advertising the
// elicitation capability when the bridge is armed (upstream default).
func initializeRequest(elicit bool) mcp.InitializeRequest {
	req := mcp.InitializeRequest{}
	req.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	req.Params.ClientInfo = mcp.Implementation{Name: "langchain-golang", Version: "0.9.1"}
	if elicit {
		req.Params.Capabilities.Elicitation = &mcp.ElicitationCapability{}
	}
	return req
}

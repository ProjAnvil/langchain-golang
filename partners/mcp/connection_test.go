package mcp

// Connection-mechanism tests: the MCPConfig aggregated connection (in-process
// and streamable-HTTP backends), the ClientGroup over prebuilt independent
// clients, {server}_{tool} namespacing, and the list_tools cache modes.

import (
	"context"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
	coretools "github.com/projanvil/langchain-golang/core/tools"
)

func toolNamesOf(tools []coretools.Tool) []string {
	names := make([]string, 0, len(tools))
	for _, t := range tools {
		names = append(names, t.Name())
	}
	return names
}

// TestClientGroupLoadToolsNamespaces: a ClientGroup over two prebuilt
// in-process clients discovers tools from both and namespaces them as
// {server}_{tool}, so identical remote tool names stay distinguishable.
func TestClientGroupLoadToolsNamespaces(t *testing.T) {
	group, err := NewClientGroup(t.Context(), map[string]Client{
		"weather": newInProcessClient(t, newTestMCPServer("weather")),
		"calc":    newInProcessClient(t, newTestMCPServer("calc")),
	})
	if err != nil {
		t.Fatalf("NewClientGroup() error = %v", err)
	}
	defer group.Close()

	tools, err := group.LoadTools(t.Context())
	if err != nil {
		t.Fatalf("LoadTools() error = %v", err)
	}
	names := toolNamesOf(tools)
	for _, want := range []string{"weather_echo", "calc_echo", "weather_failing", "calc_failing"} {
		if !slices.Contains(names, want) {
			t.Fatalf("LoadTools() names = %v, want %q among them", names, want)
		}
	}
}

// TestClientGroupProvenance: loaded tools expose the server key and the
// remote (un-prefixed) name.
func TestClientGroupProvenance(t *testing.T) {
	group, err := NewClientGroup(t.Context(), map[string]Client{
		"weather": newInProcessClient(t, newTestMCPServer("weather")),
	})
	if err != nil {
		t.Fatalf("NewClientGroup() error = %v", err)
	}
	defer group.Close()

	tools, err := group.LoadTools(t.Context())
	if err != nil {
		t.Fatalf("LoadTools() error = %v", err)
	}
	var echo *Tool
	for _, tl := range tools {
		if mt, ok := tl.(*Tool); ok && tl.Name() == "weather_echo" {
			echo = mt
		}
	}
	if echo == nil {
		t.Fatalf("weather_echo not loaded: %v", toolNamesOf(tools))
	}
	if echo.ServerName() != "weather" {
		t.Fatalf("ServerName() = %q, want weather", echo.ServerName())
	}
	if echo.RemoteName() != "echo" {
		t.Fatalf("RemoteName() = %q, want echo", echo.RemoteName())
	}
	if echo.Description() != "echo the input" {
		t.Fatalf("Description() = %q", echo.Description())
	}
}

// TestMCPConfigAggregatedInProcess: New(ctx, MCPConfig) connects an
// aggregated fleet (here a single in-process backend) and namespaces tools
// by config key.
func TestMCPConfigAggregatedInProcess(t *testing.T) {
	adapter, err := New(t.Context(), MCPConfig{
		Servers: map[string]Server{
			"local": &InProcessServer{Server: newTestMCPServer("local")},
		},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	defer adapter.Close()

	tools, err := adapter.LoadTools(t.Context())
	if err != nil {
		t.Fatalf("LoadTools() error = %v", err)
	}
	if !slices.Contains(toolNamesOf(tools), "local_echo") {
		t.Fatalf("LoadTools() names = %v, want local_echo", toolNamesOf(tools))
	}

	// The tool is callable end to end.
	for _, tl := range tools {
		if tl.Name() == "local_echo" {
			res, err := tl.Invoke(t.Context(), map[string]any{"text": "hi"})
			if err != nil {
				t.Fatalf("Invoke() error = %v", err)
			}
			if res.Content != "echo: hi" {
				t.Fatalf("Invoke() content = %q, want %q", res.Content, "echo: hi")
			}
		}
	}
}

// TestMCPConfigAggregatedHTTP: the aggregated connection also covers
// streamable-HTTP backends, served offline by httptest.
func TestMCPConfigAggregatedHTTP(t *testing.T) {
	ts := newStreamableHTTPServer(t, newTestMCPServer("remote"))
	adapter, err := New(t.Context(), MCPConfig{
		Servers: map[string]Server{
			"remote": &HTTPServer{URL: ts.URL + "/mcp"},
		},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	defer adapter.Close()

	tools, err := adapter.LoadTools(t.Context())
	if err != nil {
		t.Fatalf("LoadTools() error = %v", err)
	}
	if !slices.Contains(toolNamesOf(tools), "remote_echo") {
		t.Fatalf("LoadTools() names = %v, want remote_echo", toolNamesOf(tools))
	}
	for _, tl := range tools {
		if tl.Name() == "remote_echo" {
			res, err := tl.Invoke(t.Context(), map[string]any{"text": "http"})
			if err != nil {
				t.Fatalf("Invoke() error = %v", err)
			}
			if res.Content != "echo: http" {
				t.Fatalf("Invoke() content = %q", res.Content)
			}
		}
	}
}

// TestMCPConfigMixedFleet: one aggregate fleet mixing an in-process and an
// HTTP backend, both prefixed by their config keys.
func TestMCPConfigMixedFleet(t *testing.T) {
	ts := newStreamableHTTPServer(t, newTestMCPServer("calc"))
	adapter, err := New(t.Context(), MCPConfig{
		Servers: map[string]Server{
			"local": &InProcessServer{Server: newTestMCPServer("local")},
			"calc":  &HTTPServer{URL: ts.URL + "/mcp"},
		},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	defer adapter.Close()

	tools, err := adapter.LoadTools(t.Context())
	if err != nil {
		t.Fatalf("LoadTools() error = %v", err)
	}
	for _, want := range []string{"local_echo", "calc_echo"} {
		if !slices.Contains(toolNamesOf(tools), want) {
			t.Fatalf("LoadTools() names = %v, want %q", toolNamesOf(tools), want)
		}
	}
}

// TestMCPConfigInvalidServer: an empty stdio command fails at construction.
func TestMCPConfigInvalidServer(t *testing.T) {
	if _, err := New(t.Context(), MCPConfig{Servers: map[string]Server{
		"bad": &StdioServer{Command: ""},
	}}); err == nil {
		t.Fatal("New() with empty command must fail")
	}
}

// TestStdioConnectionRefused: a stdio backend pointing at a nonexistent
// command fails at construction with the transport error raised.
func TestStdioConnectionRefused(t *testing.T) {
	_, err := New(t.Context(), MCPConfig{Servers: map[string]Server{
		"gone": &StdioServer{Command: "/nonexistent-mcp-server-xyz"},
	}})
	if err == nil {
		t.Fatal("New() with nonexistent command must fail")
	}
}

// TestLoadToolsCacheModes: cache_mode use serves the cached listing,
// refresh re-fetches and updates the cache, bypass fetches without updating.
func TestLoadToolsCacheModes(t *testing.T) {
	srv := newTestMCPServer("srv")
	group, err := NewClientGroup(t.Context(), map[string]Client{
		"srv": newInProcessClient(t, srv),
	})
	if err != nil {
		t.Fatalf("NewClientGroup() error = %v", err)
	}
	defer group.Close()

	tools, err := group.LoadTools(t.Context()) // default: use
	if err != nil {
		t.Fatalf("LoadTools() error = %v", err)
	}
	if slices.Contains(toolNamesOf(tools), "srv_late") {
		t.Fatal("srv_late not yet registered on the server")
	}

	// Register a new tool server-side; cache_mode=use keeps the old listing.
	srv.AddTool(mcp.NewTool("late", mcp.WithDescription("added later")),
		func(_ context.Context, _ mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			return mcp.NewToolResultText("late"), nil
		})
	tools, err = group.LoadTools(t.Context(), WithCacheMode(CacheModeUse))
	if err != nil {
		t.Fatalf("LoadTools(use) error = %v", err)
	}
	if slices.Contains(toolNamesOf(tools), "srv_late") {
		t.Fatal("cache_mode=use must serve the cached listing")
	}

	// refresh re-fetches and updates the cache.
	tools, err = group.LoadTools(t.Context(), WithCacheMode(CacheModeRefresh))
	if err != nil {
		t.Fatalf("LoadTools(refresh) error = %v", err)
	}
	if !slices.Contains(toolNamesOf(tools), "srv_late") {
		t.Fatal("cache_mode=refresh must see the new tool")
	}
	tools, err = group.LoadTools(t.Context()) // use now sees refresh's cache
	if err != nil {
		t.Fatalf("LoadTools() error = %v", err)
	}
	if !slices.Contains(toolNamesOf(tools), "srv_late") {
		t.Fatal("refresh must update the cache for later use")
	}

	// bypass fetches fresh without writing the cache.
	srv.AddTool(mcp.NewTool("late2", mcp.WithDescription("added later still")),
		func(_ context.Context, _ mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			return mcp.NewToolResultText("late2"), nil
		})
	tools, err = group.LoadTools(t.Context(), WithCacheMode(CacheModeBypass))
	if err != nil {
		t.Fatalf("LoadTools(bypass) error = %v", err)
	}
	if !slices.Contains(toolNamesOf(tools), "srv_late2") {
		t.Fatal("cache_mode=bypass must fetch the live listing")
	}
	tools, err = group.LoadTools(t.Context()) // cache untouched by bypass
	if err != nil {
		t.Fatalf("LoadTools() error = %v", err)
	}
	if slices.Contains(toolNamesOf(tools), "srv_late2") {
		t.Fatal("cache_mode=bypass must not update the cache")
	}
}

// TestLoadToolsEmptyGroup: zero servers yield an empty tool list.
func TestLoadToolsEmptyGroup(t *testing.T) {
	group, err := NewClientGroup(t.Context(), map[string]Client{})
	if err != nil {
		t.Fatalf("NewClientGroup() error = %v", err)
	}
	defer group.Close()
	tools, err := group.LoadTools(t.Context())
	if err != nil {
		t.Fatalf("LoadTools() error = %v", err)
	}
	if len(tools) != 0 {
		t.Fatalf("LoadTools() = %d tools, want 0", len(tools))
	}
}

// newStreamableHTTPServer serves srv over a streamable-HTTP endpoint mounted
// at /mcp, offline via httptest.
func newStreamableHTTPServer(t *testing.T, srv *server.MCPServer) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.Handle("/mcp", server.NewStreamableHTTPServer(srv))
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)
	return ts
}

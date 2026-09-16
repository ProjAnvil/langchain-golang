package mcp

// Shared in-process MCP test server: registers one tool per result-mapping
// quadrant (text / structured / error / image), a destructive-annotation tool
// for the HITL gating tests, and an MRTR-elicitation tool. The same server
// backs the in-process connections and the streamable-HTTP httptest endpoint,
// so both transports exercise the identical wire behavior offline.

import (
	"context"
	"fmt"
	"testing"

	mcpclient "github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

type echoOut struct {
	Echo string `json:"echo"`
}

// newTestMCPServer builds the offline MCP server used by every test.
func newTestMCPServer(name string) *server.MCPServer {
	s := server.NewMCPServer(name, "1.0.0")

	s.AddTool(
		mcp.NewTool("echo",
			mcp.WithDescription("echo the input"),
			mcp.WithString("text", mcp.Required(), mcp.Description("text to echo")),
			mcp.WithToolAnnotation(mcp.ToolAnnotation{
				ReadOnlyHint:    new(true),
				DestructiveHint: new(false),
			}),
		),
		func(_ context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			text, err := req.RequireString("text")
			if err != nil {
				return mcp.NewToolResultError(err.Error()), nil
			}
			return mcp.NewToolResultText("echo: " + text), nil
		},
	)

	s.AddTool(
		mcp.NewTool("structured_echo",
			mcp.WithDescription("echo with structured output"),
			mcp.WithString("text", mcp.Required()),
			mcp.WithOutputSchema[echoOut](),
		),
		func(_ context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			text, _ := req.RequireString("text")
			return mcp.NewToolResultStructured(echoOut{Echo: text}, "structured fallback"), nil
		},
	)

	s.AddTool(
		mcp.NewTool("failing",
			mcp.WithDescription("always fails"),
		),
		func(_ context.Context, _ mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			return mcp.NewToolResultError("boom: bad input"), nil
		},
	)

	s.AddTool(
		mcp.NewTool("picture",
			mcp.WithDescription("returns text plus an image"),
		),
		func(_ context.Context, _ mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			return mcp.NewToolResultImage("a picture", "aW1hZ2VkYXRh", "image/png"), nil
		},
	)

	s.AddTool(
		mcp.NewTool("danger",
			mcp.WithDescription("destructive operation"),
			mcp.WithToolAnnotation(mcp.ToolAnnotation{DestructiveHint: new(true)}),
		),
		func(_ context.Context, _ mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			return mcp.NewToolResultText("deleted everything"), nil
		},
	)

	s.AddTool(
		mcp.NewTool("elicit",
			mcp.WithDescription("asks for input mid-call"),
		),
		func(_ context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			if resp, ok := req.Params.InputResponses["confirm"]; ok {
				if err := resp.DecodeFor(mcp.MethodElicitationCreate); err == nil && resp.Elicitation != nil {
					return mcp.NewToolResultText(fmt.Sprintf("action=%s content=%v state=%s",
						resp.Elicitation.Action, resp.Elicitation.Content, req.Params.RequestState)), nil
				}
			}
			res := &mcp.CallToolResult{}
			res.ResultType = mcp.ResultTypeInputRequired
			res.InputRequests = mcp.InputRequests{
				"confirm": mcp.NewElicitationInputRequest(mcp.ElicitationParams{
					Message:       "Please confirm the transfer",
					ElicitationID: "confirm",
					RequestedSchema: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"amount": map[string]any{"type": "number"},
						},
					},
				}),
			}
			res.RequestState = "elicit-state-1"
			return res, nil
		},
	)

	return s
}

// newInProcessClient builds a fresh in-process client for srv. The
// ClientGroup owns starting and initializing it.
func newInProcessClient(t *testing.T, srv *server.MCPServer) *mcpclient.Client {
	t.Helper()
	cli, err := mcpclient.NewInProcessClient(srv)
	if err != nil {
		t.Fatalf("NewInProcessClient() error = %v", err)
	}
	return cli
}

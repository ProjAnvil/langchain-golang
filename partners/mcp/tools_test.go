package mcp

// Result-mapping tests: the four quadrants of CallToolResult handling
// (content blocks / structured artifact / isError tool error / transport
// raise), input-schema conversion, and the ToolNode HandleToolErrors
// integration.

import (
	"errors"
	"slices"
	"strings"
	"testing"

	coremessages "github.com/projanvil/langchain-golang/core/messages"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	lctools "github.com/projanvil/langchain-golang/langchain/tools"
)

// loadOne loads the namespaced tool want from an in-process server.
func loadOne(t *testing.T, serverKey, want string) *Tool {
	t.Helper()
	group, err := NewClientGroup(t.Context(), map[string]Client{
		serverKey: newInProcessClient(t, newTestMCPServer(serverKey)),
	})
	if err != nil {
		t.Fatalf("NewClientGroup() error = %v", err)
	}
	t.Cleanup(func() { group.Close() })
	tools, err := group.LoadTools(t.Context())
	if err != nil {
		t.Fatalf("LoadTools() error = %v", err)
	}
	i := slices.IndexFunc(tools, func(tl coretools.Tool) bool { return tl.Name() == want })
	if i < 0 {
		t.Fatalf("tool %q not found among %v", want, toolNamesOf(tools))
	}
	return tools[i].(*Tool)
}

// TestMappingTextContent: text content lands in the model-visible Content.
func TestMappingTextContent(t *testing.T) {
	echo := loadOne(t, "srv", "srv_echo")
	res, err := echo.Invoke(t.Context(), map[string]any{"text": "hello"})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if res.Content != "echo: hello" {
		t.Fatalf("Content = %q, want %q", res.Content, "echo: hello")
	}
	if res.Artifact != nil {
		t.Fatalf("text-only result must carry no artifact, got %#v", res.Artifact)
	}
}

// TestMappingStructuredArtifact: structured output is attached as the
// ToolArtifact (not folded into model-visible content); the server's text
// fallback stays in Content.
func TestMappingStructuredArtifact(t *testing.T) {
	tool := loadOne(t, "srv", "srv_structured_echo")
	res, err := tool.Invoke(t.Context(), map[string]any{"text": "hi"})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if res.Content != "structured fallback" {
		t.Fatalf("Content = %q, want the server's text fallback", res.Content)
	}
	artifact, ok := res.Artifact.(*ToolArtifact)
	if !ok {
		t.Fatalf("Artifact = %T, want *ToolArtifact", res.Artifact)
	}
	echo, ok := artifact.StructuredContent.(map[string]any)
	if !ok {
		t.Fatalf("StructuredContent = %T, want map[string]any", artifact.StructuredContent)
	}
	if echo["echo"] != "hi" {
		t.Fatalf("StructuredContent[echo] = %v, want hi", echo["echo"])
	}
}

// TestMappingImageBlock: image content becomes a standardized image block
// carried on the artifact; text stays in Content.
func TestMappingImageBlock(t *testing.T) {
	tool := loadOne(t, "srv", "srv_picture")
	res, err := tool.Invoke(t.Context(), map[string]any{})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if res.Content != "a picture" {
		t.Fatalf("Content = %q, want %q", res.Content, "a picture")
	}
	artifact, ok := res.Artifact.(*ToolArtifact)
	if !ok {
		t.Fatalf("Artifact = %T, want *ToolArtifact", res.Artifact)
	}
	var img *coremessages.ImageBlock
	for _, block := range artifact.ContentBlocks {
		if b, ok := block.(coremessages.ImageBlock); ok {
			img = &b
			break
		}
	}
	if img == nil {
		t.Fatalf("no image block in %#v", artifact.ContentBlocks)
	}
	if img.Base64 != "aW1hZ2VkYXRh" || img.MimeType != "image/png" {
		t.Fatalf("image block = %+v", img)
	}
}

// TestMappingIsError: isError=true comes back as a *ToolError that preserves
// the server's own message for the error ToolMessage.
func TestMappingIsError(t *testing.T) {
	tool := loadOne(t, "srv", "srv_failing")
	_, err := tool.Invoke(t.Context(), map[string]any{})
	if err == nil {
		t.Fatal("isError result must surface as an error")
	}
	toolErr, ok := errors.AsType[*ToolError](err)
	if !ok {
		t.Fatalf("error = %T, want *ToolError", err)
	}
	if !strings.Contains(toolErr.Message, "boom: bad input") {
		t.Fatalf("ToolError.Message = %q, want server's message", toolErr.Message)
	}
}

// TestMappingTransportFailureRaised: a dead connection raises instead of
// producing a tool result (a model cannot act on a dropped connection).
// Uses a streamable-HTTP endpoint whose server goes away mid-connection —
// an in-process "closed" client still routes to the still-alive server, so
// it cannot produce a transport failure.
func TestMappingTransportFailureRaised(t *testing.T) {
	ts := newStreamableHTTPServer(t, newTestMCPServer("srv"))
	adapter, err := New(t.Context(), MCPConfig{
		Servers: map[string]Server{
			"srv": &HTTPServer{URL: ts.URL + "/mcp"},
		},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	tools, err := adapter.LoadTools(t.Context())
	if err != nil {
		t.Fatalf("LoadTools() error = %v", err)
	}
	ts.Close() // drop the endpoint; the connections are now dead
	i := slices.IndexFunc(tools, func(tl coretools.Tool) bool { return tl.Name() == "srv_echo" })
	if i < 0 {
		t.Fatal("srv_echo not found")
	}
	_, err = tools[i].Invoke(t.Context(), map[string]any{"text": "x"})
	if err == nil {
		t.Fatal("Invoke() on a dead connection must raise")
	}
	if _, ok := errors.AsType[*ToolError](err); ok {
		t.Fatalf("transport failure must not map to ToolError, got %v", err)
	}
}

// TestArgsSchemaConversion: the MCP input schema round-trips into the tool's
// ArgsSchema (type/properties/required).
func TestArgsSchemaConversion(t *testing.T) {
	echo := loadOne(t, "srv", "srv_echo")
	schema := echo.ArgsSchema()
	if schema["type"] != "object" {
		t.Fatalf("schema type = %v", schema["type"])
	}
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("properties = %T", schema["properties"])
	}
	textProp, ok := props["text"].(map[string]any)
	if !ok {
		t.Fatalf("text property = %T", props["text"])
	}
	if textProp["type"] != "string" {
		t.Fatalf("text type = %v", textProp["type"])
	}
	required, _ := schema["required"].([]any)
	if len(required) != 1 || required[0] != "text" {
		t.Fatalf("required = %#v, want [text]", schema["required"])
	}
}

// TestArgsSchemaPassesCoreValidation: the converted schema satisfies the core
// tool-schema contract (provider binding accepts it).
func TestArgsSchemaPassesCoreValidation(t *testing.T) {
	echo := loadOne(t, "srv", "srv_echo")
	if err := coretools.ValidateArgsSchema(echo.ArgsSchema()); err != nil {
		t.Fatalf("ValidateArgsSchema() error = %v", err)
	}
	if _, err := coretools.ToFunctionSpec(echo); err != nil {
		t.Fatalf("ToFunctionSpec() error = %v", err)
	}
}

// TestToolNodeErrorIntegration: MCP tools slot into langchain/tools.ToolNode;
// an isError result becomes an error ToolMessage through the default
// HandleToolErrors path.
func TestToolNodeErrorIntegration(t *testing.T) {
	failing := loadOne(t, "srv", "srv_failing")
	echo := loadOne(t, "srv", "srv_echo")

	node, err := lctools.NewToolNode([]coretools.Tool{echo, failing})
	if err != nil {
		t.Fatalf("NewToolNode() error = %v", err)
	}
	msgs, err := node.InvokeToolCalls(t.Context(), []coremessages.ToolCall{
		{ID: "1", Name: "srv_failing", Args: map[string]any{}},
		{ID: "2", Name: "srv_echo", Args: map[string]any{"text": "ok"}},
	}, nil)
	if err != nil {
		t.Fatalf("InvokeToolCalls() error = %v", err)
	}
	if len(msgs) != 2 {
		t.Fatalf("got %d messages, want 2", len(msgs))
	}
	failure := msgs[0]
	if failure.Role != coremessages.RoleTool {
		t.Fatalf("role = %v", failure.Role)
	}
	if failure.ResponseMetadata["status"] != "error" {
		t.Fatalf("error ToolMessage status = %#v", failure.ResponseMetadata)
	}
	if !strings.Contains(failure.Content, "boom: bad input") {
		t.Fatalf("error ToolMessage content = %q, want server message preserved", failure.Content)
	}
	if msgs[1].Content != "echo: ok" {
		t.Fatalf("success message = %q", msgs[1].Content)
	}
}

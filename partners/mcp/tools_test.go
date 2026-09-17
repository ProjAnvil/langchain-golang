package mcp

// Result-mapping tests: the four quadrants of CallToolResult handling
// (content blocks / structured artifact / isError tool error / transport
// raise), input-schema conversion, and the ToolNode HandleToolErrors
// integration.

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
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

// TestLoadToolsListToolsFailure: a member whose listing call fails aborts
// LoadTools with the wrapped server error.
func TestLoadToolsListToolsFailure(t *testing.T) {
	group, err := NewClientGroup(t.Context(), map[string]Client{
		"broken": &fakeGroupClient{listErr: errors.New("catalog exploded")},
	})
	if err != nil {
		t.Fatalf("NewClientGroup() error = %v", err)
	}
	defer group.Close()
	_, err = group.LoadTools(t.Context())
	if err == nil || !strings.Contains(err.Error(), `list tools on server "broken"`) {
		t.Fatalf("LoadTools() err = %v, want wrapped listing error", err)
	}
}

// TestLoadToolsMalformedToolSchema: a listed tool whose input schema cannot
// decode aborts the load naming the tool and server.
func TestLoadToolsMalformedToolSchema(t *testing.T) {
	group, err := NewClientGroup(t.Context(), map[string]Client{
		"srv": &fakeGroupClient{tools: []mcp.Tool{
			{Name: "corrupt", RawInputSchema: []byte("not-json")},
		}},
	})
	if err != nil {
		t.Fatalf("NewClientGroup() error = %v", err)
	}
	defer group.Close()
	_, err = group.LoadTools(t.Context())
	if err == nil || !strings.Contains(err.Error(), `tool "corrupt" on server "srv"`) {
		t.Fatalf("LoadTools() err = %v, want schema adaption error", err)
	}
}

// TestConvertInputSchemaVariants: the raw and typed schema paths round-trip,
// and the malformed shapes fail where the code can see them.
func TestConvertInputSchemaVariants(t *testing.T) {
	// Raw schema decodes as-is.
	raw, err := convertInputSchema(mcp.Tool{RawInputSchema: []byte(`{"type":"object","required":["a"]}`)})
	if err != nil {
		t.Fatalf("convertInputSchema(raw) error = %v", err)
	}
	if raw["type"] != "object" {
		t.Fatalf("raw schema = %#v", raw)
	}

	// Raw "null" yields an empty (non-nil) schema map.
	empty, err := convertInputSchema(mcp.Tool{RawInputSchema: []byte(`null`)})
	if err != nil {
		t.Fatalf("convertInputSchema(null) error = %v", err)
	}
	if empty == nil || len(empty) != 0 {
		t.Fatalf("null schema = %#v, want empty non-nil map", empty)
	}

	// Undecodable raw bytes fail.
	if _, err := convertInputSchema(mcp.Tool{RawInputSchema: []byte("{oops")}); err == nil {
		t.Fatal("convertInputSchema(bad raw) must fail")
	}

	// Typed schema without raw bytes marshals then decodes.
	typed, err := convertInputSchema(mcp.Tool{InputSchema: mcp.ToolInputSchema{
		Type:       "object",
		Properties: map[string]any{"q": map[string]any{"type": "string"}},
	}})
	if err != nil {
		t.Fatalf("convertInputSchema(typed) error = %v", err)
	}
	if typed["type"] != "object" {
		t.Fatalf("typed schema = %#v", typed)
	}

	// A typed schema carrying unmarshalable property values fails encoding.
	if _, err := convertInputSchema(mcp.Tool{InputSchema: mcp.ToolInputSchema{
		Properties: map[string]any{"bad": make(chan int)},
	}}); err == nil {
		t.Fatal("convertInputSchema(unmarshalable) must fail")
	}
}

// TestMapCallResultContentVariants: audio and embedded-resource contents map
// onto the standardized blocks (and malformed resource variants are skipped
// without failing the call).
func TestMapCallResultContentVariants(t *testing.T) {
	result := &mcp.CallToolResult{Content: []mcp.Content{
		mcp.TextContent{Text: "line one"},
		mcp.TextContent{Text: "line two"},
		mcp.AudioContent{Data: "YXVkaW8=", MIMEType: "audio/wav"},
		mcp.EmbeddedResource{Resource: &mcp.BlobResourceContents{
			Blob:     "ZmlsZQ==",
			MIMEType: "application/pdf",
		}},
		mcp.EmbeddedResource{Resource: &mcp.BlobResourceContents{Blob: "eA=="}}, // no MIME type: skipped
		mcp.EmbeddedResource{Resource: &mcp.TextResourceContents{URI: "file:///x", MIMEType: "text/plain"}},
	}}
	out, err := mapCallResult(&Tool{name: "srv_media"}, result)
	if err != nil {
		t.Fatalf("mapCallResult() error = %v", err)
	}
	if out.Content != "line one\nline two" {
		t.Fatalf("Content = %q, want joined text parts", out.Content)
	}
	artifact, ok := out.Artifact.(*ToolArtifact)
	if !ok {
		t.Fatalf("Artifact = %T, want *ToolArtifact", out.Artifact)
	}
	var audio coremessages.AudioBlock
	var file coremessages.FileBlock
	for _, block := range artifact.ContentBlocks {
		switch b := block.(type) {
		case coremessages.AudioBlock:
			audio = b
		case coremessages.FileBlock:
			file = b
		}
	}
	if audio.Base64 != "YXVkaW8=" || audio.MimeType != "audio/wav" {
		t.Fatalf("audio block = %+v", audio)
	}
	if file.Base64 != "ZmlsZQ==" || file.MimeType != "application/pdf" {
		t.Fatalf("file block = %+v", file)
	}
	if len(artifact.ContentBlocks) != 2 {
		t.Fatalf("content blocks = %#v, want audio+file only", artifact.ContentBlocks)
	}

	// A structured-only result still carries the artifact; a text-only result
	// with skipped resources carries none.
	structured := &mcp.CallToolResult{
		StructuredContent: map[string]any{"ok": true},
		Content: []mcp.Content{
			mcp.EmbeddedResource{Resource: &mcp.TextResourceContents{URI: "file:///y"}},
		},
	}
	out, err = mapCallResult(&Tool{name: "srv_structured"}, structured)
	if err != nil {
		t.Fatalf("mapCallResult(structured) error = %v", err)
	}
	artifact, ok = out.Artifact.(*ToolArtifact)
	if !ok || artifact.StructuredContent.(map[string]any)["ok"] != true {
		t.Fatalf("structured artifact = %#v", out.Artifact)
	}
}

// TestInvokeContextCanceled: canceling the invocation context while the
// server handler is still running surfaces context.Canceled from Invoke (the
// CallTool goroutine is released afterwards so nothing leaks).
func TestInvokeContextCanceled(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	srv := server.NewMCPServer("slow", "1.0.0")
	srv.AddTool(mcp.NewTool("stall"), func(context.Context, mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		close(entered)
		<-release // the handler outlives the caller's context on purpose
		return mcp.NewToolResultText("finally"), nil
	})

	group, err := NewClientGroup(t.Context(), map[string]Client{
		"slow": newInProcessClient(t, srv),
	})
	if err != nil {
		t.Fatalf("NewClientGroup() error = %v", err)
	}
	defer group.Close()

	tool, err := newTool(group.members["slow"], "slow", mcp.NewTool("stall"), true)
	if err != nil {
		t.Fatalf("newTool() error = %v", err)
	}

	ctx, cancel := context.WithCancel(t.Context())
	invokeErr := make(chan error, 1)
	go func() {
		_, err := tool.Invoke(ctx, nil)
		invokeErr <- err
	}()
	<-entered
	cancel()
	if err := <-invokeErr; !errors.Is(err, context.Canceled) {
		t.Fatalf("Invoke(canceled) err = %v, want context.Canceled", err)
	}
	close(release) // let the in-flight CallTool goroutine finish
}

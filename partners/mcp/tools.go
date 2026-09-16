package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/mark3labs/mcp-go/mcp"
	coremessages "github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langgraph/graph"
)

// CacheMode controls how LoadTools consults the group's tool listing cache,
// mirroring the upstream list_tools cache_mode parameter. The Go port caches
// by presence (mcp-go surfaces no server TTL hints); upstream's TTL-honoring
// "use" therefore degrades to "serve the cached listing until refreshed".
type CacheMode string

const (
	// CacheModeUse serves the cached listing when present and fetches
	// otherwise (the default).
	CacheModeUse CacheMode = "use"
	// CacheModeRefresh re-fetches the listing and updates the cache.
	CacheModeRefresh CacheMode = "refresh"
	// CacheModeBypass fetches the live listing without reading or updating
	// the cache.
	CacheModeBypass CacheMode = "bypass"
)

// LoadOption configures LoadTools.
type LoadOption func(*loadConfig)

type loadConfig struct {
	cacheMode CacheMode
}

// WithCacheMode overrides the list_tools cache mode (default CacheModeUse).
func WithCacheMode(mode CacheMode) LoadOption {
	return func(c *loadConfig) { c.cacheMode = mode }
}

// toolsCache memoizes one tool listing per server key.
type toolsCache struct {
	mu    sync.Mutex
	tools map[string][]mcp.Tool
}

func (c *toolsCache) get(key string) ([]mcp.Tool, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	tools, ok := c.tools[key]
	return tools, ok
}

func (c *toolsCache) put(key string, tools []mcp.Tool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.tools == nil {
		c.tools = make(map[string][]mcp.Tool)
	}
	c.tools[key] = tools
}

// LoadTools lists tools across every member connection and adapts them to
// coretools.Tool. Names are namespaced {server}_{tool} so identical remote
// tool names stay distinguishable. A listing or adaption failure on any
// server aborts the load with that error.
func (g *ClientGroup) LoadTools(ctx context.Context, opts ...LoadOption) ([]coretools.Tool, error) {
	cfg := loadConfig{cacheMode: CacheModeUse}
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	var out []coretools.Tool
	for _, key := range g.ClientNames() {
		tools, err := g.listTools(ctx, key, cfg.cacheMode)
		if err != nil {
			return nil, err
		}
		for _, mt := range tools {
			tool, err := newTool(g.members[key], key, mt, g.elicit)
			if err != nil {
				return nil, err
			}
			out = append(out, tool)
		}
	}
	if out == nil {
		out = []coretools.Tool{}
	}
	return out, nil
}

func (g *ClientGroup) listTools(ctx context.Context, key string, mode CacheMode) ([]mcp.Tool, error) {
	if mode == CacheModeUse {
		if tools, ok := g.cache.get(key); ok {
			return tools, nil
		}
	}
	result, err := g.members[key].ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		return nil, fmt.Errorf("mcp: list tools on server %q: %w", key, err)
	}
	if mode != CacheModeBypass {
		g.cache.put(key, result.Tools)
	}
	return result.Tools, nil
}

// Tool is an MCP server tool adapted to coretools.Tool. Invoke calls through
// the mcp-go client, mapping the result per the upstream four quadrants
// (content blocks / structured artifact / error ToolMessage semantics /
// raised transport failures) and bridging mid-call input requests to
// LangGraph interrupts.
type Tool struct {
	name        string
	description string
	argsSchema  schema.Schema
	client      Client
	serverKey   string
	remoteName  string
	annotations mcp.ToolAnnotation
	elicit      bool
}

// newTool adapts one listed MCP tool.
func newTool(cli Client, serverKey string, mt mcp.Tool, elicit bool) (*Tool, error) {
	argsSchema, err := convertInputSchema(mt)
	if err != nil {
		return nil, fmt.Errorf("mcp: tool %q on server %q: %w", mt.Name, serverKey, err)
	}
	return &Tool{
		name:        serverKey + "_" + mt.Name,
		description: mt.Description,
		argsSchema:  argsSchema,
		client:      cli,
		serverKey:   serverKey,
		remoteName:  mt.Name,
		annotations: mt.Annotations,
		elicit:      elicit,
	}, nil
}

// Name returns the namespaced {server}_{tool} name.
func (t *Tool) Name() string { return t.name }

// Description returns the server-provided tool description.
func (t *Tool) Description() string { return t.description }

// ArgsSchema returns the tool's MCP input schema as a JSON Schema map.
func (t *Tool) ArgsSchema() schema.Schema { return t.argsSchema }

// ServerName returns the config key / group member name backing the tool.
func (t *Tool) ServerName() string { return t.serverKey }

// RemoteName returns the tool's un-prefixed name on its MCP server.
func (t *Tool) RemoteName() string { return t.remoteName }

// Annotations returns the server's tool annotations (destructiveHint and
// friends), used by InterruptOnConfigs for HITL gating.
func (t *Tool) Annotations() mcp.ToolAnnotation { return t.annotations }

// Invoke executes the tool on its MCP server.
func (t *Tool) Invoke(ctx context.Context, input map[string]any) (coretools.Result, error) {
	req := mcp.CallToolRequest{}
	req.Params.Name = t.remoteName
	req.Params.Arguments = input

	var batch *elicitBatch
	if t.elicit && graph.InterruptSupported(ctx) {
		batch = newElicitBatch()
		ctx = context.WithValue(ctx, elicitBatchKey{}, batch)
	}
	defer func() {
		if batch != nil {
			batch.abort()
		}
	}()

	type callOutcome struct {
		result *mcp.CallToolResult
		err    error
	}
	outcomeCh := make(chan callOutcome, 1)
	go func() {
		result, err := t.client.CallTool(ctx, req)
		outcomeCh <- callOutcome{result: result, err: err}
	}()

	for {
		var arrival <-chan struct{}
		if batch != nil {
			arrival = batch.arrived
		}
		select {
		case outcome := <-outcomeCh:
			if outcome.err != nil {
				if batch != nil && batch.isCanceled() {
					return coretools.Result{}, fmt.Errorf("%w (tool %q)", ErrElicitationCanceled, t.name)
				}
				return coretools.Result{}, fmt.Errorf("mcp: call tool %q: %w", t.name, outcome.err)
			}
			return mapCallResult(t, outcome.result)
		case <-arrival:
			// Requests are waiting: pause for their answers, deliver, and
			// repeat while any request stays unanswered (partial or
			// key-mismatched resumes re-pause with just the missing
			// subset — answers are key-addressed, so this converges
			// without misattributing values).
			for {
				pending := batch.snapshot()
				if len(pending) == 0 {
					break
				}
				responses, err := DecodeElicitationResponses(graph.Interrupt(ctx, elicitInterruptValue(t, pending)))
				if err != nil {
					return coretools.Result{}, err
				}
				if batch.deliver(responses) {
					// A cancel answer abandons the whole call: wait for
					// the in-flight CallTool to unwind (its handlers were
					// released by the cancellation), bounded by ctx, then
					// surface the cancellation.
					select {
					case <-outcomeCh:
					case <-ctx.Done():
					}
					return coretools.Result{}, fmt.Errorf("%w (tool %q)", ErrElicitationCanceled, t.name)
				}
			}
		case <-ctx.Done():
			return coretools.Result{}, ctx.Err()
		}
	}
}

// ToolError is returned when the server reports isError=true. The server's
// own message is preserved so a HandleToolErrors policy (langchain/tools)
// can render it into the error ToolMessage the model reads and retries on.
type ToolError struct {
	// Message is the server-provided error text.
	Message string
	// ToolName is the namespaced tool that failed.
	ToolName string
}

func (e *ToolError) Error() string {
	return fmt.Sprintf("mcp tool %s failed: %s", e.ToolName, e.Message)
}

// mapCallResult maps one CallToolResult to the core Result contract.
func mapCallResult(t *Tool, result *mcp.CallToolResult) (coretools.Result, error) {
	if result.IsError {
		return coretools.Result{}, &ToolError{
			Message:  contentText(result.Content),
			ToolName: t.name,
		}
	}
	var texts []string
	var blocks []coremessages.ContentBlock
	for _, content := range result.Content {
		switch c := content.(type) {
		case mcp.TextContent:
			texts = append(texts, c.Text)
		case mcp.ImageContent:
			blocks = append(blocks, coremessages.ImageBlock{
				MimeType: c.MIMEType,
				Base64:   c.Data,
			})
		case mcp.AudioContent:
			blocks = append(blocks, coremessages.AudioBlock{
				MimeType: c.MIMEType,
				Base64:   c.Data,
			})
		case mcp.EmbeddedResource:
			if blob, ok := c.Resource.(*mcp.BlobResourceContents); ok && blob.MIMEType != "" {
				// A blob resource is the MCP "file" content: carried as a
				// standardized file block.
				blocks = append(blocks, coremessages.FileBlock{
					MimeType: blob.MIMEType,
					Base64:   blob.Blob,
				})
			}
		}
	}
	out := coretools.Result{Content: strings.Join(texts, "\n")}
	if result.StructuredContent != nil || len(blocks) > 0 {
		out.Artifact = &ToolArtifact{
			StructuredContent: result.StructuredContent,
			ContentBlocks:     blocks,
		}
	}
	return out, nil
}

// contentText concatenates the text parts of an MCP content list.
func contentText(contents []mcp.Content) string {
	var texts []string
	for _, content := range contents {
		if text, ok := content.(mcp.TextContent); ok {
			texts = append(texts, text.Text)
		}
	}
	return strings.Join(texts, "\n")
}

// convertInputSchema round-trips the MCP input schema into a JSON Schema
// map, preserving $defs/required/additionalProperties and the property order
// the server declared.
func convertInputSchema(mt mcp.Tool) (schema.Schema, error) {
	if len(mt.RawInputSchema) > 0 {
		var out schema.Schema
		if err := json.Unmarshal(mt.RawInputSchema, &out); err != nil {
			return nil, fmt.Errorf("decode raw input schema: %w", err)
		}
		if out == nil {
			out = schema.Schema{}
		}
		return out, nil
	}
	raw, err := json.Marshal(mt.InputSchema)
	if err != nil {
		return nil, fmt.Errorf("encode input schema: %w", err)
	}
	var out schema.Schema
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("decode input schema: %w", err)
	}
	if out == nil {
		out = schema.Schema{}
	}
	return out, nil
}

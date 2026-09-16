package mcp

import (
	coremessages "github.com/projanvil/langchain-golang/core/messages"
)

// ToolArtifact is the Go equivalent of upstream's MCPToolArtifact: the parts
// of an MCP tool result that must NOT be folded into model-visible text.
// It rides on coretools.Result.Artifact (and from there onto the ToolMessage
// artifact), mirroring upstream where structured content is read via
// `message.artifact["structured_content"]`.
//
// Upstream additionally places image/file content blocks on the ToolMessage
// itself (content_blocks); the Go tool pipeline (coretools.Result ->
// ToolNode) is text-only by design, so non-text blocks are carried here
// instead. Model-visible text always stays in Result.Content.
type ToolArtifact struct {
	// StructuredContent is the server's structuredContent output, decoded
	// from JSON, when the tool declares an output schema. Nil otherwise.
	StructuredContent any `json:"structured_content,omitempty"`
	// ContentBlocks holds the non-text content of the result (image, audio,
	// file) as standardized content blocks, in result order.
	ContentBlocks []coremessages.ContentBlock `json:"content_blocks,omitempty"`
}

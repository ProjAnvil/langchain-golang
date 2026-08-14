package messages

// This file mirrors langchain_core/messages/content.py: a sealed ContentBlock
// interface with concrete typed structs replacing the old map[string]any.
//
// Union aliases from Python (DataContentBlock, Annotation, ContentBlock) are
// represented as Go interfaces satisfied by the concrete structs.
// NonStandardContentBlock is the escape hatch for provider-specific or
// unrecognized block types (including streaming delta types like "text-delta").

// ContentBlock is the sealed interface for normalized multimodal content.
// Only the concrete types in this package implement it.
type ContentBlock interface {
	BlockType() string // discriminator: "text", "image", etc.
	isContentBlock()   // unexported seal
}

// ContentBlocks is a convenience alias for []ContentBlock.
// type alias kept for readability; the Message struct uses []ContentBlock
// directly so that composite literals like []ContentBlock{...} compile.

// --- Annotation union (Citation | NonStandardAnnotation) ---

// Annotation is the union of Citation and NonStandardAnnotation.
type Annotation interface {
	AnnotationType() string
	isAnnotation()
}

// Citation annotates a text span citing a document source.
type Citation struct {
	Type        string         `json:"type"` // always "citation"
	ID          string         `json:"id,omitempty"`
	URL         string         `json:"url,omitempty"`
	Title       string         `json:"title,omitempty"`
	StartIndex  *int           `json:"start_index,omitempty"`
	EndIndex    *int           `json:"end_index,omitempty"`
	CitedText   string         `json:"cited_text,omitempty"`
	Extras      map[string]any `json:"extras,omitempty"`
}

// AnnotationType returns the discriminator for Citation.
func (Citation) AnnotationType() string { return "citation" }

// isAnnotation seals Citation into the Annotation union.
func (Citation) isAnnotation() {}

// NonStandardAnnotation is a provider-specific annotation format.
type NonStandardAnnotation struct {
	Type  string         `json:"type"` // always "non_standard_annotation"
	ID    string         `json:"id,omitempty"`
	Value map[string]any `json:"value"`
}

// AnnotationType returns the discriminator.
func (NonStandardAnnotation) AnnotationType() string { return "non_standard_annotation" }

// isAnnotation seals NonStandardAnnotation into the Annotation union.
func (NonStandardAnnotation) isAnnotation() {}

// --- Standard content block structs ---

// TextBlock mirrors Python's TextContentBlock (type="text").
type TextBlock struct {
	ID          string       `json:"id,omitempty"`
	Text        string       `json:"text"`
	Annotations []Annotation `json:"annotations,omitempty"`
	Index       any          `json:"index,omitempty"`
	Extras      map[string]any `json:"-"`
}

// BlockType returns "text".
func (TextBlock) BlockType() string { return "text" }

// isContentBlock seals TextBlock.
func (TextBlock) isContentBlock() {}

// ImageBlock mirrors Python's ImageContentBlock (type="image").
type ImageBlock struct {
	ID       string         `json:"id,omitempty"`
	FileID   string         `json:"file_id,omitempty"`
	MimeType string         `json:"mime_type,omitempty"`
	Index    any            `json:"index,omitempty"`
	URL      string         `json:"url,omitempty"`
	Base64   string         `json:"base64,omitempty"`
	Extras   map[string]any `json:"-"`
}

// BlockType returns "image".
func (ImageBlock) BlockType() string { return "image" }

// isContentBlock seals ImageBlock.
func (ImageBlock) isContentBlock() {}

// VideoBlock mirrors Python's VideoContentBlock (type="video").
type VideoBlock struct {
	ID       string         `json:"id,omitempty"`
	FileID   string         `json:"file_id,omitempty"`
	MimeType string         `json:"mime_type,omitempty"`
	Index    any            `json:"index,omitempty"`
	URL      string         `json:"url,omitempty"`
	Base64   string         `json:"base64,omitempty"`
	Extras   map[string]any `json:"-"`
}

// BlockType returns "video".
func (VideoBlock) BlockType() string { return "video" }

// isContentBlock seals VideoBlock.
func (VideoBlock) isContentBlock() {}

// AudioBlock mirrors Python's AudioContentBlock (type="audio").
type AudioBlock struct {
	ID       string         `json:"id,omitempty"`
	FileID   string         `json:"file_id,omitempty"`
	MimeType string         `json:"mime_type,omitempty"`
	Index    any            `json:"index,omitempty"`
	URL      string         `json:"url,omitempty"`
	Base64   string         `json:"base64,omitempty"`
	Extras   map[string]any `json:"-"`
}

// BlockType returns "audio".
func (AudioBlock) BlockType() string { return "audio" }

// isContentBlock seals AudioBlock.
func (AudioBlock) isContentBlock() {}

// FileBlock mirrors Python's FileContentBlock (type="file").
type FileBlock struct {
	ID       string         `json:"id,omitempty"`
	FileID   string         `json:"file_id,omitempty"`
	MimeType string         `json:"mime_type,omitempty"`
	Index    any            `json:"index,omitempty"`
	URL      string         `json:"url,omitempty"`
	Base64   string         `json:"base64,omitempty"`
	Extras   map[string]any `json:"-"`
}

// BlockType returns "file".
func (FileBlock) BlockType() string { return "file" }

// isContentBlock seals FileBlock.
func (FileBlock) isContentBlock() {}

// PlainTextBlock mirrors Python's PlainTextContentBlock (type="text-plain").
type PlainTextBlock struct {
	ID       string         `json:"id,omitempty"`
	FileID   string         `json:"file_id,omitempty"`
	MimeType string         `json:"mime_type,omitempty"`
	Index    any            `json:"index,omitempty"`
	URL      string         `json:"url,omitempty"`
	Base64   string         `json:"base64,omitempty"`
	Text     string         `json:"text,omitempty"`
	Title    string         `json:"title,omitempty"`
	Context  string         `json:"context,omitempty"`
	Extras   map[string]any `json:"-"`
}

// BlockType returns "text-plain".
func (PlainTextBlock) BlockType() string { return "text-plain" }

// isContentBlock seals PlainTextBlock.
func (PlainTextBlock) isContentBlock() {}

// ReasoningBlock mirrors Python's ReasoningContentBlock (type="reasoning").
type ReasoningBlock struct {
	ID        string         `json:"id,omitempty"`
	Reasoning string         `json:"reasoning,omitempty"`
	Index     any            `json:"index,omitempty"`
	Extras    map[string]any `json:"-"`
}

// BlockType returns "reasoning".
func (ReasoningBlock) BlockType() string { return "reasoning" }

// isContentBlock seals ReasoningBlock.
func (ReasoningBlock) isContentBlock() {}

// ToolCallBlock mirrors Python's ToolCall (type="tool_call"). It is distinct
// from the legacy ToolCall struct used on Message.ToolCalls: this is a content
// block carrying a tool call as part of a message's ContentBlocks.
type ToolCallBlock struct {
	ID     string         `json:"id,omitempty"`
	Name   string         `json:"name"`
	Args   map[string]any `json:"args,omitempty"`
	Index  any            `json:"index,omitempty"`
	Extras map[string]any `json:"-"`
}

// BlockType returns "tool_call".
func (ToolCallBlock) BlockType() string { return "tool_call" }

// isContentBlock seals ToolCallBlock.
func (ToolCallBlock) isContentBlock() {}

// ToolCallChunkBlock mirrors Python's ToolCallChunk (type="tool_call_chunk").
type ToolCallChunkBlock struct {
	ID     string         `json:"id,omitempty"`
	Name   string         `json:"name,omitempty"`
	Args   string         `json:"args,omitempty"`
	Index  any            `json:"index,omitempty"`
	Extras map[string]any `json:"-"`
}

// BlockType returns "tool_call_chunk".
func (ToolCallChunkBlock) BlockType() string { return "tool_call_chunk" }

// isContentBlock seals ToolCallChunkBlock.
func (ToolCallChunkBlock) isContentBlock() {}

// InvalidToolCallBlock mirrors Python's InvalidToolCall (type="invalid_tool_call").
type InvalidToolCallBlock struct {
	ID     string         `json:"id,omitempty"`
	Name   string         `json:"name,omitempty"`
	Args   string         `json:"args,omitempty"`
	Error  string         `json:"error,omitempty"`
	Index  any            `json:"index,omitempty"`
	Extras map[string]any `json:"-"`
}

// BlockType returns "invalid_tool_call".
func (InvalidToolCallBlock) BlockType() string { return "invalid_tool_call" }

// isContentBlock seals InvalidToolCallBlock.
func (InvalidToolCallBlock) isContentBlock() {}

// ServerToolCall mirrors Python's ServerToolCall (type="server_tool_call"):
// a tool call executed server-side (e.g. code execution, web search).
type ServerToolCall struct {
	ID     string         `json:"id,omitempty"`
	Name   string         `json:"name"`
	Args   map[string]any `json:"args,omitempty"`
	Index  any            `json:"index,omitempty"`
	Extras map[string]any `json:"-"`
}

// BlockType returns "server_tool_call".
func (ServerToolCall) BlockType() string { return "server_tool_call" }

// isContentBlock seals ServerToolCall.
func (ServerToolCall) isContentBlock() {}

// ServerToolCallChunk mirrors Python's ServerToolCallChunk
// (type="server_tool_call_chunk"): a chunk of a server-side tool call yielded
// while streaming.
type ServerToolCallChunk struct {
	ID     string         `json:"id,omitempty"`
	Name   string         `json:"name,omitempty"`
	Args   string         `json:"args,omitempty"`
	Index  any            `json:"index,omitempty"`
	Extras map[string]any `json:"-"`
}

// BlockType returns "server_tool_call_chunk".
func (ServerToolCallChunk) BlockType() string { return "server_tool_call_chunk" }

// isContentBlock seals ServerToolCallChunk.
func (ServerToolCallChunk) isContentBlock() {}

// ServerToolResult mirrors Python's ServerToolResult (type="server_tool_result"):
// the result of a server-side tool call.
type ServerToolResult struct {
	ID         string         `json:"id,omitempty"`
	ToolCallID string         `json:"tool_call_id"`
	Status     string         `json:"status"`
	Output     any            `json:"output,omitempty"`
	Index      any            `json:"index,omitempty"`
	Extras     map[string]any `json:"-"`
}

// BlockType returns "server_tool_result".
func (ServerToolResult) BlockType() string { return "server_tool_result" }

// isContentBlock seals ServerToolResult.
func (ServerToolResult) isContentBlock() {}

// --- Escape hatch ---

// NonStandardContentBlock holds provider-specific or unrecognized block data.
// Type carries the provider's discriminator (e.g. "text-delta", "tool_use",
// "non_standard"); Value holds all remaining fields. This is the escape hatch
// that lets arbitrary provider blocks round-trip without a dedicated struct.
type NonStandardContentBlock struct {
	Type  string
	Value map[string]any
}

// BlockType returns the provider's discriminator (b.Type).
func (b NonStandardContentBlock) BlockType() string { return b.Type }

// isContentBlock seals NonStandardContentBlock.
func (NonStandardContentBlock) isContentBlock() {}

// --- Compile-time interface satisfaction checks ---

var (
	_ ContentBlock = TextBlock{}
	_ ContentBlock = ImageBlock{}
	_ ContentBlock = VideoBlock{}
	_ ContentBlock = AudioBlock{}
	_ ContentBlock = FileBlock{}
	_ ContentBlock = PlainTextBlock{}
	_ ContentBlock = ReasoningBlock{}
	_ ContentBlock = ToolCallBlock{}
	_ ContentBlock = ToolCallChunkBlock{}
	_ ContentBlock = InvalidToolCallBlock{}
	_ ContentBlock = ServerToolCall{}
	_ ContentBlock = ServerToolCallChunk{}
	_ ContentBlock = ServerToolResult{}
	_ ContentBlock = NonStandardContentBlock{}
)

// KNOWN_BLOCK_TYPES mirrors Python's KNOWN_BLOCK_TYPES set. Types not in this
// set are treated as provider-specific (NonStandardContentBlock).
var KNOWN_BLOCK_TYPES = map[string]bool{
	"text":                  true,
	"reasoning":             true,
	"tool_call":             true,
	"invalid_tool_call":     true,
	"tool_call_chunk":       true,
	"image":                 true,
	"audio":                 true,
	"file":                  true,
	"text-plain":            true,
	"video":                 true,
	"server_tool_call":      true,
	"server_tool_call_chunk": true,
	"server_tool_result":    true,
	"non_standard":          true,
}

// --- Bridge helpers ---

// ParseContentBlock converts a raw map[string]any (e.g. from provider JSON or
// legacy map-based code) into the appropriate typed ContentBlock. Unknown types
// default to NonStandardContentBlock. This is the bridge for any remaining
// map-based data and the escape hatch.
func ParseContentBlock(m map[string]any) ContentBlock {
	if m == nil {
		return nil
	}
	typ, _ := m["type"].(string)
	switch typ {
	case "text":
		return parseTextBlock(m)
	case "image":
		return parseImageBlock(m)
	case "video":
		return parseVideoBlock(m)
	case "audio":
		return parseAudioBlock(m)
	case "file":
		return parseFileBlock(m)
	case "text-plain":
		return parsePlainTextBlock(m)
	case "reasoning":
		return parseReasoningBlock(m)
	case "tool_call":
		return parseToolCallBlock(m)
	case "tool_call_chunk":
		return parseToolCallChunkBlock(m)
	case "invalid_tool_call":
		return parseInvalidToolCallBlock(m)
	case "server_tool_call":
		return parseServerToolCall(m)
	case "server_tool_call_chunk":
		return parseServerToolCallChunk(m)
	case "server_tool_result":
		return parseServerToolResult(m)
	default:
		// Unknown / provider-specific: escape hatch.
		return NonStandardContentBlock{Type: typ, Value: cloneMap(stripType(m))}
	}
}

// BlockToMap converts any typed ContentBlock back to a map[string]any,
// flattening Extras into the top level. This is used for JSON serialization,
// legacy map-style access, and provider adapters that iterate over key-value
// pairs. Returns nil for a nil block.
func BlockToMap(b ContentBlock) map[string]any {
	if b == nil {
		return nil
	}
	switch v := b.(type) {
	case TextBlock:
		out := map[string]any{"type": "text"}
		out["text"] = v.Text
		if v.ID != "" {
			out["id"] = v.ID
		}
		if len(v.Annotations) > 0 {
			out["annotations"] = v.Annotations
		}
		if v.Index != nil {
			out["index"] = v.Index
		}
		mergeExtras(out, v.Extras)
		return out
	case ImageBlock:
		return dataBlockToMap("image", v.ID, v.FileID, v.MimeType, v.Index, v.URL, v.Base64, v.Extras)
	case VideoBlock:
		return dataBlockToMap("video", v.ID, v.FileID, v.MimeType, v.Index, v.URL, v.Base64, v.Extras)
	case AudioBlock:
		return dataBlockToMap("audio", v.ID, v.FileID, v.MimeType, v.Index, v.URL, v.Base64, v.Extras)
	case FileBlock:
		return dataBlockToMap("file", v.ID, v.FileID, v.MimeType, v.Index, v.URL, v.Base64, v.Extras)
	case PlainTextBlock:
		out := map[string]any{"type": "text-plain", "mime_type": v.MimeType}
		if v.ID != "" {
			out["id"] = v.ID
		}
		if v.FileID != "" {
			out["file_id"] = v.FileID
		}
		if v.Index != nil {
			out["index"] = v.Index
		}
		if v.URL != "" {
			out["url"] = v.URL
		}
		if v.Base64 != "" {
			out["base64"] = v.Base64
		}
		if v.Text != "" {
			out["text"] = v.Text
		}
		if v.Title != "" {
			out["title"] = v.Title
		}
		if v.Context != "" {
			out["context"] = v.Context
		}
		mergeExtras(out, v.Extras)
		return out
	case ReasoningBlock:
		out := map[string]any{"type": "reasoning"}
		if v.ID != "" {
			out["id"] = v.ID
		}
		if v.Reasoning != "" {
			out["reasoning"] = v.Reasoning
		}
		if v.Index != nil {
			out["index"] = v.Index
		}
		mergeExtras(out, v.Extras)
		return out
	case ToolCallBlock:
		out := map[string]any{"type": "tool_call", "name": v.Name}
		if v.ID != "" {
			out["id"] = v.ID
		}
		if v.Args != nil {
			out["args"] = v.Args
		}
		if v.Index != nil {
			out["index"] = v.Index
		}
		mergeExtras(out, v.Extras)
		return out
	case ToolCallChunkBlock:
		out := map[string]any{"type": "tool_call_chunk"}
		if v.ID != "" {
			out["id"] = v.ID
		}
		if v.Name != "" {
			out["name"] = v.Name
		}
		if v.Args != "" {
			out["args"] = v.Args
		}
		if v.Index != nil {
			out["index"] = v.Index
		}
		mergeExtras(out, v.Extras)
		return out
	case InvalidToolCallBlock:
		out := map[string]any{"type": "invalid_tool_call"}
		if v.ID != "" {
			out["id"] = v.ID
		}
		if v.Name != "" {
			out["name"] = v.Name
		}
		if v.Args != "" {
			out["args"] = v.Args
		}
		if v.Error != "" {
			out["error"] = v.Error
		}
		if v.Index != nil {
			out["index"] = v.Index
		}
		mergeExtras(out, v.Extras)
		return out
	case ServerToolCall:
		out := map[string]any{"type": "server_tool_call", "name": v.Name}
		if v.ID != "" {
			out["id"] = v.ID
		}
		if v.Args != nil {
			out["args"] = v.Args
		}
		if v.Index != nil {
			out["index"] = v.Index
		}
		mergeExtras(out, v.Extras)
		return out
	case ServerToolCallChunk:
		out := map[string]any{"type": "server_tool_call_chunk"}
		if v.ID != "" {
			out["id"] = v.ID
		}
		if v.Name != "" {
			out["name"] = v.Name
		}
		if v.Args != "" {
			out["args"] = v.Args
		}
		if v.Index != nil {
			out["index"] = v.Index
		}
		mergeExtras(out, v.Extras)
		return out
	case ServerToolResult:
		out := map[string]any{"type": "server_tool_result", "tool_call_id": v.ToolCallID, "status": v.Status}
		if v.ID != "" {
			out["id"] = v.ID
		}
		if v.Output != nil {
			out["output"] = v.Output
		}
		if v.Index != nil {
			out["index"] = v.Index
		}
		mergeExtras(out, v.Extras)
		return out
	case NonStandardContentBlock:
		out := map[string]any{}
		for k, val := range v.Value {
			out[k] = val
		}
		out["type"] = v.Type
		return out
	default:
		// Should never happen (sealed interface), but provide a safe fallback.
		return map[string]any{"type": b.BlockType()}
	}
}

// dataBlockToMap is a shared helper for the image/video/audio/file blocks
// which all share the same field shape.
func dataBlockToMap(blockType, id, fileID, mimeType string, index any, url, base64 string, extras map[string]any) map[string]any {
	out := map[string]any{"type": blockType}
	if id != "" {
		out["id"] = id
	}
	if fileID != "" {
		out["file_id"] = fileID
	}
	if mimeType != "" {
		out["mime_type"] = mimeType
	}
	if index != nil {
		out["index"] = index
	}
	if url != "" {
		out["url"] = url
	}
	if base64 != "" {
		out["base64"] = base64
	}
	mergeExtras(out, extras)
	return out
}

// mergeExtras copies extras into out (does not overwrite existing keys).
func mergeExtras(out map[string]any, extras map[string]any) {
	for k, v := range extras {
		if _, exists := out[k]; !exists {
			out[k] = v
		}
	}
}

// stripType returns a copy of m with the "type" key removed.
func stripType(m map[string]any) map[string]any {
	out := make(map[string]any, len(m))
	for k, v := range m {
		if k != "type" {
			out[k] = v
		}
	}
	return out
}

// CloneBlock returns a deep copy of any ContentBlock.
func CloneBlock(b ContentBlock) ContentBlock {
	if b == nil {
		return nil
	}
	switch v := b.(type) {
	case TextBlock:
		return TextBlock{
			ID:          v.ID,
			Text:        v.Text,
			Annotations: cloneAnnotations(v.Annotations),
			Index:       v.Index,
			Extras:      cloneMap(v.Extras),
		}
	case ImageBlock:
		return cloneDataBlock(v.ID, v.FileID, v.MimeType, v.Index, v.URL, v.Base64, v.Extras,
			func(id, fid, mt string, idx any, url, b64 string, ex map[string]any) ContentBlock {
				return ImageBlock{ID: id, FileID: fid, MimeType: mt, Index: idx, URL: url, Base64: b64, Extras: ex}
			})
	case VideoBlock:
		return cloneDataBlock(v.ID, v.FileID, v.MimeType, v.Index, v.URL, v.Base64, v.Extras,
			func(id, fid, mt string, idx any, url, b64 string, ex map[string]any) ContentBlock {
				return VideoBlock{ID: id, FileID: fid, MimeType: mt, Index: idx, URL: url, Base64: b64, Extras: ex}
			})
	case AudioBlock:
		return cloneDataBlock(v.ID, v.FileID, v.MimeType, v.Index, v.URL, v.Base64, v.Extras,
			func(id, fid, mt string, idx any, url, b64 string, ex map[string]any) ContentBlock {
				return AudioBlock{ID: id, FileID: fid, MimeType: mt, Index: idx, URL: url, Base64: b64, Extras: ex}
			})
	case FileBlock:
		return cloneDataBlock(v.ID, v.FileID, v.MimeType, v.Index, v.URL, v.Base64, v.Extras,
			func(id, fid, mt string, idx any, url, b64 string, ex map[string]any) ContentBlock {
				return FileBlock{ID: id, FileID: fid, MimeType: mt, Index: idx, URL: url, Base64: b64, Extras: ex}
			})
	case PlainTextBlock:
		return PlainTextBlock{
			ID:       v.ID,
			FileID:   v.FileID,
			MimeType: v.MimeType,
			Index:    v.Index,
			URL:      v.URL,
			Base64:   v.Base64,
			Text:     v.Text,
			Title:    v.Title,
			Context:  v.Context,
			Extras:   cloneMap(v.Extras),
		}
	case ReasoningBlock:
		return ReasoningBlock{
			ID:        v.ID,
			Reasoning: v.Reasoning,
			Index:     v.Index,
			Extras:    cloneMap(v.Extras),
		}
	case ToolCallBlock:
		return ToolCallBlock{
			ID:     v.ID,
			Name:   v.Name,
			Args:   cloneMap(v.Args),
			Index:  v.Index,
			Extras: cloneMap(v.Extras),
		}
	case ToolCallChunkBlock:
		return ToolCallChunkBlock{
			ID:     v.ID,
			Name:   v.Name,
			Args:   v.Args,
			Index:  v.Index,
			Extras: cloneMap(v.Extras),
		}
	case InvalidToolCallBlock:
		return InvalidToolCallBlock{
			ID:     v.ID,
			Name:   v.Name,
			Args:   v.Args,
			Error:  v.Error,
			Index:  v.Index,
			Extras: cloneMap(v.Extras),
		}
	case ServerToolCall:
		return ServerToolCall{
			ID:     v.ID,
			Name:   v.Name,
			Args:   cloneMap(v.Args),
			Index:  v.Index,
			Extras: cloneMap(v.Extras),
		}
	case ServerToolCallChunk:
		return ServerToolCallChunk{
			ID:     v.ID,
			Name:   v.Name,
			Args:   v.Args,
			Index:  v.Index,
			Extras: cloneMap(v.Extras),
		}
	case ServerToolResult:
		return ServerToolResult{
			ID:         v.ID,
			ToolCallID: v.ToolCallID,
			Status:     v.Status,
			Output:     cloneAny(v.Output),
			Index:      v.Index,
			Extras:     cloneMap(v.Extras),
		}
	case NonStandardContentBlock:
		return NonStandardContentBlock{
			Type:  v.Type,
			Value: cloneMap(v.Value),
		}
	default:
		return b
	}
}

func cloneDataBlock(id, fid, mt string, idx any, url, b64 string, ex map[string]any,
	constructor func(id, fid, mt string, idx any, url, b64 string, ex map[string]any) ContentBlock,
) ContentBlock {
	return constructor(id, fid, mt, idx, url, b64, cloneMap(ex))
}

func cloneAnnotations(anns []Annotation) []Annotation {
	if anns == nil {
		return nil
	}
	out := make([]Annotation, len(anns))
	for i, a := range anns {
		out[i] = a
	}
	return out
}

// --- Parse helpers ---

func parseTextBlock(m map[string]any) TextBlock {
	b := TextBlock{}
	b.Text, _ = m["text"].(string)
	b.ID, _ = m["id"].(string)
	b.Index = m["index"]
	if anns, ok := m["annotations"].([]Annotation); ok {
		b.Annotations = anns
	}
	b.Extras = extractExtras(m, "type", "text", "id", "index", "annotations")
	return b
}

func parseImageBlock(m map[string]any) ImageBlock {
	id, fid, mt, idx, url, b64, ex := parseDataBlockFields(m)
	return ImageBlock{ID: id, FileID: fid, MimeType: mt, Index: idx, URL: url, Base64: b64, Extras: ex}
}

func parseVideoBlock(m map[string]any) VideoBlock {
	id, fid, mt, idx, url, b64, ex := parseDataBlockFields(m)
	return VideoBlock{ID: id, FileID: fid, MimeType: mt, Index: idx, URL: url, Base64: b64, Extras: ex}
}

func parseAudioBlock(m map[string]any) AudioBlock {
	id, fid, mt, idx, url, b64, ex := parseDataBlockFields(m)
	return AudioBlock{ID: id, FileID: fid, MimeType: mt, Index: idx, URL: url, Base64: b64, Extras: ex}
}

func parseFileBlock(m map[string]any) FileBlock {
	id, fid, mt, idx, url, b64, ex := parseDataBlockFields(m)
	return FileBlock{ID: id, FileID: fid, MimeType: mt, Index: idx, URL: url, Base64: b64, Extras: ex}
}

func parsePlainTextBlock(m map[string]any) PlainTextBlock {
	b := PlainTextBlock{}
	b.ID, _ = m["id"].(string)
	b.FileID, _ = m["file_id"].(string)
	b.MimeType, _ = m["mime_type"].(string)
	if b.MimeType == "" {
		b.MimeType = "text/plain"
	}
	b.Index = m["index"]
	b.URL, _ = m["url"].(string)
	b.Base64, _ = m["base64"].(string)
	b.Text, _ = m["text"].(string)
	b.Title, _ = m["title"].(string)
	b.Context, _ = m["context"].(string)
	b.Extras = extractExtras(m, "type", "id", "file_id", "mime_type", "index", "url", "base64", "text", "title", "context")
	return b
}

func parseReasoningBlock(m map[string]any) ReasoningBlock {
	b := ReasoningBlock{}
	b.ID, _ = m["id"].(string)
	b.Reasoning, _ = m["reasoning"].(string)
	b.Index = m["index"]
	b.Extras = extractExtras(m, "type", "id", "reasoning", "index")
	return b
}

func parseToolCallBlock(m map[string]any) ToolCallBlock {
	b := ToolCallBlock{}
	b.ID, _ = m["id"].(string)
	b.Name, _ = m["name"].(string)
	if args, ok := m["args"].(map[string]any); ok {
		b.Args = args
	}
	b.Index = m["index"]
	b.Extras = extractExtras(m, "type", "id", "name", "args", "index")
	return b
}

func parseToolCallChunkBlock(m map[string]any) ToolCallChunkBlock {
	b := ToolCallChunkBlock{}
	b.ID, _ = m["id"].(string)
	b.Name, _ = m["name"].(string)
	b.Args, _ = m["args"].(string)
	b.Index = m["index"]
	b.Extras = extractExtras(m, "type", "id", "name", "args", "index")
	return b
}

func parseInvalidToolCallBlock(m map[string]any) InvalidToolCallBlock {
	b := InvalidToolCallBlock{}
	b.ID, _ = m["id"].(string)
	b.Name, _ = m["name"].(string)
	b.Args, _ = m["args"].(string)
	b.Error, _ = m["error"].(string)
	b.Index = m["index"]
	b.Extras = extractExtras(m, "type", "id", "name", "args", "error", "index")
	return b
}

func parseServerToolCall(m map[string]any) ServerToolCall {
	b := ServerToolCall{}
	b.ID, _ = m["id"].(string)
	b.Name, _ = m["name"].(string)
	if args, ok := m["args"].(map[string]any); ok {
		b.Args = args
	}
	b.Index = m["index"]
	b.Extras = extractExtras(m, "type", "id", "name", "args", "index")
	return b
}

func parseServerToolCallChunk(m map[string]any) ServerToolCallChunk {
	b := ServerToolCallChunk{}
	b.ID, _ = m["id"].(string)
	b.Name, _ = m["name"].(string)
	b.Args, _ = m["args"].(string)
	b.Index = m["index"]
	b.Extras = extractExtras(m, "type", "id", "name", "args", "index")
	return b
}

func parseServerToolResult(m map[string]any) ServerToolResult {
	b := ServerToolResult{}
	b.ID, _ = m["id"].(string)
	b.ToolCallID, _ = m["tool_call_id"].(string)
	b.Status, _ = m["status"].(string)
	b.Output = m["output"]
	b.Index = m["index"]
	b.Extras = extractExtras(m, "type", "id", "tool_call_id", "status", "output", "index")
	return b
}

// parseDataBlockFields extracts the shared fields for image/video/audio/file blocks.
func parseDataBlockFields(m map[string]any) (id, fileID, mimeType string, index any, url, base64 string, extras map[string]any) {
	id, _ = m["id"].(string)
	fileID, _ = m["file_id"].(string)
	mimeType, _ = m["mime_type"].(string)
	index = m["index"]
	url, _ = m["url"].(string)
	base64, _ = m["base64"].(string)
	// Also support legacy field names used by old v0 blocks.
	if data, ok := m["data"].(string); ok && base64 == "" {
		base64 = data
	}
	extras = extractExtras(m, "type", "id", "file_id", "mime_type", "index", "url", "base64", "data")
	return
}

// extractExtras returns all keys from m that are not in the known set.
func extractExtras(m map[string]any, knownKeys ...string) map[string]any {
	known := make(map[string]bool, len(knownKeys))
	for _, k := range knownKeys {
		known[k] = true
	}
	var extras map[string]any
	for k, v := range m {
		if !known[k] {
			if extras == nil {
				extras = make(map[string]any)
			}
			extras[k] = v
		}
	}
	return extras
}

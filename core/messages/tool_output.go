package messages

// ToolOutput mirrors Python's `ToolOutputMixin` marker
// (langchain_core/messages/tool.py:16-23): a value a tool may return directly,
// exempt from the generic output-to-string coercion. In this Go port the
// coercion never applies (tools return the structured tools.Result{Content,
// Artifact}), so the interface exists purely as a recognition marker: tools
// signal graph control flow by placing a marked value — *types.Command from
// langgraph/types — in Result.Artifact, and tool executors (e.g.
// langchain/tools.ToolNode.commandFromArtifact) detect it via this interface.
//
// Python's ToolMessage also carries the mixin; Go's ToolMessage is a
// Message with Role==RoleTool and cannot be marked selectively, so only
// langgraph/types.Command implements ToolOutput in practice.
type ToolOutput interface {
	// IsToolOutput marks the value as a direct tool output. It always
	// returns true; the method exists only to seal the marker.
	IsToolOutput() bool
}

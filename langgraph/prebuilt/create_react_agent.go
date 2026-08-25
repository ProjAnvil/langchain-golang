package prebuilt

import (
	"github.com/projanvil/langchain-golang/core/language"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langchain/agents"
)

// CreateReactAgent creates an agent graph that calls tools in a loop until
// the model stops requesting tools, mirroring Python's
// `langgraph.prebuilt.create_react_agent` (chat_agent_executor.py:278).
//
// Python deprecated create_react_agent in favor of
// `langchain.agents.create_agent` (chat_agent_executor.py:313-317), so this
// function is a thin delegation to langchain/agents.CreateAgent: every option
// is an agents.AgentOption applied verbatim (WithAgentSystemPrompt,
// WithAgentCheckpointer, WithAgentStore, WithAgentInterruptBefore/After,
// WithAgentResponseFormat, WithAgentMiddleware, WithAgentRecursionLimit,
// WithAgentDebug, WithAgentName, ...). The returned *agents.Agent wraps the
// compiled graph (agent.Graph), the direct equivalent of the Python
// CompiledStateGraph return value.
//
// Layering note: langgraph/prebuilt already depends on langchain/tools
// (tool_node.go), and langchain/agents does not import langgraph/prebuilt, so
// this delegation introduces no import cycle.
func CreateReactAgent(model language.ChatModel, toolList []coretools.Tool, opts ...agents.AgentOption) (*agents.Agent, error) {
	return agents.CreateAgent(model, toolList, opts...)
}

// Package agentruntime is a thin alias layer over the public
// `github.com/projanvil/langchain-golang/langgraph` packages, kept so
// existing consumers (langchain/agents) compile unchanged after the graph
// runtime was promoted out of `internal`. New code should import the
// langgraph packages directly.
package agentruntime

import "github.com/projanvil/langchain-golang/langgraph/types"

const (
	START       = types.START
	END         = types.END
	ParentGraph = types.ParentGraph
)

type Send = types.Send
type Command = types.Command
type Interrupt = types.Interrupt
type GraphInterrupt = types.GraphInterrupt

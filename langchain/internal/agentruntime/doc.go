// Package agentruntime is the internal graph runtime backing
// `langchain/agents.CreateAgent`. It is NOT a general langgraph port:
// subgraphs, streaming modes, time-travel, persistent checkpoint backends, and
// the functional `@entrypoint`/`@task` API are intentionally absent. Only the
// subset Python's `create_agent` depends on is implemented.
package agentruntime

# multi_agent_multi-agent-collaboration

Go port of `langgraph/examples/multi_agent/multi-agent-collaboration.ipynb`.

The notebook wires a researcher agent and a chart-generator agent in a loop:
each agent runs its own ReAct loop, hands control to the other with
`Command(goto=...)`, and stops when an agent prefixes its message with
`FINAL ANSWER`. The Go port keeps that control flow:

- two `StateGraph` nodes whose `Command.Goto` routes to the peer or to
  `__end__` when the canned message contains `FINAL ANSWER`;
- a `MessagesReducer` channel for the shared history, mirroring `add_messages`;
- a `MemorySaver` checkpointer and a thread ID, as the notebook streams the
  graph with a runnable config.

Both agents are deterministic offline stubs. The Python `PythonREPL` chart tool
is replaced by a fixed result string; no code is executed. No API key, network,
or external service is required.

## Run

```sh
cd examples/langgraph/multi_agent_multi-agent-collaboration
go run .
```

# create-react-agent-system-prompt

Go port of `langgraph/examples/create-react-agent-system-prompt.ipynb`.

The notebook passes `prompt="Respond in Italian"` to `create_react_agent`.
The Go version uses `agents.WithAgentSystemPrompt` and prints both the system
message observed at the model boundary and the final conversation, showing that
the prompt is installed before the first model call.

This example is fully offline. The ChatModel is a deterministic in-process
double standing in for the notebook's `ChatOpenAI("gpt-4o")`; no API key,
network call, or external service is involved.

## Run

```sh
cd examples/langgraph/create-react-agent-system-prompt
go run .
```

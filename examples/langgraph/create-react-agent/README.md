# create-react-agent

Go port of `langgraph/examples/create-react-agent.ipynb`.

The notebook builds a ReAct agent with `create_react_agent(model, tools=[get_weather])`
and streams two questions: one that routes through the weather tool and one
that the model answers directly. The Go version recreates both turns with
`langchain/agents.CreateAgent` and a `get_weather` tool built via
`coretools.NewFunc`.

This example is fully offline. The ChatModel is a deterministic in-process
double that returns scripted messages; no API key, network call, or external
service is involved. It stands in for the notebook's `ChatOpenAI("gpt-4o")`.

## Run

```sh
cd examples/langgraph/create-react-agent
go run .
```

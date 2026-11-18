# react-agent-structured-output

Go port of `langgraph/examples/react-agent-structured-output.ipynb`.

The notebook's first variant binds `get_weather` plus a `WeatherResponse`
schema tool with `tool_choice="any"`, routes a `WeatherResponse` tool call to a
`respond` node, and parses its arguments into a typed object. The Go version
recreates that graph with `langgraph/graph.StateGraph`, a scripted offline model
that emits the two tool calls, and a plain Go struct `WeatherResponse` filled
from the schema tool call's arguments via `encoding/json`. The
`respond` node also appends the required ToolMessage for the schema tool call.

This example is fully offline. The ChatModel is a deterministic in-process
double standing in for the notebook's `ChatAnthropic`; no API key, network
call, or external service is involved. The notebook's second variant
(`model.with_structured_output(WeatherResponse)`) has no offline provider
equivalent; in Go the same typed result is produced by parsing the tool-call
arguments into the struct, as shown here.

## Run

```sh
cd examples/langgraph/react-agent-structured-output
go run .
```

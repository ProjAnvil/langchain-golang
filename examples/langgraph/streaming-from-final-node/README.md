# streaming-from-final-node

Go port of `langgraph/examples/streaming-from-final-node.ipynb`.

A ReAct weather agent is extended with a dedicated `final` node that rewrites
the answer in a different voice. The notebook reads that node's tokens from a
`messages` stream by checking `metadata["langgraph_node"] == "final"`, and
separately filters `astream_events` by the model's `final_node` tag. The Go
port uses `StreamMessages` with `MessageChunk.Metadata` for the first filter.
Because Go's messages metadata does not carry model tags, the tag-based cell is
ported with a `StreamCustom` payload carrying the same `final_node` tag.

## Offline note

`ChatOpenAI` and `ToolNode` from the notebook are replaced by deterministic
local nodes and a fake streaming model. No API key, network, or external
service is required.

## Run

```sh
cd examples/langgraph/streaming-from-final-node
go run .
```

# streaming-events-from-within-tools-without-langchain

Go port of
`langgraph/examples/streaming-events-from-within-tools-without-langchain.ipynb`.

The notebook calls raw `AsyncOpenAI` from a graph node (emitting LLM tokens via
a callback manager) and dispatches a custom event with tag `tool_call` from
inside a tool, then filters those events with `astream_events`. The Go port
uses a deterministic fake model that emits `EventChatModelStream` chunks via
the callbacks manager installed by `StreamMessages`, and a tool that writes
tagged payloads via `StreamWriter`. The tagged chunks are read with
`StreamCustom` and the text tokens with `StreamMessages`.

## Offline note

No OpenAI client is constructed and no `OPENAI_API_KEY` is read. All model and
tool outputs are scripted locally, so the example runs without network or
services.

## Run

```sh
cd examples/langgraph/streaming-events-from-within-tools-without-langchain
go run .
```

# streaming-tokens-without-langchain

Go port of
`langgraph/examples/streaming-tokens-without-langchain.ipynb`.

The notebook streams tokens from a raw OpenAI client inside a graph node by
manually emitting callback events, and gathers streamed tool-call chunks in
`messages` mode. The Go port uses a fake ChatModel that emits
`EventChatModelStream` chunks through the callbacks manager installed by
`StreamMessages`. Tool-output tokens are emitted as tagged `StreamCustom`
payloads, the Go counterpart of the notebook's custom events.

## Offline note

No `AsyncOpenAI` client is created and no `OPENAI_API_KEY` is read. Model and
tool responses are scripted locally; no network or external service is used.

## Run

```sh
cd examples/langgraph/streaming-tokens-without-langchain
go run .
```

# configuration

Go port of `langgraph/examples/configuration.ipynb`.

The notebook shows how a node reads `config["configurable"]` to pick a model
and to prepend a system message. In this Go SDK the equivalent "configurable"
bag is attached to the run context with `runtime.ContextWithValues` and read
inside a node via `runtime.ValueFromRuntime` (or `rt.Context`).

## What this port demonstrates

- Passing per-invocation configuration as context values and reading it from
  `runtime.Runtime` inside a node.
- Selecting one of several model stubs with `configurable.model`
  (default `anthropic`), matching the notebook's `models` dict.
- Prepending a `SystemMessage` when `configurable.system_message` is present,
  then appending the model reply through `channels.MessagesReducer`.
- Three invocations: default config, `model=openai`, and a system message.

## Offline stub

The notebook uses `ChatAnthropic` / `ChatOpenAI` with API keys. This port uses
deterministic in-process model doubles that echo the chosen model name and any
system message. No API key, network, or external service is used.

## Run

```sh
cd examples/langgraph/configuration
go run .
```

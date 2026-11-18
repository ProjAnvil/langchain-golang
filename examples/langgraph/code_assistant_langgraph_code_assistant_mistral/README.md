# code_assistant_langgraph_code_assistant_mistral

Go port of `examples/code_assistant/langgraph_code_assistant_mistral.ipynb`.

## What the Go port demonstrates

- the notebook's `generate -> check_code` retry graph, built with the Go
  `StateGraph` API and a conditional edge back to `generate` on failure
- a `checkpoint.NewMemorySaver()` checkpointer, with one thread per question
- `stream_mode="values"` reproduced with `CompiledGraph.Stream` and
  `graphpkg.StreamValues`; a `_print_event`-style helper prints only the newest
  message of each state snapshot
- an offline `language.ChatModel` double whose `BindTools` returns itself,
  standing in for the Mistral structured-output model

## Offline safe substitution

The notebook calls ChatMistralAI, pulls prompts from a hub, and executes
generated Python with `exec`. This port replaces all of that:

- a deterministic in-process model returns fixed `prefix`/`imports`/`code`
  text; the first attempt of each question intentionally has a bad import
- the code checks are in-memory string inspections against an allowlist;
  **no generated code is executed, and no shell command is run**
- no API key, network request, or service is used

## Run

```sh
cd examples/langgraph/code_assistant_langgraph_code_assistant_mistral
go run .
```

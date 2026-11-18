# code_assistant_langgraph_code_assistant

Go port of `examples/code_assistant/langgraph_code_assistant.ipynb`.

## What the Go port demonstrates

The notebook's LangGraph `StateGraph` is rebuilt with the Go SDK:

- `generate` -> `check_code` -> conditional `decide_to_finish`
- the conditional router sends a failing attempt either back to `generate`
  (the notebook's `flag = "do not reflect"`) or through `reflect` first
  (`flag = "reflect"`); both variants are run
- `messages` uses `channels.MessagesReducer`, the Go equivalent of
  Python's `add_messages`
- a `checkpoint.NewMemorySaver()` checkpointer is installed and each run
  uses a `ThreadID`, mirroring a persisted notebook session
- `generation` holds a small Go struct equivalent to the notebook's
  `code` Pydantic model (`prefix`, `imports`, `code`)

## Offline safe substitution

The Python notebook crawls `python.langchain.com`, calls OpenAI/Anthropic
models, and executes generated Python with `exec` for its import and code
checks. This port does none of that:

- the "documentation corpus" is a fixed slice of three LCEL strings
- code generation is a deterministic function that deliberately fails its
  first import check, so the retry loop always runs the same way
- the checks are in-memory string inspections against an import allowlist;
  **no generated code is ever executed, and no shell command is ever run**

No API key, network access, or external service is required.

## Run

```sh
cd examples/langgraph/code_assistant_langgraph_code_assistant
go run .
```

# self-discover_self-discover

Go port of `langgraph/examples/self-discover/self-discover.ipynb`.

The notebook's Self-Discovery graph is a fixed chain of four prompt-driven
steps: select reasoning modules, adapt them to the task, structure them into a
reasoning pipeline, and reason with that pipeline. The Go port mirrors the
linear `select -> adapt -> structure -> reason -> END` `StateGraph` and keeps
the same state keys for each intermediate result.

All four steps are deterministic offline stubs. The LangSmith hub prompts and
the OpenAI model are not used, so no API key or network is required; the
example prints each intermediate artifact and the final answer.

## Run

```sh
cd examples/langgraph/self-discover_self-discover
go run .
```

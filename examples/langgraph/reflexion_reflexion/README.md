# reflexion_reflexion

Go port of `langgraph/examples/reflexion/reflexion.ipynb`.

The notebook's graph is `draft -> execute_tools -> revise` with a conditional
edge on `revise` that loops back to the tool node until `MAX_ITERATIONS` is
exceeded. The Go port keeps exactly that loop and the message-accumulating
state channel:

- three nodes (`draft`, `execute_tools`, `revise`) with static edges and a
  conditional edge using the same trailing-message counter as the notebook;
- a `messages` channel using `channels.MessagesReducer` (Python's
  `add_messages`), with tool observations as tool-role messages;
  `types.END` for the notebook's `END`.

The structured `AnswerQuestion`/`ReviseAnswer` LLM calls are replaced by
deterministic offline stubs that emit an answer, a critique, and search
queries. The search tool is a safe in-memory lookup; no API key, network, or
code execution is involved.

## Run

```sh
cd examples/langgraph/reflexion_reflexion
go run .
```

# web-navigation_web_voyager

Go port of `examples/web-navigation/web_voyager.ipynb`.

## What the Go port demonstrates

The notebook's WebVoyager graph is rebuilt with the Go `StateGraph` API:

- an `agent` node predicts the next action (here: a deterministic sequence of
  `Google`, `Type`, `Click`, `Scroll`, then `ANSWER`)
- `AddConditionalEdges` routes the prediction either to `END` (for `ANSWER`)
  or to the matching action node
- each of the six tool nodes executes a `core/tools` tool through
  `langchain/tools.ToolNode` and stores its result as the new `observation`
- `update_scratchpad` appends the observation and loops back to `agent`
- the graph is compiled with a `checkpoint.NewMemorySaver()` and a `ThreadID`

The `Action:` decision protocol, valid-bounding-box descriptions and
step-numbered scratchpad all mirror the notebook.

## Offline safe substitution

The notebook launches headless Chromium with Playwright, renders screenshots,
calls a GPT-4 vision model, and navigates real websites (Google, arXiv, XKCD,
LangChain blog, Google Maps). This port uses none of that:

- the "browser" is a single string in graph state, transformed by pure Go
  functions
- bounding boxes are a fixed three-item slice
- the vision-language agent is a deterministic function of the step counter
- **no browser, screenshot, network request, or external service is used**

## Run

```sh
cd examples/langgraph/web-navigation_web_voyager
go run .
```

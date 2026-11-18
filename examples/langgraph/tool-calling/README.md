# tool-calling

Go port of `langgraph/examples/tool-calling.ipynb`.

The notebook explores the tool-calling surface: defining a `@tool`, invoking a
tool directly, invoking it from a tool-call dict, running a `create_react_agent`
with tools, `bind_tools` with `tool_choice` and `parallel_tool_calls`, and
`return_direct`. The Go version maps these to:

- `coretools.NewFunc` for the two-argument `multiply` tool;
- `multiply.Invoke` for a direct call;
- `langchain/tools.ToolNode` for turning a tool call into a `ToolMessage`;
- `langchain/agents.CreateAgent` for the ReAct loop;
- `language.BindToolsOptions` / `ToolChoiceAny` for forced tool choice and
  `parallel_tool_calls=false`;
- `coretools.Func.WithReturnDirect` for return-direct tools.

This example is fully offline. All models are deterministic in-process doubles;
no API key, network call, or external service is involved. The notebook's
provider-specific cells (dynamic tool configuration via runtime context,
`InjectedState`, store access, server-side web-search tools) have no offline
equivalent and are represented by the equivalent local tool surfaces above.

## Run

```sh
cd examples/langgraph/tool-calling
go run .
```

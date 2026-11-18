# chatbots_information-gather-prompting

Go port of `langgraph/examples/chatbots/information-gather-prompting.ipynb`.

The example builds a three-node `StateGraph` with a MemorySaver checkpointer:

- `info` prepends a system template that asks the user for the prompt's
  objective, variables, constraints, and requirements, then calls a model
  bound to a structured `PromptInstructions` tool;
- `add_tool_message` answers the tool call with a `ToolMessage`;
- `prompt` takes the conversation after the tool call and generates the final
  prompt template.

A conditional edge after `info` routes to `add_tool_message`, loops back to
`info` for another question, or ends the run. A scripted sequence of user
inputs drives the whole exchange offline.

## Offline stub

`ChatOpenAI` and the structured tool are replaced by a deterministic scripted
model and in-process tool data. No API key, network access, or service is
used.

## Run

```sh
cd examples/langgraph/chatbots_information-gather-prompting && go run .
```

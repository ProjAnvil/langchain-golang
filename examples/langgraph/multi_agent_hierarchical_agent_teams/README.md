# multi_agent_hierarchical_agent_teams

Go port of `langgraph/examples/multi_agent/hierarchical_agent_teams.ipynb`.

The notebook composes supervisor/worker teams at three levels: a research team,
a document-writing team, and a top-level graph that calls both teams as
subgraphs. The Go port keeps that topology:

- each team is a `StateGraph` with a supervisor node that returns a
  `*types.Command` routing to one worker at a time (or to `__end__`, the Go
  form of the notebook's `FINISH`);
- every worker reports back to its supervisor with `Command.Goto`;
- the top-level graph registers the two teams with `StateGraph.AddSubgraph`,
  so they share the parent's state and run as nested graphs.

All "LLM" decisions are deterministic offline stubs and the search/scrape/write
tools are canned strings. There is no API key, network access, or service, and
the Python notebook's `PythonREPL` code-execution tool is not reproduced.

## Run

```sh
cd examples/langgraph/multi_agent_hierarchical_agent_teams
go run .
```

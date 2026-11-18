# input_output_schema

Go port of `langgraph/examples/input_output_schema.ipynb`.

The notebook declares three TypedDicts: `InputState` (`question`), `OutputState`
(`answer`) and `OverallState` (both), and passes `input=`/`output=` to
`StateGraph`. Python then projects the input to `question` and the result to
`answer`.

## What this port demonstrates

- The same single-node graph topology: `START -> answer_node -> END`.
- `answer_node` reading `question` and returning both `answer` and `question`.
- Manual projection of the input and result to mirror Python's input/output
  schemas.

## Deviation from Python

The Go SDK's `NewStateGraph` does not take input/output schema arguments: state
is a flat `map[string]any` with per-key reducers. This port therefore emulates
the notebook's schema projection explicitly in Go code: `projectInput` keeps
only the `question` key before invoking, and `projectOutput` keeps only the
`answer` key after invoking. A run with an extra, non-schema input key is also
shown to confirm it is ignored.

## Run

```sh
cd examples/langgraph/input_output_schema
go run .
```

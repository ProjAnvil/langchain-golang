# state-model

Go port of `langgraph/examples/state-model.ipynb`.

The notebook uses Pydantic `BaseModel` state schemas to demonstrate input
validation, output validation, nested models, type coercion and message lists.

## What this port demonstrates

- Valid and invalid input for a state whose field `a` must be a string.
- A node that produces an invalid output (`a` as an int) and the resulting
  validation failure.
- A nested state shape (`text`, `count`, `nested.value`) processed by a node.
- Coercion of string inputs to `int` / `bool`, plus a coercion failure.
- A message-list state where a node appends an AI message.

## Deviation from Python

The Go SDK has no Pydantic-style schema object: `NewStateGraph` takes only a
plain `map[string]any` state. This port therefore expresses the same checks as
explicit Go helpers (`validateStringA`, `coerceInt`, `coerceBool`,
`validateNested`): input is validated before `Invoke`, and node output is
validated in the consuming node. Where Python raises immediately on a bad node
update, Go surfaces the error when the next node inspects the value. The
observable behavior (a descriptive error for invalid data, coerced values for
valid strings) is preserved.

## Run

```sh
cd examples/langgraph/state-model
go run .
```

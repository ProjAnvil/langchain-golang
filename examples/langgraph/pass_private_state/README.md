# pass_private_state

Go port of `langgraph/examples/pass_private_state.ipynb`.

The notebook uses node-level TypedDicts so a value written by `node_1`
(`private_data`) is visible only to `node_2`; `node_3` declares only the
overall state and cannot see it.

## What this port demonstrates

- The same three-node chain: `START -> node_1 -> node_2 -> node_3 -> END`.
- `node_1` writing `private_data`, `node_2` consuming it, and `node_3` working
  only with the public `a` key.
- Each node prints the input it receives and the update it returns, making the
  visibility boundary explicit.

## Deviation from Python

The Go SDK state is a flat `map[string]any`: there is no node-level schema that
hides a channel from a downstream node. To reproduce the notebook's behavior,
`node_2` consumes the private key by overwriting it with an empty value, so
`node_3` observes no private data. The key still exists in the final flat state
with an empty value — Python's private channel would be dropped entirely. The
semantic point (only `node_2` can read the private data) is preserved.

## Run

```sh
cd examples/langgraph/pass_private_state
go run .
```

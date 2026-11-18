# map-reduce

Go port of `langgraph/examples/map-reduce.ipynb`.

The notebook fans out from one `generate_topics` node to many `generate_joke`
invocations using `Send`, then reduces the generated jokes with an
`operator.add` channel and picks the best one.

## What this port demonstrates

- `types.Send` returned from a conditional edge to dynamically fan out to
  `generate_joke`, one invocation per subject (Python's `continue_to_jokes`).
- `channels.AppendSliceReducer` on the `jokes` key: the Go form of
  `Annotated[list, operator.add]`, merging every parallel node's joke into one
  list (the "reduce" step).
- The full topology: `START -> generate_topics -> (Send) -> generate_joke ->
  best_joke -> END`.
- Both `Invoke` (final state) and `Stream` (updates, showing one chunk per
  fanned-out joke).

## Deviation from Python

The Go executor dispatches one `best_joke` task per fanned-out `generate_joke`
branch, whereas Python deduplicates them into a single execution. All branches
compute the same answer, and a last-write-wins reducer on `best_selected_joke`
merges their writes, so the final state is identical.

## Offline stub

The notebook's three `ChatAnthropic` structured-output calls are replaced with
deterministic in-process doubles: a fixed subject list for `animals`, one joke
template per subject, and a fixed "best joke" selector. No API key, network,
or external service is used.

## Run

```sh
cd examples/langgraph/map-reduce
go run .
```

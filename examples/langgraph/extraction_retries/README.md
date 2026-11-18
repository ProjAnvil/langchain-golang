# extraction_retries

Go port of `langgraph/examples/extraction/retries.ipynb`.

The notebook builds a `bind_validator_with_retries` graph: a model emits a
structured tool call, a `ValidationNode` validates it, a field validator
requires the answer to mention "Llama V3", and on failure a fallback model is
asked again until a valid answer appears or `max_attempts` is exhausted; a
finalizer then returns only the final AI message. The Go version rebuilds that
graph with `langgraph/graph.StateGraph`, `langgraph/prebuilt.NewValidationNode`,
a custom messages reducer that supports the "finalize" overwrite, and an
integer sum reducer for `attempt_number`.

Two scenarios run offline: one where the fallback model returns a valid answer
and the graph finalizes it, and one where every attempt fails and the router
raises the max-attempts error.

This example is fully offline. All models are deterministic in-process doubles
standing in for the notebook's `ChatAnthropic`; no API key, network call, or
external service is involved. Deviations: the notebook's pydantic
`field_validator` is expressed as a Go string check, and the JSONPatch-based
healing strategy (which needs an offline JSON-patch model) is represented by
the fallback model emitting a corrected full tool call rather than by applying
an RFC-6902 patch.

## Run

```sh
cd examples/langgraph/extraction_retries
go run .
```

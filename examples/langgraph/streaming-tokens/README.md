# streaming-tokens

Go port of `langgraph/examples/streaming-tokens.ipynb`.

A single node runs a "joke" model and a "poem" model and streams their text
token by token. The notebook reads every token in `messages` mode, inspects the
chunk metadata (which carries model tags in Python), and later writes tagged
(chunk, metadata) pairs into a `custom` stream, filtering on the `poem` tag.
The Go port uses fake models that emit `EventChatModelStream` chunks for
`StreamMessages` and tagged custom payloads for `StreamCustom`.

## Offline note

`ChatOpenAI` and `AsyncOpenAI` are replaced by deterministic local fakes. No
API key, network, or external service is used.

## Deviation

Go's `MessageChunk.Metadata` carries `langgraph_node`, `langgraph_step`, and
`langgraph_checkpoint_ns`, but not per-model tags. The notebook's tag-filtered
messages cell is therefore demonstrated with the equivalently tagged custom
stream (`StreamCustom`), while the messages-mode run shows every token.

## Run

```sh
cd examples/langgraph/streaming-tokens
go run .
```

# persistence_redis

Go port of `langgraph/examples/persistence_redis.ipynb`.

Source notebook: `langgraph/examples/persistence_redis.ipynb`.

What the Go port demonstrates: the notebook implements custom `RedisSaver` and
`AsyncRedisSaver` checkpoint savers, drives a `create_react_agent` through
several `thread_id`s, and reads checkpoints with `get(config)`,
`get_tuple(config)` and `list(config)`. Here those three read patterns map to
`CompiledGraph.GetState` and `GetStateHistory`, and the agent is a single
deterministic weather node.

Explicit offline note: **no Redis service is contacted.** The notebook's
hand-written Redis savers, `Redis` connection and `from_conn_info` context
managers are replaced by one deterministic `checkpoint.NewMemorySaver()` stub.
The `ChatOpenAI` model and all network calls are intentionally not reproduced;
all checkpoints live only in process memory.

## Run

```sh
cd examples/langgraph/persistence_redis
go run .
```

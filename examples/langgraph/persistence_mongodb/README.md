# persistence_mongodb

Go port of `langgraph/examples/persistence_mongodb.ipynb`.

Source notebook: `langgraph/examples/persistence_mongodb.ipynb`.

What the Go port demonstrates: the notebook's core checkpoint lifecycle around
`MongoDBSaver` -- create a saver, build an agent, invoke several `thread_id`s
through it, and fetch the latest checkpoint with `get_tuple(config)` / list
them with the saver API. Here the agent is replaced by a single deterministic
weather node (the `@tool get_weather` logic is kept as a plain function), and
`CompiledGraph.GetState` / `GetStateHistory` are the Go equivalents of
`get_tuple` and `list`.

Explicit offline note: **no MongoDB service is contacted.** `MongoDBSaver`
(including its async and `from_conn_string` variants) is replaced by a
deterministic `checkpoint.NewMemorySaver()` stub. The `MONGODB_URI`, OpenAI
`ChatOpenAI` model and network calls from the notebook are intentionally not
reproduced; all checkpoints live only in process memory.

## Run

```sh
cd examples/langgraph/persistence_mongodb
go run .
```

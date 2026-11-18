# Examples

All examples are standalone Go projects. Every example directory has its own
`go.mod`, `go.sum`, `main.go`, and `README.md`. Category examples use:

```go
replace github.com/projanvil/langchain-golang => ../../..
```

Top-level `deepagent/` uses `../..` because it sits one level shallower.

## Categories

| Category | Count | What it covers |
| --- | ---: | --- |
| [`langchain/`](langchain/) | 8 | LangChain core, agents, middleware, retrieval, RAG, SQL, MCP, provider integrations |
| [`deepagent/`](deepagent/) | 1 | `deepagents` runtime (single standalone project) |
| [`langgraph/`](langgraph/) | 77 | Native LangGraph runtime examples plus 73 Go ports of Python `langgraph/examples` |

## Run one example

```sh
cd examples/langchain/quickstart
go run .
```

## Validate all examples

```sh
./examples/validate.sh
```

Result summary: [`VALIDATION.md`](VALIDATION.md).

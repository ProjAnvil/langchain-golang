# LangGraph examples in Go

This directory contains 77 independent Go modules:

- 4 native Go SDK examples: `fault-tolerance`, `hitl-interrupt`,
  `subgraph-resume`, `trace-policy`
- 73 Go ports of every notebook under Python `langgraph/examples/`

Each example is an independent Go module with its own `go.mod`, so it can be
copied or run on its own.

## Layout

Native Go SDK examples keep their topic names at this level.

One Python notebook maps to one flat directory. Nested notebook paths use
`_` as a separator, for example:

- `langgraph/examples/rag/langgraph_crag.ipynb` → `rag_langgraph_crag/`
- `langgraph/examples/tutorials/tnt-llm/tnt-llm.ipynb` → `tutorials_tnt-llm_tnt-llm/`

Every module has:

- `go.mod` with `replace github.com/projanvil/langchain-golang => ../../..`
- `main.go` with an offline-runnable Go SDK program
- `README.md` with the source path, the demonstrated concept, and substitution notes

## Run one example

```sh
cd examples/langgraph/branching
go run .
```

No API keys, network access, databases, or browser services are required.
Notebooks that originally needed a model provider, LangSmith, LangGraph
Cloud, MongoDB, Postgres, Redis, Pinecone, Cohere, a local model download,
SQL execution, or browser automation are ported with deterministic in-memory
equivalents. See each example README for the exact substitution.

## Source map

| Go example | Python source |
| --- | --- |
| `async` | `langgraph/examples/async.ipynb` |
| `branching` | `langgraph/examples/branching.ipynb` |
| `chatbot-simulation-evaluation_agent-simulation-evaluation` | `langgraph/examples/chatbot-simulation-evaluation/agent-simulation-evaluation.ipynb` |
| `chatbot-simulation-evaluation_langsmith-agent-simulation-evaluation` | `langgraph/examples/chatbot-simulation-evaluation/langsmith-agent-simulation-evaluation.ipynb` |
| `chatbots_information-gather-prompting` | `langgraph/examples/chatbots/information-gather-prompting.ipynb` |
| `cloud_examples_langgraph_to_langgraph_cloud` | `langgraph/examples/cloud_examples/langgraph_to_langgraph_cloud.ipynb` |
| `code_assistant_langgraph_code_assistant` | `langgraph/examples/code_assistant/langgraph_code_assistant.ipynb` |
| `code_assistant_langgraph_code_assistant_mistral` | `langgraph/examples/code_assistant/langgraph_code_assistant_mistral.ipynb` |
| `configuration` | `langgraph/examples/configuration.ipynb` |
| `create-react-agent-hitl` | `langgraph/examples/create-react-agent-hitl.ipynb` |
| `create-react-agent-memory` | `langgraph/examples/create-react-agent-memory.ipynb` |
| `create-react-agent-system-prompt` | `langgraph/examples/create-react-agent-system-prompt.ipynb` |
| `create-react-agent` | `langgraph/examples/create-react-agent.ipynb` |
| `customer-support_customer-support` | `langgraph/examples/customer-support/customer-support.ipynb` |
| `extraction_retries` | `langgraph/examples/extraction/retries.ipynb` |
| `human_in_the_loop_wait-user-input` | `langgraph/examples/human_in_the_loop/wait-user-input.ipynb` |
| `input_output_schema` | `langgraph/examples/input_output_schema.ipynb` |
| `lats_lats` | `langgraph/examples/lats/lats.ipynb` |
| `llm-compiler_LLMCompiler` | `langgraph/examples/llm-compiler/LLMCompiler.ipynb` |
| `map-reduce` | `langgraph/examples/map-reduce.ipynb` |
| `memory_add-summary-conversation-history` | `langgraph/examples/memory/add-summary-conversation-history.ipynb` |
| `memory_delete-messages` | `langgraph/examples/memory/delete-messages.ipynb` |
| `memory_manage-conversation-history` | `langgraph/examples/memory/manage-conversation-history.ipynb` |
| `multi_agent_hierarchical_agent_teams` | `langgraph/examples/multi_agent/hierarchical_agent_teams.ipynb` |
| `multi_agent_multi-agent-collaboration` | `langgraph/examples/multi_agent/multi-agent-collaboration.ipynb` |
| `node-retries` | `langgraph/examples/node-retries.ipynb` |
| `pass-config-to-tools` | `langgraph/examples/pass-config-to-tools.ipynb` |
| `pass-run-time-values-to-tools` | `langgraph/examples/pass-run-time-values-to-tools.ipynb` |
| `pass_private_state` | `langgraph/examples/pass_private_state.ipynb` |
| `persistence` | `langgraph/examples/persistence.ipynb` |
| `persistence_mongodb` | `langgraph/examples/persistence_mongodb.ipynb` |
| `persistence_postgres` | `langgraph/examples/persistence_postgres.ipynb` |
| `persistence_redis` | `langgraph/examples/persistence_redis.ipynb` |
| `plan-and-execute_plan-and-execute` | `langgraph/examples/plan-and-execute/plan-and-execute.ipynb` |
| `rag_langgraph_adaptive_rag` | `langgraph/examples/rag/langgraph_adaptive_rag.ipynb` |
| `rag_langgraph_adaptive_rag_cohere` | `langgraph/examples/rag/langgraph_adaptive_rag_cohere.ipynb` |
| `rag_langgraph_adaptive_rag_local` | `langgraph/examples/rag/langgraph_adaptive_rag_local.ipynb` |
| `rag_langgraph_agentic_rag` | `langgraph/examples/rag/langgraph_agentic_rag.ipynb` |
| `rag_langgraph_crag` | `langgraph/examples/rag/langgraph_crag.ipynb` |
| `rag_langgraph_crag_local` | `langgraph/examples/rag/langgraph_crag_local.ipynb` |
| `rag_langgraph_self_rag` | `langgraph/examples/rag/langgraph_self_rag.ipynb` |
| `rag_langgraph_self_rag_local` | `langgraph/examples/rag/langgraph_self_rag_local.ipynb` |
| `rag_langgraph_self_rag_pinecone_movies` | `langgraph/examples/rag/langgraph_self_rag_pinecone_movies.ipynb` |
| `react-agent-from-scratch` | `langgraph/examples/react-agent-from-scratch.ipynb` |
| `react-agent-structured-output` | `langgraph/examples/react-agent-structured-output.ipynb` |
| `recursion-limit` | `langgraph/examples/recursion-limit.ipynb` |
| `reflection_reflection` | `langgraph/examples/reflection/reflection.ipynb` |
| `reflexion_reflexion` | `langgraph/examples/reflexion/reflexion.ipynb` |
| `rewoo_rewoo` | `langgraph/examples/rewoo/rewoo.ipynb` |
| `run-id-langsmith` | `langgraph/examples/run-id-langsmith.ipynb` |
| `self-discover_self-discover` | `langgraph/examples/self-discover/self-discover.ipynb` |
| `state-model` | `langgraph/examples/state-model.ipynb` |
| `storm_storm` | `langgraph/examples/storm/storm.ipynb` |
| `stream-multiple` | `langgraph/examples/stream-multiple.ipynb` |
| `stream-updates` | `langgraph/examples/stream-updates.ipynb` |
| `stream-values` | `langgraph/examples/stream-values.ipynb` |
| `streaming-content` | `langgraph/examples/streaming-content.ipynb` |
| `streaming-events-from-within-tools-without-langchain` | `langgraph/examples/streaming-events-from-within-tools-without-langchain.ipynb` |
| `streaming-events-from-within-tools` | `langgraph/examples/streaming-events-from-within-tools.ipynb` |
| `streaming-from-final-node` | `langgraph/examples/streaming-from-final-node.ipynb` |
| `streaming-subgraphs` | `langgraph/examples/streaming-subgraphs.ipynb` |
| `streaming-tokens-without-langchain` | `langgraph/examples/streaming-tokens-without-langchain.ipynb` |
| `streaming-tokens` | `langgraph/examples/streaming-tokens.ipynb` |
| `subgraph-transform-state` | `langgraph/examples/subgraph-transform-state.ipynb` |
| `subgraph` | `langgraph/examples/subgraph.ipynb` |
| `subgraphs-manage-state` | `langgraph/examples/subgraphs-manage-state.ipynb` |
| `tool-calling-errors` | `langgraph/examples/tool-calling-errors.ipynb` |
| `tool-calling` | `langgraph/examples/tool-calling.ipynb` |
| `tutorials_sql-agent` | `langgraph/examples/tutorials/sql-agent.ipynb` |
| `tutorials_tnt-llm_tnt-llm` | `langgraph/examples/tutorials/tnt-llm/tnt-llm.ipynb` |
| `usaco_usaco` | `langgraph/examples/usaco/usaco.ipynb` |
| `visualization` | `langgraph/examples/visualization.ipynb` |
| `web-navigation_web_voyager` | `langgraph/examples/web-navigation/web_voyager.ipynb` |

## Validate all examples

```sh
./examples/validate.sh
```

The script formats, tidies, builds, and runs every standalone module. It stops
on the first failure and prints the example path. Latest result: 86/86 passed;
see [`VALIDATION.md`](../VALIDATION.md).

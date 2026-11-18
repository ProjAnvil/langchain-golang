# chatbot-simulation-evaluation_langsmith-agent-simulation-evaluation

Go port of
`langgraph/examples/chatbot-simulation-evaluation/langsmith-agent-simulation-evaluation.ipynb`.

The example keeps the notebook's red-team evaluation structure:

- an in-memory dataset cloned from a public URL (the offline analogue of
  `client.clone_public_dataset`);
- a `create_chat_simulator` graph that passes messages between an adversarial
  simulated customer and the assistant under test;
- a deterministic `did_resist` evaluator returning the same `score`/`comment`
  shape as the notebook's structured-output LLM evaluator;
- an `evaluate` loop that records one run ID per dataset row, mirroring
  `client.evaluate`.

One example is resisted (score 1) and one is not (score 0), so the aggregate
score is meaningful.

## Offline stub

The LangSmith `Client`, OpenAI assistant, simulated-user LLM, dataset URL, and
GPT-4o evaluator are all replaced with deterministic in-process equivalents.
No API key, network access, or service is used.

## Run

```sh
cd examples/langgraph/chatbot-simulation-evaluation_langsmith-agent-simulation-evaluation && go run .
```

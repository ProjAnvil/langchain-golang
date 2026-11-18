# chatbot-simulation-evaluation_agent-simulation-evaluation

Go port of
`langgraph/examples/chatbot-simulation-evaluation/agent-simulation-evaluation.ipynb`.

A two-node `StateGraph` simulates an airline support conversation:

- `chat_bot` prepends a support-agent system message and answers from a
  deterministic queue (the stand-in for `my_chat_bot` calling OpenAI);
- `user` swaps the roles of the transcript, drives a scripted customer who is
  asking for a refund of a five-year-old Alaska trip, and flips the reply back
  into a `HumanMessage`;
- a conditional edge stops when the customer says `FINISHED` or the transcript
  exceeds six messages.

The run is observed with `StreamValues`.

## Offline stub

`openai.chat.completions.create` and `ChatOpenAI` are replaced with scripted
in-process models. No API key, network access, or service is used.

## Run

```sh
cd examples/langgraph/chatbot-simulation-evaluation_agent-simulation-evaluation && go run .
```

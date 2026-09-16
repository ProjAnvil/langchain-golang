# gemini-agent

A tool-calling agent on Google Gemini (`partners/gemini`, `gemini-2.5-flash`), with a local word-count tool exercised through a real tool-call round trip.

## Run

```sh
GEMINI_API_KEY=... go run ./examples/gemini-agent
```

Required env: `GEMINI_API_KEY` (or `GOOGLE_API_KEY`). Without a key the example prints an explanation and exits 0.

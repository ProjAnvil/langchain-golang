# Contributing

Thanks for contributing! New partner integrations are especially welcome (Google Gemini, AWS Bedrock, more vector stores, ...).

## Getting started

1. Fork the repository and create a feature branch (`git checkout -b feat/your-feature`).
2. Make your change with tests.
3. Ensure the full gate passes locally (mirrors CI):

   ```bash
   go build ./... && go vet ./... && go test -race ./...
   make test-sqlite test-redis test-postgres   # nested checkpoint modules (offline; postgres downloads embedded binaries once)
   golangci-lint run ./...                      # same config as CI (.golangci.yml)
   ```

4. Match the existing code style and Python-parity conventions.
5. Submit a pull request.

## Conventions

- **Python is authoritative**: this is a port of LangChain/LangGraph. When in doubt, check what the Python source does and cite the file:line in a comment.
- **Zero breaking changes to shipped interfaces** within a minor: add optional capability interfaces (see `core/vectorstores` `TextAdder` or `core/retrievers` searcher interfaces for the pattern) instead of widening existing ones.
- **Trust `go build/vet/test`**, not editor diagnostics (gopls may show false positives).
- Every package should have compile-checked examples in `example_test.go`.
- Bilingual docs: add both `guide.md` and `guide.zh-CN.md` under `docs/usage/` for new user-facing features.
- Tests use in-process fakes (`httptest`, miniredis, embedded postgres) — CI never needs docker or live API keys. Integration tests behind the `integration` build tag read keys from `.env` (see `.env.example`).
- Commit messages use conventional prefixes (`feat:`, `fix:`, `docs:`, `chore:`, `test:`).
- New partner adapters must pass the `standardtests` conformance suites.
- Update `CHANGELOG.md` under `## [Unreleased]` for user-visible changes.

## Testing

- `make test` — offline unit suite (root module)
- `make test-sqlite` / `make test-redis` / `make test-postgres` — nested checkpoint saver modules
- `make test-integration` — live-provider tests (needs `.env`)
- `make vet-integration` — type-check integration tests without network

## Security

See [SECURITY.md](SECURITY.md). Do not open public issues for vulnerabilities.

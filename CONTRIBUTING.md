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
- `make check-dep-alignment` — verify the root go.mod pins of pgx/go-redis match the nested checkpoint modules (enforced in CI; keep versions in sync when bumping either side)

### Coverage gate (95% red line)

`make coverage` runs the **parity catch-up packages** with per-package
coverprofiles, prints the per-package table plus the merged total, and
**exits non-zero when any package falls below 95%** (`COVERAGE_THRESHOLD`,
overridable per invocation, e.g. `make coverage COVERAGE_THRESHOLD=90`).
`make coverage-html` re-runs the gate and renders the merged profile to
`coverage.html`. CI enforces the same gate in the `coverage` job and posts
the table to the job summary.

The 95% red line applies to every package in `COVERAGE_PKGS` (Makefile):

- `core/vectorstores`, `core/retrievers`, `core/documentloaders`, `core/callbacks`
- `partners/pgvector`, `partners/redisvector`, `partners/cohere`, `partners/jina`, `partners/mcp`, `partners/gemini`
- `langchain/toolkits/sqltoolkit`
- `langgraph/graph`

When touching these packages, check `make coverage` locally before pushing;
if a change cannot reasonably reach 95% (e.g. a branch that needs a live
server), call it out explicitly in the PR and add it to the e2e/integration
suites instead.

## Security

See [SECURITY.md](SECURITY.md). Do not open public issues for vulnerabilities.

.PHONY: test test-sqlite test-integration vet-integration

## Run the offline unit test suite (default; no network, no keys).
test:
	go test ./...

## Run the nested SQLite checkpoint saver module's tests (its own go.mod;
## the root `test` target does not cross the nested-module boundary).
test-sqlite:
	cd langgraph/checkpoint/sqlite && go test ./...

## Run integration tests against real providers.
## Requires a .env file (copy .env.example) with at least one *_ENABLED=1.
test-integration:
	@test -f .env || { \
		echo "ERROR: .env not found. Copy .env.example to .env and fill in real values."; \
		exit 1; \
	}
	@set -a; . ./.env; set +a; \
	echo "→ integration tests (OPENAI_ENABLED=$${OPENAI_ENABLED:-0} ANTHROPIC_ENABLED=$${ANTHROPIC_ENABLED:-0})"; \
	go test -tags=integration -v -timeout 15m ./integration/...

## Type-check integration tests without running them (no network needed).
vet-integration:
	go vet -tags=integration ./integration/...

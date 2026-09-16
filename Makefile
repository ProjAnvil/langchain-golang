.PHONY: test test-sqlite test-postgres test-redis test-integration vet-integration check-dep-alignment

## Run the offline unit test suite (default; no network, no keys).
test:
	go test ./...

## Run the nested SQLite checkpoint saver module's tests (its own go.mod;
## the root `test` target does not cross the nested-module boundary).
test-sqlite:
	cd langgraph/checkpoint/sqlite && go test ./...

## Run the nested Postgres checkpoint saver module's tests (its own go.mod).
## Spins up an in-process embedded Postgres (first run downloads ~30MB to
## ~/.embedded-postgres-go — cache that directory in CI). Use -short to skip:
##   cd langgraph/checkpoint/postgres && go test -short ./...
test-postgres:
	cd langgraph/checkpoint/postgres && go test ./...

## Run the nested Redis checkpoint saver module's tests (its own go.mod).
## Fully offline: tests run against miniredis, no Redis server needed.
test-redis:
	cd langgraph/checkpoint/redis && go test ./...

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

## Verify the root go.mod pins of pgx/go-redis match the nested checkpoint
## modules (spec §10 dependency policy: partners/pgvector and
## partners/redisvector must not drift from langgraph/checkpoint/{postgres,redis}).
## Exits non-zero on any mismatch.
check-dep-alignment:
	@set -e; \
	for pair in \
		"langgraph/checkpoint/postgres/go.mod github.com/jackc/pgx/v5" \
		"langgraph/checkpoint/redis/go.mod github.com/redis/go-redis/v9"; do \
		set -- $$pair; nested=$$1; dep=$$2; \
		root_ver=$$(awk -v d="$$dep" '$$1==d {print $$2}' go.mod); \
		nested_ver=$$(awk -v d="$$dep" '$$1==d {print $$2}' $$nested); \
		if [ -z "$$root_ver" ]; then \
			echo "ERROR: $$dep is missing from root go.mod"; exit 1; \
		fi; \
		if [ -z "$$nested_ver" ]; then \
			echo "ERROR: $$dep is missing from $$nested"; exit 1; \
		fi; \
		if [ "$$root_ver" != "$$nested_ver" ]; then \
			echo "ERROR: $$dep version mismatch: root go.mod has $$root_ver but $$nested has $$nested_ver"; exit 1; \
		fi; \
		echo "OK: $$dep $$root_ver (root == $$nested)"; \
	done

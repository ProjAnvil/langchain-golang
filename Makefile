.PHONY: test test-sqlite test-postgres test-redis test-integration vet-integration check-dep-alignment coverage coverage-html

## Statement-coverage gate for the 2026-09 parity catch-up packages.
## COVERAGE_THRESHOLD is the per-package red line in percent (>= passes).
COVERAGE_THRESHOLD ?= 95
COVERAGE_PKGS := \
	core/vectorstores \
	core/retrievers \
	core/documentloaders \
	core/callbacks \
	partners/pgvector \
	partners/redisvector \
	partners/cohere \
	partners/jina \
	partners/mcp \
	partners/gemini \
	langchain/toolkits/sqltoolkit \
	langgraph/graph
COVERAGE_DIR := .coverage
COVERAGE_OUT := coverage.out
COVERAGE_HTML := coverage.html

## Run the parity catch-up packages with per-package coverprofiles, print the
## per-package coverage table plus the merged total, and exit non-zero when
## any package is below COVERAGE_THRESHOLD percent.
coverage:
	@set -e; \
	rm -rf $(COVERAGE_DIR); mkdir -p $(COVERAGE_DIR); \
	printf 'mode: set\n' > $(COVERAGE_OUT); \
	fail=0; \
	printf '%-38s %10s\n' PACKAGE COVERAGE; \
	for pkg in $(COVERAGE_PKGS); do \
		prof="$(COVERAGE_DIR)/$$(printf '%s' "$$pkg" | tr / _).out"; \
		if ! log=$$(go test -count=1 -coverprofile="$$prof" "./$$pkg/" 2>&1); then \
			printf '%-38s %10s\n' "$$pkg" TEST-FAIL; \
			printf '%s\n' "$$log" | sed 's/^/    /' | tail -n 20; \
			fail=1; continue; \
		fi; \
		set -- $$(awk -F' ' 'NR>1 {t+=$$2; if ($$3>0) c+=$$2} END {pct=(t==0)?100:100*c/t; printf "%.1f %d", pct, (pct < $(COVERAGE_THRESHOLD))}' "$$prof"); \
		pct=$$1; below=$$2; \
		printf '%-38s %9s%%\n' "$$pkg" "$$pct"; \
		awk 'NR>1' "$$prof" >> $(COVERAGE_OUT); \
		if [ "$$below" = "1" ]; then \
			printf '%-38s %10s\n' '' "below $(COVERAGE_THRESHOLD)%"; \
			fail=1; \
		fi; \
	done; \
	total=$$(go tool cover -func=$(COVERAGE_OUT) | awk '/^total:/ {sub(/%/, "", $$3); print $$3}'); \
	printf '%-38s %9s%%\n' 'TOTAL (merged)' "$$total"; \
	if [ "$$fail" = "1" ]; then \
		echo "ERROR: coverage gate failed: every package in COVERAGE_PKGS must be >= $(COVERAGE_THRESHOLD)%"; \
		exit 1; \
	fi

## Render the merged profile produced by `make coverage` to coverage.html.
coverage-html: coverage
	@go tool cover -html=$(COVERAGE_OUT) -o $(COVERAGE_HTML)
	@echo "wrote $(COVERAGE_HTML)"

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

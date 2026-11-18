#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
count=0

while IFS= read -r mod; do
  dir="$(dirname "$mod")"
  name="${dir#$script_dir/}"
  printf '[%d] %s\n' "$((count + 1))" "$name"
  (
    cd "$dir"
    gofmt -w *.go
    go mod tidy
    go vet ./...
    go build -o /dev/null .
    go run . >/dev/null
  )
  count=$((count + 1))
done < <(find "$script_dir" -mindepth 1 -maxdepth 3 -name go.mod | sort)

printf 'validated %d standalone examples\n' "$count"

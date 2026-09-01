module github.com/projanvil/langchain-golang/langgraph/checkpoint/redis

go 1.26

replace github.com/projanvil/langchain-golang => ../../../

require (
	github.com/alicebob/miniredis/v2 v2.38.0
	github.com/projanvil/langchain-golang v0.6.4
	github.com/redis/go-redis/v9 v9.22.0
)

require (
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/yuin/gopher-lua v1.1.1 // indirect
	go.uber.org/atomic v1.11.0 // indirect
	golang.org/x/sys v0.33.0 // indirect
)

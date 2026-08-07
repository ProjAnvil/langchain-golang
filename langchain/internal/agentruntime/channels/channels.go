// Package channels is a thin alias layer over
// `github.com/projanvil/langchain-golang/langgraph/channels`; see that
// package for documentation. New code should import langgraph/channels
// directly.
package channels

import "github.com/projanvil/langchain-golang/langgraph/channels"

type Reducer = channels.Reducer

var (
	LastValueReducer   = channels.LastValueReducer
	AppendSliceReducer = channels.AppendSliceReducer
	MessagesReducer    = channels.MessagesReducer
)

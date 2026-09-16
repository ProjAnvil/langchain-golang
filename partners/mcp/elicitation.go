package mcp

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/projanvil/langchain-golang/langgraph/graph"
)

// ErrElicitationCanceled reports that a human answered an elicitation
// request with the cancel action: the whole tool call is abandoned (upstream
// semantics) rather than completed with a declined answer.
var ErrElicitationCanceled = errors.New("mcp: elicitation canceled")

// ElicitationAction is the human's answer to one elicitation request.
type ElicitationAction string

const (
	// ElicitationAccept accepts the request; Content must match the
	// requested schema.
	ElicitationAccept ElicitationAction = "accept"
	// ElicitationDecline refuses the request; the tool call continues.
	ElicitationDecline ElicitationAction = "decline"
	// ElicitationCancel abandons the whole tool call.
	ElicitationCancel ElicitationAction = "cancel"
)

// ElicitationAnswer is one keyed answer inside a resume payload.
type ElicitationAnswer struct {
	Action  ElicitationAction `json:"action"`
	Content any               `json:"content,omitempty"`
}

// ElicitationResponses is the typed resume payload answering an elicitation
// interrupt: answers are keyed by the server's request key, mirroring the
// upstream Command(resume={"responses": {key: answer}}) shape.
//
// Resume interplay: a plain map[string]any Resume is interpreted by the
// graph as interrupt-NS/ID addressing, so the wire-map form of this payload
// must be nested under the interrupt's NS or ID. The typed forms
// (ElicitationResponses, map[string]ElicitationAnswer) are not
// map[string]any and are fed directly to the single pending elicitation
// interrupt — the recommended resume form:
//
//	resume := mcp.ElicitationResponses{Responses: map[string]mcp.ElicitationAnswer{
//		"confirm": {Action: mcp.ElicitationAccept, Content: map[string]any{"amount": 42}},
//	}}
//	graph.Options{ThreadID: thread, Resume: resume}
type ElicitationResponses struct {
	Responses map[string]ElicitationAnswer `json:"responses"`
}

// DecodeElicitationResponses decodes an elicitation resume value into its
// keyed answers, accepting the typed forms a Go caller passes directly and
// the map wire forms left behind by JSON checkpoint round-trips (either
// snake_case or capitalized spellings). Anything else is an error.
func DecodeElicitationResponses(v any) (map[string]ElicitationAnswer, error) {
	switch resp := v.(type) {
	case nil:
		return nil, fmt.Errorf("mcp: elicitation resume value is required")
	case ElicitationResponses:
		return resp.Responses, nil
	case *ElicitationResponses:
		if resp == nil {
			return nil, fmt.Errorf("mcp: elicitation resume value is required")
		}
		return resp.Responses, nil
	case map[string]ElicitationAnswer:
		return resp, nil
	case map[string]any:
		raw, ok := resp["responses"]
		if !ok {
			raw, ok = resp["Responses"]
		}
		if !ok {
			return nil, fmt.Errorf("mcp: elicitation resume map without a \"responses\" key")
		}
		answers, ok := raw.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("mcp: elicitation \"responses\" must be an object, got %T", raw)
		}
		out := make(map[string]ElicitationAnswer, len(answers))
		for key, answer := range answers {
			decoded, err := decodeElicitationAnswer(answer)
			if err != nil {
				return nil, fmt.Errorf("mcp: elicitation answer %q: %w", key, err)
			}
			out[key] = decoded
		}
		return out, nil
	default:
		return nil, fmt.Errorf("mcp: cannot decode elicitation resume from %T", v)
	}
}

func decodeElicitationAnswer(v any) (ElicitationAnswer, error) {
	switch answer := v.(type) {
	case ElicitationAnswer:
		return answer, nil
	case map[string]any:
		out := ElicitationAnswer{}
		raw, ok := answer["action"]
		if !ok {
			raw, ok = answer["Action"]
		}
		if !ok {
			return out, fmt.Errorf("missing \"action\" key")
		}
		action, ok := raw.(string)
		if !ok {
			return out, fmt.Errorf("\"action\" must be a string, got %T", raw)
		}
		switch ElicitationAction(action) {
		case ElicitationAccept, ElicitationDecline, ElicitationCancel:
			out.Action = ElicitationAction(action)
		default:
			return out, fmt.Errorf("unknown action %q", action)
		}
		if content, ok := answer["content"]; ok {
			out.Content = content
		} else if content, ok := answer["Content"]; ok {
			out.Content = content
		}
		return out, nil
	default:
		return ElicitationAnswer{}, fmt.Errorf("cannot decode from %T", v)
	}
}

// toMCP converts an answer to the mcp-go elicitation result the server
// receives.
func (a ElicitationAnswer) toMCP() *mcp.ElicitationResult {
	result := &mcp.ElicitationResult{}
	result.Action = mcp.ElicitationResponseAction(a.Action)
	result.Content = a.Content
	return result
}

// elicitBatchKey carries the per-invocation elicitation batch on the tool
// call's context. mcp-go fulfills modern multi-round-trip input requests
// with handlers invoked on contexts derived from the CallTool context, so a
// context value routes each request back to the invocation it belongs to —
// transport-independent.
type elicitBatchKey struct{}

// elicitationBridge implements mcp-go's client elicitation handler,
// translating server input requests into LangGraph interrupts.
type elicitationBridge struct{}

// Elicit answers one server elicitation request.
//
// Modern multi-round-trip requests arrive while one of our tool invocations
// is in flight: the request registers on that invocation's batch and blocks
// until the invocation's wrapper delivers the answer produced by a
// graph.Interrupt on the node's own goroutine.
//
// Legacy handshake-session requests (server-initiated, pre-2026-07-28)
// dispatch outside our call stack. On in-process connections the dispatch is
// synchronous on the caller's goroutine, so the interrupt is bridged
// directly; on remote transports mcp-go dispatches from the transport's read
// loop, where a panic-based interrupt cannot unwind to the graph run — those
// fail with a descriptive error instead (upstream likewise supports no
// legacy-session elicitation).
func (elicitationBridge) Elicit(ctx context.Context, request mcp.ElicitationRequest) (*mcp.ElicitationResult, error) {
	if batch, ok := ctx.Value(elicitBatchKey{}).(*elicitBatch); ok {
		return batch.wait(ctx, request.Params)
	}
	if graph.InterruptSupported(ctx) {
		key := request.Params.ElicitationID
		if key == "" {
			key = "request"
		}
		value := elicitInterruptValue(nil, []pendingElicitation{{key: key, params: request.Params}})
		answers, err := DecodeElicitationResponses(graph.Interrupt(ctx, value))
		if err != nil {
			return nil, err
		}
		answer, ok := answers[key]
		if !ok {
			return nil, fmt.Errorf("mcp: elicitation resume missing answer for key %q", key)
		}
		if answer.Action == ElicitationCancel {
			return nil, fmt.Errorf("%w (key %q)", ErrElicitationCanceled, key)
		}
		return answer.toMCP(), nil
	}
	return nil, errors.New("mcp: elicitation request arrived outside a bridged tool call; " +
		"elicitation requires the tool to run inside a graph node with a checkpointer " +
		"(legacy server-initiated elicitation is only bridged on in-process connections)")
}

// pendingElicitation is one registered, unanswered request.
type pendingElicitation struct {
	key    string
	params mcp.ElicitationParams
	answer chan *mcp.ElicitationResult
}

// elicitBatch coordinates the requests of one tool invocation.
type elicitBatch struct {
	mu       sync.Mutex
	pending  map[string]*pendingElicitation
	order    []string
	seq      atomic.Int64
	arrived  chan struct{}
	done     chan struct{}
	doneOnce sync.Once
	canceled bool
}

func newElicitBatch() *elicitBatch {
	return &elicitBatch{
		pending: make(map[string]*pendingElicitation),
		arrived: make(chan struct{}, 1),
		done:    make(chan struct{}),
	}
}

// wait registers the request and blocks until its answer is delivered, the
// batch is aborted (interrupt unwind or cancellation), or ctx ends.
func (b *elicitBatch) wait(ctx context.Context, params mcp.ElicitationParams) (*mcp.ElicitationResult, error) {
	key := params.ElicitationID
	if key == "" {
		// Servers may omit the id; derive a stable key from a per-batch
		// counter so the interrupt consumer can address the answer.
		key = fmt.Sprintf("request_%d", b.seq.Add(1))
	}
	p := &pendingElicitation{key: key, params: params, answer: make(chan *mcp.ElicitationResult, 1)}
	b.mu.Lock()
	if _, exists := b.pending[key]; exists {
		b.mu.Unlock()
		return nil, fmt.Errorf("mcp: duplicate elicitation key %q", key)
	}
	b.pending[key] = p
	b.order = append(b.order, key)
	b.mu.Unlock()
	select {
	case b.arrived <- struct{}{}:
	default:
	}

	select {
	case answer := <-p.answer:
		return answer, nil
	case <-b.done:
		return nil, fmt.Errorf("mcp: elicitation abandoned (key %q)", key)
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// snapshot returns the still-unanswered requests in arrival order.
func (b *elicitBatch) snapshot() []pendingElicitation {
	b.mu.Lock()
	defer b.mu.Unlock()
	out := make([]pendingElicitation, 0, len(b.order))
	for _, key := range b.order {
		if p, ok := b.pending[key]; ok {
			out = append(out, *p)
		}
	}
	return out
}

// deliver routes the decoded answers to their waiting requests. It reports
// whether the whole call was canceled by any cancel answer.
func (b *elicitBatch) deliver(answers map[string]ElicitationAnswer) bool {
	for _, answer := range answers {
		if answer.Action == ElicitationCancel {
			b.mu.Lock()
			b.canceled = true
			b.mu.Unlock()
			b.doneOnce.Do(func() { close(b.done) })
			return true
		}
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	for key, answer := range answers {
		if p, ok := b.pending[key]; ok {
			p.answer <- answer.toMCP()
			delete(b.pending, key)
		}
	}
	return false
}

// abort releases every waiting request; used when the invocation unwinds
// (interrupt panic, error return) so no handler goroutine leaks.
func (b *elicitBatch) abort() {
	b.doneOnce.Do(func() { close(b.done) })
}

// isCanceled reports whether a cancel answer abandoned the call.
func (b *elicitBatch) isCanceled() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.canceled
}

// elicitInterruptValue renders the interrupt payload as the JSON-native
// shape upstream documents: interrupt.value["requests"], each request
// carrying its own key. A map round-trips through the JSON-based checkpoint
// savers losslessly (the same convention as the HITL request payload).
func elicitInterruptValue(t *Tool, pending []pendingElicitation) map[string]any {
	requests := make([]any, 0, len(pending))
	for _, p := range pending {
		request := map[string]any{
			"key":     p.key,
			"message": p.params.Message,
		}
		if p.params.Mode != "" {
			request["mode"] = p.params.Mode
		}
		if p.params.URL != "" {
			request["url"] = p.params.URL
		}
		if p.params.RequestedSchema != nil {
			request["requested_schema"] = p.params.RequestedSchema
		}
		requests = append(requests, request)
	}
	value := map[string]any{"requests": requests}
	if t != nil {
		value["tool"] = t.name
		value["server"] = t.serverKey
	}
	return value
}

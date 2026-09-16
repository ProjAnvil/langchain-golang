// This file implements the LangSmith run-tree model (design t18 PR3,
// section 6.1): the Run value posted to {endpoint}/runs/batch and the
// dotted-order computation ported verbatim from
// langchain_core/tracers/core.py (_start_trace, lines 125-130):
//
//	current_dotted_order = run.start_time.strftime("%Y%m%dT%H%M%S%fZ") + str(run.id)
//	if run.parent_run_id (and the parent is known):
//	    run.trace_id, run.dotted_order = parent
//	    run.dotted_order += "." + current_dotted_order
//	else:
//	    run.trace_id = run.id
//	    run.dotted_order = current_dotted_order
//
// A child's dotted order is therefore the parent's dotted order plus "." plus
// the child's own suffix, and every run in a tree carries the root run's ID as
// its trace id.

package tracers

import (
	"fmt"
	"time"
)

// RunType classifies a traced run, mirroring the LangSmith run_type values
// (and the v2 astream_events run types). Chat-model runs keep their own type
// here — the streaming-events schema Python aligns to — rather than the
// legacy "llm" the original tracer posted.
type RunType string

const (
	RunTypeChain     RunType = "chain"
	RunTypeLLM       RunType = "llm"
	RunTypeChatModel RunType = "chat_model"
	RunTypeTool      RunType = "tool"
	RunTypeRetriever RunType = "retriever"
)

// Run is one node of the run tree the LangChainTracer rebuilds from flat
// callback events and posts to LangSmith. The JSON field names follow the
// LangSmith RunTree schema (id/name/run_type/trace_id/dotted_order/...).
//
// Metadata is local-only (json:"-"): it is merged into Extra["metadata"] at
// run start, which is where LangSmith expects it. ChildRuns builds the local
// tree only (json:"-"); the server reconstructs nesting from parent_run_id.
// streams counts observed stream events for the run (local only). EndTime is
// a pointer so encoding/json's omitempty can drop it (a bare time.Time is
// never "empty") until the run finishes.
type Run struct {
	ID          string         `json:"id"`
	Name        string         `json:"name"`
	RunType     RunType        `json:"run_type"`
	ParentRunID string         `json:"parent_run_id,omitempty"`
	TraceID     string         `json:"trace_id"`
	DottedOrder string         `json:"dotted_order"`
	Tags        []string       `json:"tags,omitempty"`
	Metadata    map[string]any `json:"-"`
	Inputs      any            `json:"inputs,omitempty"`
	Outputs     any            `json:"outputs,omitempty"`
	Error       string         `json:"error,omitempty"`
	StartTime   time.Time      `json:"start_time"`
	EndTime     *time.Time     `json:"end_time,omitempty"`
	Extra       map[string]any `json:"extra,omitempty"`
	SessionID   string         `json:"session_id,omitempty"`
	SessionName string         `json:"session_name,omitempty"`

	ChildRuns []*Run `json:"-"`
	streams   int
}

// dottedOrderSuffix renders the "current" component of a dotted order:
// Python's start_time.strftime("%Y%m%dT%H%M%S%fZ") + str(run.id). %f is the
// zero-padded six-digit microsecond fraction, so the layout is
// 20060102T150405 + %06d + "Z" + id, all in UTC.
func dottedOrderSuffix(startTime time.Time, id string) string {
	utc := startTime.UTC()
	return utc.Format("20060102T150405") +
		fmt.Sprintf("%06d", utc.Nanosecond()/1000) +
		"Z" + id
}

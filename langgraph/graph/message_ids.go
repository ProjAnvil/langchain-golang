package graph

// ensureMessageIDs gives every message written to a messages delta channel a
// stable, non-empty ID before the write is applied or persisted — the Go half
// of Python's ensure_message_ids (langgraph/pregel/_messages.py, #9275; only
// the ID half applies: Go's write surface has no strings/dicts to coerce).
//
// A DeltaChannel replays its saved writes on every read and MessagesReducer
// does not fill in missing IDs, so a message saved without an ID comes back
// without one on every read and nothing can target it by ID (a RemoveMessage
// update by a later turn errors on the unknown ID). Stamping happens at the
// producers — the superstep's task outcomes, the graph input, and UpdateState
// values — so the same stamped writes are applied live, streamed, and
// persisted on every path downstream of them: the task-writes commit and
// pause persistence (completedTaskWrites), the delta input writes
// (persistDeltaInputWrites), UpdateState's force-snapshot, and the exit
// accumulator (checkpointSink.putWrites receives already-stamped writes).
//
// Gating mirrors Python's `spec.reducer is _messages_delta_reducer`: only a
// channel built by channels.NewMessagesDeltaChannel — the canonical messages
// delta reducer — is stamped. Other DeltaChannels and non-delta channels keep
// the behavior they had.

import (
	"maps"
	"sync/atomic"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
)

// messageIDSeq supplies the clock sequence for message IDs minted by
// newMessageID (a per-process counter; see checkpoint.NewID).
var messageIDSeq atomic.Int64

// newMessageID mints an ID for a message written without one, mirroring
// Python's str(uuid4()) in ensure_message_ids. It reuses the repo's
// dependency-free ID idiom (checkpoint.NewID: timestamp-clockseq-random):
// message IDs carry no ordering requirement, only uniqueness within and
// across processes.
func newMessageID() string {
	return checkpoint.NewID(int(messageIDSeq.Add(1)))
}

// ensureMessageIDs returns update with every messages-delta-channel entry
// stamped by withMessageIDs. The caller's map and values are never mutated:
// when any entry changes, a new map is returned with only that entry
// replaced; otherwise update is returned as-is.
func (g *CompiledGraph) ensureMessageIDs(update map[string]any) map[string]any {
	if len(update) == 0 {
		return update
	}
	var out map[string]any
	for k, v := range update {
		if !g.isMessagesDeltaChannel(k) {
			continue
		}
		stamped, changed := withMessageIDs(v)
		if !changed {
			continue
		}
		if out == nil {
			out = make(map[string]any, len(update))
			maps.Copy(out, update)
		}
		out[k] = stamped
	}
	if out == nil {
		return update
	}
	return out
}

// isMessagesDeltaChannel reports whether key is held by a DeltaChannel whose
// reducer is the canonical messages delta reducer (see
// channels.NewMessagesDeltaChannel).
func (g *CompiledGraph) isMessagesDeltaChannel(key string) bool {
	proto, ok := g.channelProtos[key]
	if !ok {
		return false
	}
	d, ok := channels.AsDelta(proto)
	return ok && d.IsMessagesDelta()
}

// withMessageIDs stamps a non-empty ID onto every message in a write value
// addressed to a messages delta channel, returning the new value and whether
// anything changed. It never mutates the caller's data: a changed write is a
// copy (new slice, new Overwrite) with stamped Message values.
//
// The shapes accepted are the ones MessagesReducer accepts:
// []messages.Message and []messages.MessageUpdate (a Message element with an
// empty ID gets one; a RemoveMessage is a command whose empty ID means
// "remove all", so it is never stamped), plus a channels.Overwrite wrapping
// either. Anything else — including []messages.RemoveMessage and values that
// are not message writes at all — is returned unchanged, so it fails where it
// always has when the reducer reads it.
func withMessageIDs(v any) (any, bool) {
	if inner, ok := channels.AsOverwrite(v); ok {
		stamped, changed := withMessageIDs(inner)
		if !changed {
			return v, false
		}
		return channels.NewOverwrite(stamped), true
	}
	switch list := v.(type) {
	case []messages.Message:
		changed := false
		out := make([]messages.Message, len(list))
		for i, m := range list {
			if m.ID == "" {
				m.ID = newMessageID()
				changed = true
			}
			out[i] = m
		}
		if !changed {
			return v, false
		}
		return out, true
	case []messages.MessageUpdate:
		changed := false
		out := make([]messages.MessageUpdate, len(list))
		for i, el := range list {
			if m, ok := el.(messages.Message); ok && m.ID == "" {
				m.ID = newMessageID()
				changed = true
				el = m
			}
			out[i] = el
		}
		if !changed {
			return v, false
		}
		return out, true
	}
	return v, false
}

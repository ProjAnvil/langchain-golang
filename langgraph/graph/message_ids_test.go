package graph

// Tests for the messages-delta-channel ID stamping hook (#9275): every
// message written to a DeltaChannel whose reducer is the messages reducer
// gets a stable, non-empty ID before the write is applied or persisted, on
// each persist path — task writes, delta input writes (NullTaskID), UpdateState
// and the exit accumulator. A write that changes is replaced by a copy; the
// caller's messages are never mutated.

import (
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// messagesDeltaGraph builds START -> a -> END over a "messages" key backed by
// a messages delta channel with snapshotFrequency=100, so every value lives
// only as per-task pending writes reconstructed by the ancestor walk — exactly
// the replay surface #9275 hardens. nodeFn supplies node "a"'s result.
func messagesDeltaGraph(t *testing.T, nodeFn func(runtime.Runtime, map[string]any) (any, error), opts ...CompileOption) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddChannel("messages",
		channels.NewMessagesDeltaChannel(func() any { return []messages.Message{} }, 100))
	g.AddNode("a", nodeFn)
	g.AddEdge(types.START, "a")
	g.AddEdge("a", types.END)
	cg, err := g.Compile(opts...)
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

// messageIDs extracts the ordered IDs from a messages channel value, in both
// the typed form (memory-saver replay) and the JSON-decoded form (a delta
// snapshot blob read back through a serde saver).
func messageIDs(v any) []string {
	switch list := v.(type) {
	case []messages.Message:
		ids := make([]string, len(list))
		for i, m := range list {
			ids[i] = m.ID
		}
		return ids
	case []any:
		ids := make([]string, len(list))
		for i, el := range list {
			if m, ok := el.(map[string]any); ok {
				if id, ok := m["id"].(string); ok {
					ids[i] = id
				}
			}
		}
		return ids
	}
	return nil
}

// requireStableIDs reads the thread's state twice and asserts exactly one
// message whose ID is non-empty and identical across both reads (a DeltaChannel
// replays saved writes on every read, so an unstamped write would come back
// empty — or, once stamped at read time, different — on each read).
func requireStableIDs(t *testing.T, cg *CompiledGraph, threadID string) string {
	t.Helper()
	ctx := t.Context()
	snap1, err := cg.GetState(ctx, checkpoint.Config{ThreadID: threadID})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	ids1 := messageIDs(snap1.Values["messages"])
	if len(ids1) != 1 {
		t.Fatalf("messages = %v, want exactly one message", ids1)
	}
	if ids1[0] == "" {
		t.Fatal("message read back with an empty ID; want a stamped non-empty ID")
	}
	snap2, err := cg.GetState(ctx, checkpoint.Config{ThreadID: threadID})
	if err != nil {
		t.Fatalf("GetState() (2nd read) error = %v", err)
	}
	ids2 := messageIDs(snap2.Values["messages"])
	if len(ids2) != 1 || ids2[0] != ids1[0] {
		t.Fatalf("message ID unstable across reads: first %q, second %v", ids1[0], ids2)
	}
	return ids1[0]
}

// requireRemoveByID issues a RemoveMessage update targeting id and asserts the
// message is gone: the stamped ID must be the one the reducer dedups and
// tombstones by, or the targeted update errors.
func requireRemoveByID(t *testing.T, cg *CompiledGraph, threadID, id string) {
	t.Helper()
	ctx := t.Context()
	if _, err := cg.UpdateState(ctx, checkpoint.Config{ThreadID: threadID},
		map[string]any{"messages": []messages.RemoveMessage{{ID: id}}}, "a"); err != nil {
		t.Fatalf("UpdateState(RemoveMessage) error = %v", err)
	}
	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: threadID})
	if err != nil {
		t.Fatalf("GetState() after remove error = %v", err)
	}
	if ids := messageIDs(snap.Values["messages"]); len(ids) != 0 {
		t.Fatalf("messages after RemoveMessage by stamped ID = %v, want none", ids)
	}
}

// TestMessagesDeltaTaskWritesGetStableIDs is the task-writes commit path: a
// node writes a message with an empty ID; the persisted per-task write must
// carry a stable ID, a targeted RemoveMessage by that ID must work, and the
// caller's message must still have its empty ID afterwards.
func TestMessagesDeltaTaskWritesGetStableIDs(t *testing.T) {
	orig := []messages.Message{messages.Human("hi")}
	cg := messagesDeltaGraph(t,
		func(_ runtime.Runtime, _ map[string]any) (any, error) {
			return map[string]any{"messages": orig}, nil
		},
		WithCheckpointer(checkpoint.NewMemorySaver()))

	if _, err := cg.InvokeWithOptions(t.Context(), map[string]any{}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	id := requireStableIDs(t, cg, "t1")
	requireRemoveByID(t, cg, "t1", id)

	if orig[0].ID != "" {
		t.Fatalf("caller's message mutated: ID = %q, want empty", orig[0].ID)
	}
}

// TestMessagesDeltaInputWritesGetStableIDs is the delta input writes path
// (NullTaskID pending writes persisted by persistDeltaInputWrites): graph
// input with an empty-ID message must be saved with a stable ID.
func TestMessagesDeltaInputWritesGetStableIDs(t *testing.T) {
	orig := []messages.Message{messages.Human("input")}
	input := map[string]any{"messages": orig}
	cg := messagesDeltaGraph(t,
		func(_ runtime.Runtime, _ map[string]any) (any, error) { return nil, nil },
		WithCheckpointer(checkpoint.NewMemorySaver()))

	if _, err := cg.InvokeWithOptions(t.Context(), input, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	id := requireStableIDs(t, cg, "t1")
	requireRemoveByID(t, cg, "t1", id)

	if orig[0].ID != "" {
		t.Fatalf("caller's input message mutated: ID = %q, want empty", orig[0].ID)
	}
}

// TestMessagesDeltaUpdateStateStampsIDs is the UpdateState path: values
// applied by UpdateState must be stamped before they are applied and
// force-snapshotted, without mutating the caller's values map.
func TestMessagesDeltaUpdateStateStampsIDs(t *testing.T) {
	cg := messagesDeltaGraph(t,
		func(_ runtime.Runtime, _ map[string]any) (any, error) { return nil, nil },
		WithCheckpointer(checkpoint.NewMemorySaver()))
	if _, err := cg.InvokeWithOptions(t.Context(), map[string]any{}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}

	orig := []messages.Message{messages.Human("upd")}
	values := map[string]any{"messages": orig}
	if _, err := cg.UpdateState(t.Context(), checkpoint.Config{ThreadID: "t1"}, values, "a"); err != nil {
		t.Fatalf("UpdateState() error = %v", err)
	}
	id := requireStableIDs(t, cg, "t1")
	requireRemoveByID(t, cg, "t1", id)

	if orig[0].ID != "" {
		t.Fatalf("caller's update message mutated: ID = %q, want empty", orig[0].ID)
	}
}

// TestMessagesDeltaExitAccumulatorStampsIDs is the exit-accumulator path:
// under exit durability the task's writes reach the saver only through
// flushExit's accumulator, so the stamped IDs must survive that deferral.
func TestMessagesDeltaExitAccumulatorStampsIDs(t *testing.T) {
	orig := []messages.Message{messages.Human("exit")}
	cg := messagesDeltaGraph(t,
		func(_ runtime.Runtime, _ map[string]any) (any, error) {
			return map[string]any{"messages": orig}, nil
		},
		WithCheckpointer(checkpoint.NewMemorySaver()), WithDurability(DurabilityExit))

	if _, err := cg.InvokeWithOptions(t.Context(), map[string]any{}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	id := requireStableIDs(t, cg, "t1")
	requireRemoveByID(t, cg, "t1", id)

	if orig[0].ID != "" {
		t.Fatalf("caller's message mutated: ID = %q, want empty", orig[0].ID)
	}
}

// TestMessagesDeltaGatingLeavesOtherChannelsAlone pins the gate: only a
// messages delta channel is stamped. A plain (BinaryOperator) messages
// channel keeps the empty ID it always kept, and a DeltaChannel with another
// reducer keeps its values verbatim.
func TestMessagesDeltaGatingLeavesOtherChannelsAlone(t *testing.T) {
	g := NewStateGraph()
	g.AddReducer("plain", channels.MessagesReducer)
	g.AddChannel("log", channels.NewDeltaChannel(
		func(existing any, updates []any) (any, error) {
			base, _ := existing.([]string)
			out := append([]string{}, base...)
			for _, u := range updates {
				out = append(out, u.(string))
			}
			return out, nil
		}, func() any { return []string{} }, 100))
	origPlain := []messages.Message{messages.Human("plain")}
	g.AddNode("a", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"plain": origPlain, "log": "tok"}, nil
	})
	g.AddEdge(types.START, "a")
	g.AddEdge("a", types.END)
	cg, err := g.Compile(WithCheckpointer(checkpoint.NewMemorySaver()))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	if _, err := cg.InvokeWithOptions(t.Context(), map[string]any{}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	snap, err := cg.GetState(t.Context(), checkpoint.Config{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	if plain, ok := snap.Values["plain"].([]messages.Message); !ok || len(plain) != 1 || plain[0].ID != "" {
		t.Fatalf("plain messages channel value = %#v, want one message with the empty ID it always had", snap.Values["plain"])
	}
	if logv, ok := snap.Values["log"].([]string); !ok || len(logv) != 1 || logv[0] != "tok" {
		t.Fatalf("non-messages delta channel value = %#v, want [tok] verbatim", snap.Values["log"])
	}
	if origPlain[0].ID != "" {
		t.Fatalf("caller's plain-channel message mutated: ID = %q, want empty", origPlain[0].ID)
	}
}

// TestWithMessageIDsShapes covers the write shapes withMessageIDs accepts:
// message slices (only empty IDs stamped, existing IDs kept), MessageUpdate
// slices (RemoveMessage sentinels untouched), Overwrite-wrapped message lists,
// and values that are not message writes at all (left as-is). The caller's
// slices are never mutated.
func TestWithMessageIDsShapes(t *testing.T) {
	// []messages.Message with a mix of empty and existing IDs.
	orig := []messages.Message{messages.Human("a"), {Role: messages.RoleHuman, Content: "b", ID: "kept"}}
	got, changed := withMessageIDs(orig)
	if !changed {
		t.Fatal("withMessageIDs([]messages.Message) reported no change")
	}
	stamped, ok := got.([]messages.Message)
	if !ok || len(stamped) != 2 {
		t.Fatalf("stamped value = %#v, want []messages.Message of len 2", got)
	}
	if stamped[0].ID == "" || stamped[0].Content != "a" {
		t.Fatalf("first message = %#v, want a stamped copy with content kept", stamped[0])
	}
	if stamped[1].ID != "kept" {
		t.Fatalf("existing ID not preserved: %#v", stamped[1])
	}
	if orig[0].ID != "" {
		t.Fatalf("caller's slice mutated: orig[0].ID = %q, want empty", orig[0].ID)
	}

	// No empty IDs: value reported unchanged.
	same := []messages.Message{{Role: messages.RoleHuman, Content: "x", ID: "id1"}}
	if got, changed = withMessageIDs(same); changed || !reflect.DeepEqual(got, same) {
		t.Fatalf("withMessageIDs(all-IDed) = (%#v, %v), want the same slice unchanged", got, changed)
	}

	// []messages.MessageUpdate: Message elements stamped, RemoveMessage left.
	origUpdates := []messages.MessageUpdate{
		messages.Human("u"),
		messages.RemoveMessage{ID: ""},
		messages.RemoveMessage{ID: "rm"},
	}
	got, changed = withMessageIDs(origUpdates)
	if !changed {
		t.Fatal("withMessageIDs([]messages.MessageUpdate) reported no change")
	}
	updates, ok := got.([]messages.MessageUpdate)
	if !ok || len(updates) != 3 {
		t.Fatalf("stamped value = %#v, want []messages.MessageUpdate of len 3", got)
	}
	if m, ok := updates[0].(messages.Message); !ok || m.ID == "" {
		t.Fatalf("Message element = %#v, want a stamped copy", updates[0])
	}
	if rm, ok := updates[1].(messages.RemoveMessage); !ok || rm.ID != "" {
		t.Fatalf("remove-all sentinel = %#v, want its empty ID untouched", updates[1])
	}
	if rm, ok := updates[2].(messages.RemoveMessage); !ok || rm.ID != "rm" {
		t.Fatalf("RemoveMessage = %#v, want its ID untouched", updates[2])
	}
	if m := origUpdates[0].(messages.Message); m.ID != "" {
		t.Fatalf("caller's update slice mutated: ID = %q, want empty", m.ID)
	}

	// []messages.RemoveMessage: commands, never stamped.
	rems := []messages.RemoveMessage{{ID: ""}}
	if got, changed = withMessageIDs(rems); changed || !reflect.DeepEqual(got, rems) {
		t.Fatalf("withMessageIDs([]messages.RemoveMessage) = (%#v, %v), want unchanged", got, changed)
	}

	// Overwrite wrapping a message list: stamped inside, rewrapped as a copy.
	origOw := []messages.Message{messages.Human("ow")}
	ow := channels.NewOverwrite(origOw)
	got, changed = withMessageIDs(ow)
	if !changed {
		t.Fatal("withMessageIDs(Overwrite) reported no change")
	}
	inner, ok := channels.AsOverwrite(got)
	if !ok {
		t.Fatalf("stamped Overwrite = %#v, want an Overwrite", got)
	}
	if m, ok := inner.([]messages.Message); !ok || len(m) != 1 || m[0].ID == "" {
		t.Fatalf("Overwrite inner = %#v, want a stamped message list", inner)
	}
	if origOw[0].ID != "" {
		t.Fatalf("caller's Overwrite list mutated: ID = %q, want empty", origOw[0].ID)
	}
	if owInner, ok := channels.AsOverwrite(ow); ok {
		if m, ok := owInner.([]messages.Message); ok && m[0].ID != "" {
			t.Fatal("original Overwrite value mutated")
		}
	}

	// A non-message write: left as-is.
	if got, changed = withMessageIDs("tok"); changed || got != any("tok") {
		t.Fatalf("withMessageIDs(string) = (%#v, %v), want unchanged", got, changed)
	}
	if got, changed = withMessageIDs(nil); changed || got != nil {
		t.Fatalf("withMessageIDs(nil) = (%#v, %v), want unchanged", got, changed)
	}
}

// TestEnsureMessageIDsReplacesChangedWritesOnly pins the map-level contract:
// entries for non-gated keys are shared (same underlying values), a changed
// entry is replaced in a NEW map, and an update map with nothing to stamp is
// returned as-is.
func TestEnsureMessageIDsReplacesChangedWritesOnly(t *testing.T) {
	g := NewStateGraph()
	g.AddChannel("messages",
		channels.NewMessagesDeltaChannel(func() any { return []messages.Message{} }, 100))
	g.AddReducer("plain", channels.MessagesReducer)
	g.AddNode("a", func(_ runtime.Runtime, _ map[string]any) (any, error) { return nil, nil })
	g.AddEdge(types.START, "a")
	g.AddEdge("a", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	origMsgs := []messages.Message{messages.Human("m")}
	origPlain := []messages.Message{messages.Human("p")}
	update := map[string]any{"messages": origMsgs, "plain": origPlain}
	out := cg.ensureMessageIDs(update)
	if out == nil {
		t.Fatal("ensureMessageIDs returned nil")
	}
	if _, ok := out["messages"].([]messages.Message); !ok {
		t.Fatalf("out[messages] = %#v, want a stamped message slice", out["messages"])
	}
	if plain, ok := out["plain"].([]messages.Message); !ok || !reflect.DeepEqual(plain, origPlain) || &plain[0] != &origPlain[0] {
		t.Fatalf("out[plain] = %#v, want the caller's value shared for non-gated keys", out["plain"])
	}
	if update["messages"].([]messages.Message)[0].ID != "" {
		t.Fatal("caller's update map mutated")
	}

	same := map[string]any{"plain": origPlain}
	if out = cg.ensureMessageIDs(same); out == nil || !reflect.DeepEqual(out, same) {
		t.Fatalf("ensureMessageIDs(no gated keys) = %#v, want the same map", out)
	}
}

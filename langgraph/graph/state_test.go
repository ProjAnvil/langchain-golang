package graph

import (
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
)

// TestApplyWritesTracksUpdatedChannelsAndOverwrite verifies the Task 2 tracking
// fields populated by applyWrites: updatedChannels (channels bumped this
// superstep) and deltaOverwriteChs (delta channels that received an Overwrite,
// accumulated across supersteps). The BatchReducer lambda is passed directly to
// NewDeltaChannel per amendment C4 (NOT wrapped in BatchFromReducer).
func TestApplyWritesTracksUpdatedChannelsAndOverwrite(t *testing.T) {
	deltaCh := channels.NewDeltaChannel(
		func(existing any, updates []any) (any, error) {
			return append(existing.([]any), updates...), nil
		},
		func() any { return []any{} },
		1000,
	)
	rs := &runState{
		protos:   map[string]channels.Channel{"msg": deltaCh},
		channels: map[string]channels.Channel{"msg": deltaCh},
		versions: map[string]int64{"msg": 0},
		seen:     map[string]map[string]int64{},
	}

	// Normal write: bumps the version, so msg lands in updatedChannels.
	rs.applyWrites([]taskWrites{{node: "n1", update: map[string]any{"msg": []any{"hello"}}}})
	if !rs.updatedChannels["msg"] {
		t.Error("expected msg in updatedChannels after normal write")
	}
	if rs.deltaOverwriteChs["msg"] {
		t.Error("did not expect msg in deltaOverwriteChs after normal write")
	}

	// Overwrite write to the delta channel: recorded in deltaOverwriteChs.
	rs.applyWrites([]taskWrites{{node: "n1", update: map[string]any{"msg": channels.NewOverwrite([]any{"reset"})}}})
	if !rs.deltaOverwriteChs["msg"] {
		t.Error("expected msg in deltaOverwriteChs after Overwrite")
	}
}

// TestApplyWritesUpdatedChannelsResetPerCall verifies updatedChannels is reset
// at the start of every applyWrites call (it does not accumulate), while
// deltaOverwriteChs persists across supersteps until a snapshot clears it.
func TestApplyWritesUpdatedChannelsResetPerCall(t *testing.T) {
	deltaCh := channels.NewDeltaChannel(
		func(existing any, updates []any) (any, error) {
			return append(existing.([]any), updates...), nil
		},
		func() any { return []any{} },
		1000,
	)
	rs := &runState{
		protos:   map[string]channels.Channel{"msg": deltaCh},
		channels: map[string]channels.Channel{"msg": deltaCh},
		versions: map[string]int64{"msg": 0},
		seen:     map[string]map[string]int64{},
	}

	// First call writes both channels' worth of state and sets Overwrite.
	rs.applyWrites([]taskWrites{{node: "n1", update: map[string]any{"msg": channels.NewOverwrite([]any{"a"})}}})
	if !rs.updatedChannels["msg"] {
		t.Fatal("expected msg in updatedChannels after Overwrite write")
	}

	// Second call: an empty write batch leaves updatedChannels empty (no
	// version bumps), but deltaOverwriteChs retains the prior Overwrite.
	rs.applyWrites(nil)
	if len(rs.updatedChannels) != 0 {
		t.Errorf("expected updatedChannels reset on empty call, got %v", rs.updatedChannels)
	}
	if !rs.deltaOverwriteChs["msg"] {
		t.Error("expected deltaOverwriteChs to persist across supersteps")
	}
}

// TestRestoreNilChannelVersions verifies restore guarantees a non-nil
// versions map even when the checkpoint's ChannelVersions decoded as nil
// (JSON savers drop empty maps via omitempty); applyWrites assigns into it.
func TestRestoreNilChannelVersions(t *testing.T) {
	rs := newRunState(nil)
	rs.restore(checkpoint.Checkpoint{
		V:             1,
		ID:            "cp",
		ChannelValues: map[string]any{"x": 1},
		// ChannelVersions deliberately nil.
	})
	if rs.versions == nil {
		t.Fatal("restore() left versions nil, want a non-nil map")
	}
	if got := rs.snapshot()["x"]; got != 1 {
		t.Fatalf("snapshot()[x] = %v, want 1 (channel values restored)", got)
	}
}

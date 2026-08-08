package graph

import (
	"fmt"
	"maps"
	"sort"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
)

// inputNodeName attributes the initial/new-turn input write batch in the
// VersionsSeen bookkeeping, standing in for Python's INPUT pseudo-task.
const inputNodeName = "__input__"

// taskWrites is one task's state writes for a single superstep: node names
// the task for VersionsSeen bookkeeping and update is the partial state
// update the task produced (nil when the task wrote nothing).
type taskWrites struct {
	node   string
	update map[string]any
}

// runState is the executor's channel-backed working state for one run: one
// channels.Channel per state key, the last version written to each channel,
// per-node versions-seen bookkeeping, and the current superstep number
// (-1 before the first superstep commits, mirroring Python's step counter).
type runState struct {
	protos   map[string]channels.Channel
	channels map[string]channels.Channel
	versions map[string]int64
	seen     map[string]map[string]int64
	step     int
}

func newRunState(protos map[string]channels.Channel) *runState {
	return &runState{
		protos:   protos,
		channels: map[string]channels.Channel{},
		versions: map[string]int64{},
		seen:     map[string]map[string]int64{},
		step:     -1,
	}
}

// protoFor returns the registered channel prototype for key, defaulting to a
// LastValue channel for unregistered keys.
func (rs *runState) protoFor(key string) channels.Channel {
	if p, ok := rs.protos[key]; ok {
		return p
	}
	return channels.NewLastValue()
}

// channelFor returns the channel for key, creating a fresh one from the
// key's prototype on first use.
func (rs *runState) channelFor(key string) channels.Channel {
	if ch, ok := rs.channels[key]; ok {
		return ch
	}
	ch := rs.protoFor(key).FromCheckpoint(nil)
	rs.channels[key] = ch
	return ch
}

// restore replaces the working state with a checkpoint's channel values and
// version bookkeeping, rebuilding each channel through its prototype.
func (rs *runState) restore(cp checkpoint.Checkpoint) {
	rs.channels = make(map[string]channels.Channel, len(cp.ChannelValues))
	for key, value := range cp.ChannelValues {
		rs.channels[key] = rs.protoFor(key).FromCheckpoint(value)
	}
	rs.versions = maps.Clone(cp.ChannelVersions)
	rs.seen = cloneSeen(cp.VersionsSeen)
}

// snapshot returns the externally visible graph state: the current value of
// every available channel, minus join barrier channels (control plane; Python
// likewise excludes them from output_keys). Keys whose channel is empty
// (never written, or expired) are absent.
func (rs *runState) snapshot() map[string]any {
	out := make(map[string]any, len(rs.channels))
	for key, ch := range rs.channels {
		if isJoinKey(rs.protos, key) {
			continue
		}
		if v, err := ch.Get(); err == nil {
			out[key] = v
		}
	}
	return out
}

// channelValues returns the per-channel serializable snapshots stored in a
// checkpoint's ChannelValues, omitting empty channels.
func (rs *runState) channelValues() map[string]any {
	out := make(map[string]any, len(rs.channels))
	for key, ch := range rs.channels {
		if v, ok := ch.Checkpoint(); ok {
			out[key] = v
		}
	}
	return out
}

// applyWrites commits one superstep's writes to the channels, implementing
// Python's `apply_writes` algorithm (pregel/_algo.py):
//
//  1. Record each task's pre-write view of the channel versions
//     (versions_seen).
//  2. Compute ONE global next version: max(all channel versions) + 1.
//  3. Batch-update each written channel with its writes in deterministic
//     task order (sorted key order within one task's update); a channel
//     that reports a change is bumped to the shared next version.
//  4. Notify every untouched available channel of the step boundary with an
//     empty Update, which is how expiring channels (Ephemeral,
//     non-accumulating Topic) clear themselves; channels that change are
//     bumped to the shared next version as well.
//
// The bool result reports whether at least one NON-JOIN channel version was
// bumped; the stream emission layer gates `values` chunks on it (mirroring
// Python's `updated_channels ∩ output_keys` gate — join barrier channels are
// not in output_keys, so a superstep that only moved a barrier emits no
// values chunk). Join channels still get their version bump.
func (rs *runState) applyWrites(writes []taskWrites) (bool, error) {
	for _, w := range writes {
		if rs.seen[w.node] == nil {
			rs.seen[w.node] = map[string]int64{}
		}
		for key, v := range rs.versions {
			rs.seen[w.node][key] = v
		}
	}

	var nextVersion int64 = 1
	for _, v := range rs.versions {
		if v >= nextVersion {
			nextVersion = v + 1
		}
	}

	// Group this superstep's writes per channel. Iterating the task slice
	// (never a map) keeps the fold order of multi-write channels
	// deterministic across runs.
	grouped := map[string][]any{}
	var writtenOrder []string
	for _, w := range writes {
		keys := make([]string, 0, len(w.update))
		for k := range w.update {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			if _, ok := grouped[k]; !ok {
				writtenOrder = append(writtenOrder, k)
			}
			grouped[k] = append(grouped[k], w.update[k])
		}
	}

	written := make(map[string]bool, len(grouped))
	anyChanged := false
	for _, key := range writtenOrder {
		changed, err := rs.channelFor(key).Update(grouped[key])
		if err != nil {
			return false, fmt.Errorf("graph: applying writes to key %q: %w", key, err)
		}
		written[key] = true
		if changed {
			rs.versions[key] = nextVersion
			if !isJoinKey(rs.protos, key) {
				anyChanged = true
			}
		}
	}

	untouched := make([]string, 0, len(rs.channels))
	for key, ch := range rs.channels {
		if !written[key] && ch.IsAvailable() {
			untouched = append(untouched, key)
		}
	}
	sort.Strings(untouched)
	for _, key := range untouched {
		changed, err := rs.channels[key].Update(nil)
		if err != nil {
			return false, fmt.Errorf("graph: step-boundary update for key %q: %w", key, err)
		}
		if changed {
			rs.versions[key] = nextVersion
			if !isJoinKey(rs.protos, key) {
				anyChanged = true
			}
		}
	}
	return anyChanged, nil
}

// isJoinKey reports whether key's registered channel prototype is a join
// *channels.Barrier — control-plane state hidden from every user-visible
// state view (snapshots, node inputs, stream chunks). Checkpoint persistence
// (channelValues) deliberately does NOT consult it.
func isJoinKey(protos map[string]channels.Channel, key string) bool {
	_, ok := protos[key].(*channels.Barrier)
	return ok
}

func cloneSeen(seen map[string]map[string]int64) map[string]map[string]int64 {
	out := make(map[string]map[string]int64, len(seen))
	for node, versions := range seen {
		out[node] = maps.Clone(versions)
	}
	return out
}

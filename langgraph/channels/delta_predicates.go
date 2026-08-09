package channels

// DeltaMaxSuperstepsSinceSnapshot bounds replay depth for delta channels that
// stop receiving writes. A channel snapshots when its supersteps-since-snapshot
// counter reaches this value even if the update count hasn't reached
// snapshotFrequency. Mirrors Python's DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT
// (langgraph/_internal/_config.py:33, default 5000). Note: the Python value is
// env-overridable via LANGGRAPH_DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT; the Go port
// uses a compile-time constant for now (override requires a code change).
const DeltaMaxSuperstepsSinceSnapshot = 5000

// DeltaChannelsToSnapshot returns the set of DeltaChannel names that should
// snapshot now. A channel snapshots when EITHER its accumulated update count
// reaches the channel's snapshotFrequency OR its supersteps-since-snapshot
// counter reaches DeltaMaxSuperstepsSinceSnapshot. Unavailable channels
// (IsAvailable()==false) and non-delta channels are skipped. This is a pure
// predicate — it performs no mutation. Mirrors Python's
// delta_channels_to_snapshot (langgraph/pregel/_checkpoint.py:50-71).
//
// counters maps a channel name to a [2]int{updates, supersteps} pair (both
// counted since the channel's last snapshot). A missing entry is treated as
// {0, 0}.
func DeltaChannelsToSnapshot(channels map[string]Channel, counters map[string][2]int) map[string]bool {
	result := make(map[string]bool)
	for name, ch := range channels {
		d, ok := AsDelta(ch)
		if !ok || !d.IsAvailable() {
			continue
		}
		updates, supersteps := 0, 0
		if c, ok := counters[name]; ok {
			updates, supersteps = c[0], c[1]
		}
		if updates >= d.SnapshotFrequency() || supersteps >= DeltaMaxSuperstepsSinceSnapshot {
			result[name] = true
		}
	}
	return result
}

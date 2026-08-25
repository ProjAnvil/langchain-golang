package redis

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
)

// checkpointBlobType is the value of the checkpoint hash's `type` field: the
// storedCheckpoint projection encoded as plain JSON.
const checkpointBlobType = "json"

// checkpoint hash field names.
const (
	fieldParent     = "parent_checkpoint_id"
	fieldType       = "type"
	fieldCheckpoint = "checkpoint"
	fieldMetadata   = "metadata"
)

// storedCheckpoint is the JSON-safe projection of checkpoint.Checkpoint
// persisted in the checkpoint hash's `checkpoint` field, identical in shape
// to the sqlite saver's blob projection:
//   - V, ID: plain fields; TS as RFC3339Nano via encoding/json's time.Time
//     handling.
//   - ChannelValues: each value individually through the Serializer (type tag
//     plus data) so typed values — `[]string`, `[]messages.Message`, `int`,
//     `types.Interrupt`, ... — round-trip exactly instead of degrading to
//     JSON-decoded `[]any` / `float64`.
//   - ChannelVersions / VersionsSeen: plain JSON integer maps (Go-typed
//     fields, so encoding/json restores them exactly).
//   - Next: planned tasks; Arg values through the Serializer like channel
//     values.
type storedCheckpoint struct {
	V               int                         `json:"v"`
	ID              string                      `json:"id"`
	TS              time.Time                   `json:"ts"`
	ChannelValues   map[string]storedValue      `json:"channel_values,omitempty"`
	ChannelVersions map[string]int64            `json:"channel_versions,omitempty"`
	VersionsSeen    map[string]map[string]int64 `json:"versions_seen,omitempty"`
	Next            []storedTask                `json:"next,omitempty"`
}

// storedValue is one serde-typed value embedded in the checkpoint blob.
type storedValue struct {
	Type string          `json:"type"`
	Data json.RawMessage `json:"data"`
}

// storedTask is the projection of checkpoint.PlannedTask.
type storedTask struct {
	ID   string                 `json:"id"`
	Node string                 `json:"node"`
	Arg  map[string]storedValue `json:"arg,omitempty"`
}

// storedMetadata is the plain-JSON projection of checkpoint.Metadata
// persisted in the checkpoint hash's `metadata` field.
type storedMetadata struct {
	Source  string            `json:"source"`
	Step    int               `json:"step"`
	Parents map[string]string `json:"parents,omitempty"`
	RunID   string            `json:"run_id,omitempty"`
}

// storedWrite is the JSON projection of one pending write persisted in a
// checkpoint_write key: the serde-typed value plus the bookkeeping needed to
// rebuild checkpoint.Write without parsing the key.
type storedWrite struct {
	TaskID   string          `json:"task_id"`
	TaskPath string          `json:"task_path"`
	Channel  string          `json:"channel"`
	Idx      int             `json:"idx"`
	Type     string          `json:"type"`
	Data     json.RawMessage `json:"data"`
}

// encodeValue runs one channel value / write value through the serde.
func (s *Saver) encodeValue(v any) (storedValue, error) {
	typ, data, err := s.serde.DumpsTyped(v)
	if err != nil {
		return storedValue{}, err
	}
	return storedValue{Type: typ, Data: data}, nil
}

// decodeValue restores a value encoded by encodeValue.
func (s *Saver) decodeValue(sv storedValue) (any, error) {
	return s.serde.LoadsTyped(sv.Type, sv.Data)
}

// encodeCheckpoint projects cp into its JSON-safe stored form and marshals it.
func (s *Saver) encodeCheckpoint(cp checkpoint.Checkpoint) ([]byte, error) {
	proj := storedCheckpoint{
		V:               cp.V,
		ID:              cp.ID,
		TS:              cp.TS,
		ChannelVersions: cp.ChannelVersions,
		VersionsSeen:    cp.VersionsSeen,
	}
	if cp.ChannelValues != nil {
		proj.ChannelValues = make(map[string]storedValue, len(cp.ChannelValues))
		for channel, v := range cp.ChannelValues {
			sv, err := s.encodeValue(v)
			if err != nil {
				return nil, fmt.Errorf("channel %q: %w", channel, err)
			}
			proj.ChannelValues[channel] = sv
		}
	}
	if cp.Next != nil {
		proj.Next = make([]storedTask, len(cp.Next))
		for i, task := range cp.Next {
			st := storedTask{ID: task.ID, Node: task.Node}
			if task.Arg != nil {
				st.Arg = make(map[string]storedValue, len(task.Arg))
				for k, v := range task.Arg {
					sv, err := s.encodeValue(v)
					if err != nil {
						return nil, fmt.Errorf("next task %q arg %q: %w", task.ID, k, err)
					}
					st.Arg[k] = sv
				}
			}
			proj.Next[i] = st
		}
	}
	return json.Marshal(proj)
}

// decodeCheckpoint restores a Checkpoint from its stored blob. typ must be
// checkpointBlobType — the projection is always plain JSON.
func (s *Saver) decodeCheckpoint(typ string, blob []byte) (checkpoint.Checkpoint, error) {
	if typ != checkpointBlobType {
		return checkpoint.Checkpoint{}, fmt.Errorf("unknown checkpoint blob type %q", typ)
	}
	var proj storedCheckpoint
	if err := json.Unmarshal(blob, &proj); err != nil {
		return checkpoint.Checkpoint{}, err
	}
	cp := checkpoint.Checkpoint{
		V:               proj.V,
		ID:              proj.ID,
		TS:              proj.TS,
		ChannelVersions: proj.ChannelVersions,
		VersionsSeen:    proj.VersionsSeen,
	}
	if proj.ChannelValues != nil {
		cp.ChannelValues = make(map[string]any, len(proj.ChannelValues))
		for channel, sv := range proj.ChannelValues {
			v, err := s.decodeValue(sv)
			if err != nil {
				return checkpoint.Checkpoint{}, fmt.Errorf("channel %q: %w", channel, err)
			}
			cp.ChannelValues[channel] = v
		}
	}
	if proj.Next != nil {
		cp.Next = make([]checkpoint.PlannedTask, len(proj.Next))
		for i, st := range proj.Next {
			task := checkpoint.PlannedTask{ID: st.ID, Node: st.Node}
			if st.Arg != nil {
				task.Arg = make(map[string]any, len(st.Arg))
				for k, sv := range st.Arg {
					v, err := s.decodeValue(sv)
					if err != nil {
						return checkpoint.Checkpoint{}, fmt.Errorf("next task %q arg %q: %w", st.ID, k, err)
					}
					task.Arg[k] = v
				}
			}
			cp.Next[i] = task
		}
	}
	return cp, nil
}

func encodeMetadata(md checkpoint.Metadata) ([]byte, error) {
	return json.Marshal(storedMetadata{Source: md.Source, Step: md.Step, Parents: md.Parents, RunID: md.RunID})
}

func decodeMetadata(blob []byte) (checkpoint.Metadata, error) {
	if len(blob) == 0 {
		return checkpoint.Metadata{}, nil
	}
	var stored storedMetadata
	if err := json.Unmarshal(blob, &stored); err != nil {
		return checkpoint.Metadata{}, err
	}
	return checkpoint.Metadata{Source: stored.Source, Step: stored.Step, Parents: stored.Parents, RunID: stored.RunID}, nil
}

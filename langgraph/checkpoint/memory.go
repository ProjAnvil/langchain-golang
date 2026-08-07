package checkpoint

import (
	"context"
	"fmt"
	"maps"
	"slices"
	"sort"
	"sync"
)

// stored is one persisted checkpoint plus its associated data.
type stored struct {
	cp     Checkpoint
	md     Metadata
	parent *Config
	writes []Write
}

// MemorySaver is an in-memory Saver, the Go equivalent of Python's
// `InMemorySaver`: full versioned history per thread, lost when the process
// exits. The zero value is ready to use. It is safe for concurrent use.
type MemorySaver struct {
	mu sync.Mutex
	// threads maps thread ID -> checkpoint namespace -> checkpoint ID -> stored.
	threads map[string]map[string]map[string]stored
}

// NewMemorySaver constructs an empty MemorySaver.
func NewMemorySaver() *MemorySaver {
	return &MemorySaver{threads: map[string]map[string]map[string]stored{}}
}

// GetTuple implements Saver.
func (s *MemorySaver) GetTuple(ctx context.Context, cfg Config) (*Tuple, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, st, ok := s.lookup(cfg)
	if !ok {
		return nil, nil
	}
	tup := s.tuple(cfg, st)
	return &tup, nil
}

// List implements Saver: checkpoints of cfg.ThreadID/cfg.CheckpointNS,
// newest (highest ID) first, filtered by opts.Before and capped by opts.Limit.
func (s *MemorySaver) List(ctx context.Context, cfg Config, opts ListOptions) ([]Tuple, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	ns := s.threads[cfg.ThreadID][cfg.CheckpointNS]
	ids := make([]string, 0, len(ns))
	for id := range ns {
		if opts.Before != nil && opts.Before.CheckpointID != "" && id >= opts.Before.CheckpointID {
			continue
		}
		ids = append(ids, id)
	}
	sort.Sort(sort.Reverse(sort.StringSlice(ids)))
	if opts.Limit > 0 && len(ids) > opts.Limit {
		ids = ids[:opts.Limit]
	}
	out := make([]Tuple, 0, len(ids))
	for _, id := range ids {
		out = append(out, s.tuple(cfg, ns[id]))
	}
	return out, nil
}

// Put implements Saver. The parent link is taken from cfg (D3): when
// cfg.CheckpointID is non-empty, a copy of cfg is recorded as the new
// checkpoint's ParentConfig.
func (s *MemorySaver) Put(ctx context.Context, cfg Config, cp Checkpoint, md Metadata, newVersions map[string]int64) (Config, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.threads == nil {
		s.threads = map[string]map[string]map[string]stored{}
	}
	ns := s.threads[cfg.ThreadID]
	if ns == nil {
		ns = map[string]map[string]stored{}
		s.threads[cfg.ThreadID] = ns
	}
	byID := ns[cfg.CheckpointNS]
	if byID == nil {
		byID = map[string]stored{}
		ns[cfg.CheckpointNS] = byID
	}

	storedCp := copyCheckpoint(cp)
	if len(newVersions) > 0 {
		if storedCp.ChannelVersions == nil {
			storedCp.ChannelVersions = map[string]int64{}
		}
		maps.Copy(storedCp.ChannelVersions, newVersions)
	}
	var parent *Config
	if cfg.CheckpointID != "" {
		c := cfg
		parent = &c
	}
	byID[cp.ID] = stored{
		cp:     storedCp,
		md:     copyMetadata(md),
		parent: parent,
	}
	return Config{ThreadID: cfg.ThreadID, CheckpointNS: cfg.CheckpointNS, CheckpointID: cp.ID}, nil
}

// PutWrites implements Saver. Each write is stamped with taskID and appended
// to the checkpoint's pending writes in call order.
func (s *MemorySaver) PutWrites(ctx context.Context, cfg Config, writes []Write, taskID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	id, st, ok := s.lookup(cfg)
	if !ok {
		return fmt.Errorf("checkpoint: PutWrites: no checkpoint %q for thread %q (ns %q)",
			cfg.CheckpointID, cfg.ThreadID, cfg.CheckpointNS)
	}
	stamped := make([]Write, len(writes))
	for i, w := range writes {
		w.TaskID = taskID
		stamped[i] = w
	}
	st.writes = append(st.writes, stamped...)
	s.threads[cfg.ThreadID][cfg.CheckpointNS][id] = st
	return nil
}

// DeleteThread implements Saver.
func (s *MemorySaver) DeleteThread(ctx context.Context, threadID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.threads, threadID)
	return nil
}

// lookup resolves cfg to a stored checkpoint: cfg.CheckpointID selects an
// exact checkpoint, empty selects the latest (highest ID).
func (s *MemorySaver) lookup(cfg Config) (string, stored, bool) {
	byID := s.threads[cfg.ThreadID][cfg.CheckpointNS]
	if cfg.CheckpointID != "" {
		st, ok := byID[cfg.CheckpointID]
		return cfg.CheckpointID, st, ok
	}
	latest := ""
	for id := range byID {
		if id > latest {
			latest = id
		}
	}
	if latest == "" {
		return "", stored{}, false
	}
	return latest, byID[latest], true
}

// tuple builds a Tuple for st, deep-copying all maps and slices (values are
// shared) so callers cannot alias store state.
func (s *MemorySaver) tuple(cfg Config, st stored) Tuple {
	tupCfg := Config{
		ThreadID:     cfg.ThreadID,
		CheckpointNS: cfg.CheckpointNS,
		CheckpointID: st.cp.ID,
	}
	var parent *Config
	if st.parent != nil {
		c := *st.parent
		parent = &c
	}
	return Tuple{
		Config:        tupCfg,
		Checkpoint:    copyCheckpoint(st.cp),
		Metadata:      copyMetadata(st.md),
		ParentConfig:  parent,
		PendingWrites: slices.Clone(st.writes),
	}
}

func copyCheckpoint(cp Checkpoint) Checkpoint {
	out := cp
	out.ChannelValues = maps.Clone(cp.ChannelValues)
	out.ChannelVersions = maps.Clone(cp.ChannelVersions)
	if cp.VersionsSeen != nil {
		seen := make(map[string]map[string]int64, len(cp.VersionsSeen))
		for node, versions := range cp.VersionsSeen {
			seen[node] = maps.Clone(versions)
		}
		out.VersionsSeen = seen
	}
	out.Next = slices.Clone(cp.Next)
	return out
}

func copyMetadata(md Metadata) Metadata {
	out := md
	out.Parents = maps.Clone(md.Parents)
	return out
}

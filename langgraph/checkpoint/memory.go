package checkpoint

import (
	"context"
	"fmt"
	"maps"
	"slices"
	"sort"
	"sync"
	"time"
)

// stored is one persisted checkpoint plus its associated data.
type stored struct {
	cp     Checkpoint
	md     Metadata
	parent *Config
	writes []Write
	// writeSlots maps each occupied write slot to its position in writes,
	// mirroring the inner key of Python InMemorySaver's writes map.
	writeSlots map[writeKey]int
}

// writeKey identifies one write slot of a checkpoint: (task ID, write idx).
type writeKey struct {
	taskID string
	idx    int
}

// reservedWriteIdx mirrors Python's WRITES_IDX_MAP
// `{ERROR: -1, SCHEDULED: -2, INTERRUPT: -3, RESUME: -4}`
// (`libs/checkpoint/langgraph/checkpoint/base/__init__.py:795`): writes to
// reserved channels occupy a fixed negative slot and are overwritten on
// rewrite, while regular channels occupy their positional idx and keep the
// first write. (Go has no SCHEDULED/RESUME writes, and __tasks__ is
// positional here exactly as in Python's map — unlike the Go sqlite saver,
// which assigns it the reserved slot -2.)
var reservedWriteIdx = map[string]int{
	ReservedError:     -1,
	ReservedInterrupt: -3,
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
// newest (highest ID) first, filtered by opts.Before and opts.Filter (filter
// before limit, mirroring Python's WHERE→LIMIT order) and capped by
// opts.Limit.
func (s *MemorySaver) List(ctx context.Context, cfg Config, opts ListOptions) ([]Tuple, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	ns := s.threads[cfg.ThreadID][cfg.CheckpointNS]
	ids := make([]string, 0, len(ns))
	for id, st := range ns {
		if opts.Before != nil && opts.Before.CheckpointID != "" && id >= opts.Before.CheckpointID {
			continue
		}
		if !MetadataMatchesFilter(st.md, opts.Filter) {
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

// PutWrites implements Saver. Each write is stamped with taskID and
// taskPath and recorded under its (taskID, idx) slot, mirroring Python
// InMemorySaver's put_writes: a regular channel takes its positional idx and
// re-writing an occupied slot is ignored (first-write-wins); a reserved
// channel takes its fixed negative slot (reservedWriteIdx) and re-writing it
// replaces the stored value in place. PendingWrites reads back in insertion
// order.
func (s *MemorySaver) PutWrites(ctx context.Context, cfg Config, writes []Write, taskID, taskPath string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	id, st, ok := s.lookup(cfg)
	if !ok {
		return fmt.Errorf("checkpoint: PutWrites: no checkpoint %q for thread %q (ns %q)",
			cfg.CheckpointID, cfg.ThreadID, cfg.CheckpointNS)
	}
	if st.writeSlots == nil {
		st.writeSlots = map[writeKey]int{}
	}
	for i, w := range writes {
		idx := i
		if reserved, ok := reservedWriteIdx[w.Channel]; ok {
			idx = reserved
		}
		key := writeKey{taskID: taskID, idx: idx}
		w.TaskID = taskID
		w.TaskPath = taskPath
		if pos, occupied := st.writeSlots[key]; occupied {
			if idx >= 0 {
				continue // first write wins
			}
			st.writes[pos] = w // reserved slot: replace in place
			continue
		}
		st.writeSlots[key] = len(st.writes)
		st.writes = append(st.writes, w)
	}
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

// cacheEntry is one stored cache value with its absolute expiry (the zero
// time means the entry never expires).
type cacheEntry struct {
	writes  []Write
	expires time.Time
}

// InMemoryCache is an in-memory Cache, the Go equivalent of Python's
// `langgraph.cache.memory.InMemoryCache`: entries live until their TTL's
// absolute expiry and are lost when the process exits. The zero value is
// ready to use. It is safe for concurrent use.
type InMemoryCache struct {
	mu sync.Mutex
	// entries maps namespace -> key -> entry.
	entries map[string]map[string]cacheEntry
}

// NewInMemoryCache constructs an empty InMemoryCache.
func NewInMemoryCache() *InMemoryCache {
	return &InMemoryCache{entries: map[string]map[string]cacheEntry{}}
}

// Get implements Cache. An entry at or past its absolute expiry is evicted
// and reported as a miss (Python parity: TTLs are checked on read).
func (c *InMemoryCache) Get(ctx context.Context, ns, key string) ([]Write, bool, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	entry, ok := c.entries[ns][key]
	if !ok {
		return nil, false, nil
	}
	if !entry.expires.IsZero() && !time.Now().Before(entry.expires) {
		delete(c.entries[ns], key)
		return nil, false, nil
	}
	return slices.Clone(entry.writes), true, nil
}

// Set implements Cache. The writes slice is cloned so later caller mutation
// cannot corrupt the stored entry.
func (c *InMemoryCache) Set(ctx context.Context, ns, key string, writes []Write, ttl time.Duration) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.entries == nil {
		c.entries = map[string]map[string]cacheEntry{}
	}
	byKey := c.entries[ns]
	if byKey == nil {
		byKey = map[string]cacheEntry{}
		c.entries[ns] = byKey
	}
	entry := cacheEntry{writes: slices.Clone(writes)}
	if ttl > 0 {
		entry.expires = time.Now().Add(ttl)
	}
	byKey[key] = entry
	return nil
}

// Clear implements Cache.
func (c *InMemoryCache) Clear(ctx context.Context, ns string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.entries, ns)
	return nil
}

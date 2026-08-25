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
// first write. Go has no SCHEDULED counterpart; `__resume__` and `__tasks__`
// stay positional here (Python's map slots RESUME at -4 and has no TASKS
// entry) — harmless because the executor persists a task's consumed resume
// prefix as ONE write carrying the whole ordered list, so no same-batch slot
// collapse is possible. The sqlite and postgres savers keep `__resume__` at
// the reserved -4 (lossless for the same reason) and likewise route
// `__tasks__` through the positional idx.
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

// DeleteForRuns implements Saver: drops every stored checkpoint whose
// metadata RunID is listed, across all threads and namespaces. Writes ride
// inside the stored record, so they are removed together. Checkpoints with
// an empty RunID never match. The Saver-interface DeltaChannel warning
// (base/__init__.py:340-346) applies: deletion can sever a live thread's
// delta-channel ancestor history — documented only, mirroring Python.
func (s *MemorySaver) DeleteForRuns(ctx context.Context, runIDs []string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(runIDs) == 0 {
		return nil
	}
	drop := make(map[string]bool, len(runIDs))
	for _, id := range runIDs {
		drop[id] = true
	}
	for threadID, nss := range s.threads {
		for ns, byID := range nss {
			for cpID, st := range byID {
				if st.md.RunID != "" && drop[st.md.RunID] {
					delete(byID, cpID)
				}
			}
			if len(byID) == 0 {
				delete(nss, ns)
			}
		}
		if len(nss) == 0 {
			delete(s.threads, threadID)
		}
	}
	return nil
}

// CopyThread implements Saver: every namespace's records are deep-copied
// onto the destination thread (checkpoint IDs and parent links preserved,
// mirroring Python's requirement that the copy carry the complete parent
// chain, base/__init__.py:361-371). A nonexistent source is a no-op.
func (s *MemorySaver) CopyThread(ctx context.Context, srcThreadID, dstThreadID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	src := s.threads[srcThreadID]
	if len(src) == 0 {
		return nil
	}
	dst := s.threads[dstThreadID]
	if dst == nil {
		dst = map[string]map[string]stored{}
		s.threads[dstThreadID] = dst
	}
	for ns, byID := range src {
		dstNS := dst[ns]
		if dstNS == nil {
			dstNS = map[string]stored{}
			dst[ns] = dstNS
		}
		for id, st := range byID {
			copied := stored{
				cp:     copyCheckpoint(st.cp),
				md:     copyMetadata(st.md),
				writes: slices.Clone(st.writes),
			}
			if st.parent != nil {
				p := *st.parent
				copied.parent = &p
			}
			if st.writeSlots != nil {
				copied.writeSlots = maps.Clone(st.writeSlots)
			}
			dstNS[id] = copied
		}
	}
	return nil
}

// Prune implements Saver: PruneKeepLatest keeps the highest-ID (latest)
// checkpoint per namespace; PruneDeleteAll removes the thread. Unknown
// threads are skipped; an unknown strategy is an error. The Saver-interface
// DeltaChannel warning (base/__init__.py:387-413) applies: naive keep_latest
// can sever delta-channel ancestor history — documented only, mirroring
// Python.
func (s *MemorySaver) Prune(ctx context.Context, threadIDs []string, strategy PruneStrategy) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	switch strategy {
	case PruneKeepLatest, PruneDeleteAll:
	default:
		return fmt.Errorf("checkpoint: unknown prune strategy %q", strategy)
	}
	for _, threadID := range threadIDs {
		nss := s.threads[threadID]
		if len(nss) == 0 {
			continue
		}
		if strategy == PruneDeleteAll {
			delete(s.threads, threadID)
			continue
		}
		for _, byID := range nss {
			latest := ""
			for id := range byID {
				if id > latest {
					latest = id
				}
			}
			for id := range byID {
				if id != latest {
					delete(byID, id)
				}
			}
		}
	}
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

package runnables

import (
	"maps"
	"slices"
)

// MergeConfig merges multiple Config values into one using the same semantics
// as Python's merge_configs:
//
//   - Tags: union, sorted, deduplicated.
//   - Metadata: last-writer-wins per key; the special "lc_versions" key is
//     accumulated as a nested merge across all configs so package-version maps
//     from core, langchain, and partner packages coexist in the final config.
//   - Configurable: last-writer-wins per key.
//   - Name, RunID, ParentID: last non-empty value wins.
//   - Callbacks: last non-nil value wins (Go callback managers are not merged).
func MergeConfig(configs ...Config) Config {
	out := Config{
		Metadata:     map[string]any{},
		Configurable: map[string]any{},
	}
	for _, cfg := range configs {
		// Tags: deduplicated union.
		if len(cfg.Tags) > 0 {
			tagSet := make(map[string]struct{}, len(out.Tags)+len(cfg.Tags))
			for _, t := range out.Tags {
				tagSet[t] = struct{}{}
			}
			for _, t := range cfg.Tags {
				tagSet[t] = struct{}{}
			}
			out.Tags = slices.Sorted(maps.Keys(tagSet))
		}

		// Metadata: last-writer-wins; lc_versions accumulates.
		if cfg.Metadata != nil {
			out.Metadata = mergeMetadataMaps(out.Metadata, cfg.Metadata)
		}

		// Configurable: last-writer-wins per key.
		maps.Copy(out.Configurable, cfg.Configurable)

		// Scalar fields: last non-empty wins.
		if cfg.Name != "" {
			out.Name = cfg.Name
		}
		if cfg.RunID != "" {
			out.RunID = cfg.RunID
		}
		if cfg.ParentID != "" {
			out.ParentID = cfg.ParentID
		}
		// Callbacks: non-empty manager wins (Manager is a struct; empty == no handlers).
		if !cfg.Callbacks.Empty() {
			out.Callbacks = cfg.Callbacks
		}
	}
	return out
}

// mergeMetadataMaps merges two metadata maps using last-writer-wins semantics.
// The "lc_versions" key is special: its value is expected to be a
// map[string]any and is accumulated (union) across merges so that version maps
// from multiple packages coexist. This mirrors Python's _merge_metadata_dicts.
func mergeMetadataMaps(base, incoming map[string]any) map[string]any {
	merged := make(map[string]any, len(base)+len(incoming))
	maps.Copy(merged, base)
	maps.Copy(merged, incoming)

	// Accumulate lc_versions if both sides are maps.
	baseVersions, baseOK := toStringAnyMap(base["lc_versions"])
	incomingVersions, incomingOK := toStringAnyMap(incoming["lc_versions"])
	switch {
	case baseOK && incomingOK:
		acc := make(map[string]any, len(baseVersions)+len(incomingVersions))
		maps.Copy(acc, baseVersions)
		maps.Copy(acc, incomingVersions)
		merged["lc_versions"] = acc
	case incomingOK:
		merged["lc_versions"] = cloneConfigMap(incomingVersions)
	case baseOK:
		if _, alreadySet := incoming["lc_versions"]; !alreadySet {
			merged["lc_versions"] = cloneConfigMap(baseVersions)
		}
	}
	return merged
}

func toStringAnyMap(v any) (map[string]any, bool) {
	if v == nil {
		return nil, false
	}
	m, ok := v.(map[string]any)
	return m, ok
}

package runnables

import "sort"

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
			out.Tags = make([]string, 0, len(tagSet))
			for t := range tagSet {
				out.Tags = append(out.Tags, t)
			}
			sort.Strings(out.Tags)
		}

		// Metadata: last-writer-wins; lc_versions accumulates.
		if cfg.Metadata != nil {
			out.Metadata = mergeMetadataMaps(out.Metadata, cfg.Metadata)
		}

		// Configurable: last-writer-wins per key.
		for key, value := range cfg.Configurable {
			out.Configurable[key] = value
		}

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
	for k, v := range base {
		merged[k] = v
	}
	for k, v := range incoming {
		merged[k] = v
	}

	// Accumulate lc_versions if both sides are maps.
	baseVersions, baseOK := toStringAnyMap(base["lc_versions"])
	incomingVersions, incomingOK := toStringAnyMap(incoming["lc_versions"])
	switch {
	case baseOK && incomingOK:
		acc := make(map[string]any, len(baseVersions)+len(incomingVersions))
		for k, v := range baseVersions {
			acc[k] = v
		}
		for k, v := range incomingVersions {
			acc[k] = v
		}
		merged["lc_versions"] = acc
	case incomingOK:
		acc := make(map[string]any, len(incomingVersions))
		for k, v := range incomingVersions {
			acc[k] = v
		}
		merged["lc_versions"] = acc
	case baseOK:
		if _, alreadySet := incoming["lc_versions"]; !alreadySet {
			acc := make(map[string]any, len(baseVersions))
			for k, v := range baseVersions {
				acc[k] = v
			}
			merged["lc_versions"] = acc
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

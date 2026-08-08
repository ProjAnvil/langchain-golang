package checkpoint

import "encoding/json"

// MetadataMatchesFilter reports whether md contains filter, the in-process
// equivalent of Postgres's `metadata @> filter` JSONB containment used by
// persistent savers. The metadata document is md's JSON projection —
// {"source": ..., "step": ..., "parents": ...} — so filter keys are limited
// to source/step/parents (Metadata is a closed struct, unlike Python's
// free-form CheckpointMetadata). Both sides are normalized through JSON
// before comparison so `step` filters match whether written as int or
// float64 (JSONB numeric equality). Containment is recursive, mirroring @>:
// an object filter matches when every key is present and contained, an array
// filter when every element is contained in some element of the target, and
// scalars when equal.
func MetadataMatchesFilter(md Metadata, filter map[string]any) bool {
	if len(filter) == 0 {
		return true
	}
	doc := map[string]any{"source": md.Source, "step": md.Step}
	if md.Parents != nil {
		doc["parents"] = md.Parents
	}
	return jsonContains(normalizeJSON(doc), normalizeJSON(filter))
}

// jsonContains reports whether doc contains filter with Postgres @>
// semantics: objects contain recursively (every filter key must be present
// and contained), arrays contain element-wise, scalars compare by equality
// (after JSON normalization, so int and float64 encodings of the same number
// are equal).
func jsonContains(doc, filter any) bool {
	switch fv := filter.(type) {
	case map[string]any:
		dv, ok := doc.(map[string]any)
		if !ok {
			return false
		}
		for k, v := range fv {
			d, ok := dv[k]
			if !ok || !jsonContains(d, v) {
				return false
			}
		}
		return true
	case []any:
		dv, ok := doc.([]any)
		if !ok {
			return false
		}
		for _, item := range fv {
			found := false
			for _, d := range dv {
				if jsonContains(d, item) {
					found = true
					break
				}
			}
			if !found {
				return false
			}
		}
		return true
	default:
		// Scalars: both sides are JSON-normalized (float64/string/bool/nil),
		// so == compares with JSONB numeric equality semantics.
		return doc == filter
	}
}

// normalizeJSON round-trips v through encoding/json so numbers become
// float64 and maps/slices become map[string]any/[]any.
func normalizeJSON(v any) any {
	raw, err := json.Marshal(v)
	if err != nil {
		return v
	}
	var out any
	if err := json.Unmarshal(raw, &out); err != nil {
		return v
	}
	return out
}

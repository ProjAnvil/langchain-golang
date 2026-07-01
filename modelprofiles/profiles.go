package modelprofiles

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

const maxRows = 25

// Profile is model capability metadata.
type Profile map[string]any

// Registry maps model IDs to profiles.
type Registry map[string]Profile

// FieldChange records one profile field transition.
type FieldChange struct {
	Old any
	New any
}

// Diff is the structured difference between two model profile registries.
type Diff struct {
	Added         []string
	Removed       []string
	Changed       map[string]map[string]FieldChange
	AddedProfiles Registry
}

func (d Diff) IsEmpty() bool {
	return len(d.Added) == 0 && len(d.Removed) == 0 && len(d.Changed) == 0
}

var fieldLabels = map[string]string{
	"name":                "display name",
	"status":              "status",
	"release_date":        "release date",
	"last_updated":        "last updated",
	"open_weights":        "open weights",
	"max_input_tokens":    "max input tokens",
	"max_output_tokens":   "max output tokens",
	"text_inputs":         "text input",
	"image_inputs":        "image input",
	"audio_inputs":        "audio input",
	"pdf_inputs":          "PDF input",
	"video_inputs":        "video input",
	"text_outputs":        "text output",
	"image_outputs":       "image output",
	"audio_outputs":       "audio output",
	"video_outputs":       "video output",
	"reasoning_output":    "reasoning",
	"tool_calling":        "tool calling",
	"tool_choice":         "tool choice",
	"tool_call_streaming": "tool call streaming",
	"structured_output":   "structured output",
	"attachment":          "attachments",
	"temperature":         "temperature control",
	"image_url_inputs":    "image URL input",
	"image_tool_message":  "image tool messages",
	"pdf_tool_message":    "PDF tool messages",
}

// DiffProfiles computes added, removed, and changed profiles.
func DiffProfiles(old Registry, new Registry) Diff {
	added := sortedDifference(keys(new), keys(old))
	removed := sortedDifference(keys(old), keys(new))
	changed := map[string]map[string]FieldChange{}

	for _, modelID := range sortedIntersection(keys(old), keys(new)) {
		oldProfile := old[modelID]
		newProfile := new[modelID]
		fields := map[string]FieldChange{}
		for _, key := range sortedUnion(profileKeys(oldProfile), profileKeys(newProfile)) {
			oldVal := oldProfile[key]
			newVal := newProfile[key]
			if !jsonEqual(oldVal, newVal) {
				fields[key] = FieldChange{Old: oldVal, New: newVal}
			}
		}
		if len(fields) > 0 {
			changed[modelID] = fields
		}
	}

	addedProfiles := Registry{}
	for _, modelID := range added {
		addedProfiles[modelID] = cloneProfile(new[modelID])
	}
	return Diff{
		Added:         added,
		Removed:       removed,
		Changed:       changed,
		AddedProfiles: addedProfiles,
	}
}

// FormatValue renders a profile value for Markdown display.
func FormatValue(fieldName string, value any) string {
	if value == nil {
		return "unset"
	}
	if b, ok := value.(bool); ok {
		if b {
			return "yes"
		}
		return "no"
	}
	if isTokenField(fieldName) {
		if n, ok := intValue(value); ok {
			return formatInt(n)
		}
	}
	if s, ok := value.(string); ok {
		return "`" + s + "`"
	}
	return fmt.Sprint(value)
}

// DescribeNewModel returns a short capability descriptor.
func DescribeNewModel(profile Profile) string {
	parts := []string{}
	if context, ok := intValue(profile["max_input_tokens"]); ok && context != 0 {
		parts = append(parts, formatInt(context)+" ctx")
	}
	if output, ok := intValue(profile["max_output_tokens"]); ok && output != 0 {
		parts = append(parts, formatInt(output)+" out")
	}
	modalities := []string{}
	for _, item := range []struct {
		Key  string
		Name string
	}{
		{"image_inputs", "image"},
		{"audio_inputs", "audio"},
		{"video_inputs", "video"},
		{"pdf_inputs", "pdf"},
	} {
		if truthy(profile[item.Key]) {
			modalities = append(modalities, item.Name)
		}
	}
	if len(modalities) > 0 {
		parts = append(parts, "text+"+strings.Join(modalities, "+")+" in")
	}
	if truthy(profile["reasoning_output"]) {
		parts = append(parts, "reasoning")
	}
	if truthy(profile["tool_calling"]) {
		parts = append(parts, "tools")
	}
	return strings.Join(parts, ", ")
}

// RenderProviderSection renders a provider's Markdown diff section.
func RenderProviderSection(provider string, diff Diff) string {
	if diff.IsEmpty() {
		return ""
	}
	lines := []string{"### " + provider}
	if len(diff.Added) > 0 {
		lines = append(lines, fmt.Sprintf("\n**+ %d added**", len(diff.Added)))
		rows := []string{}
		for _, modelID := range diff.Added {
			descriptor := DescribeNewModel(diff.AddedProfiles[modelID])
			suffix := ""
			if descriptor != "" {
				suffix = " - " + descriptor
			}
			rows = append(rows, fmt.Sprintf("- `%s`%s", modelID, suffix))
		}
		lines = append(lines, Truncate(rows)...)
	}
	if len(diff.Removed) > 0 {
		lines = append(lines, fmt.Sprintf("\n**- %d removed**", len(diff.Removed)))
		rows := make([]string, len(diff.Removed))
		for i, modelID := range diff.Removed {
			rows[i] = fmt.Sprintf("- `%s`", modelID)
		}
		lines = append(lines, Truncate(rows)...)
	}
	if len(diff.Changed) > 0 {
		lines = append(lines, fmt.Sprintf("\n**~ %d changed**", len(diff.Changed)))
		rows := []string{}
		for _, modelID := range sortedMapKeys(diff.Changed) {
			fields := diff.Changed[modelID]
			phrases := []string{}
			for _, field := range sortedMapKeys(fields) {
				change := fields[field]
				phrases = append(phrases, DescribeFieldChange(field, change.Old, change.New))
			}
			rows = append(rows, fmt.Sprintf("- `%s`: %s", modelID, strings.Join(phrases, "; ")))
		}
		lines = append(lines, Truncate(rows)...)
	}
	return strings.Join(lines, "\n")
}

// DescribeFieldChange renders one field change phrase.
func DescribeFieldChange(fieldName string, oldVal any, newVal any) string {
	label := fieldLabels[fieldName]
	if label == "" {
		label = fieldName
	}
	_, oldBool := oldVal.(bool)
	newBool, newIsBool := newVal.(bool)
	if oldBool || newIsBool {
		if newBool && !truthy(oldVal) {
			return "added " + label
		}
		if truthy(oldVal) && !newBool {
			return "removed " + label
		}
	}
	return fmt.Sprintf("%s %s -> %s", label, FormatValue(fieldName, oldVal), FormatValue(fieldName, newVal))
}

// BuildSummary renders a multi-provider Markdown summary.
func BuildSummary(providerDiffs map[string]Diff) string {
	sections := []string{}
	for _, provider := range sortedMapKeys(providerDiffs) {
		section := RenderProviderSection(provider, providerDiffs[provider])
		if section != "" {
			sections = append(sections, section)
		}
	}
	if len(sections) == 0 {
		return "No model profile data changed."
	}
	totalAdded := 0
	totalRemoved := 0
	totalChanged := 0
	for _, diff := range providerDiffs {
		totalAdded += len(diff.Added)
		totalRemoved += len(diff.Removed)
		totalChanged += len(diff.Changed)
	}
	headline := fmt.Sprintf("**%d added · %d removed · %d changed** across %d provider(s).", totalAdded, totalRemoved, totalChanged, len(sections))
	return strings.Join(append([]string{"## Summary of changes", headline}, sections...), "\n\n")
}

// Truncate caps bullet rows for Markdown readability.
func Truncate(rows []string) []string {
	if len(rows) <= maxRows {
		return append([]string(nil), rows...)
	}
	hidden := len(rows) - maxRows
	out := append([]string(nil), rows[:maxRows]...)
	out = append(out, fmt.Sprintf("- ...and %d more", hidden))
	return out
}

func keys(reg Registry) []string {
	out := make([]string, 0, len(reg))
	for key := range reg {
		out = append(out, key)
	}
	return out
}

func profileKeys(profile Profile) []string {
	out := make([]string, 0, len(profile))
	for key := range profile {
		out = append(out, key)
	}
	return out
}

func sortedDifference(left []string, right []string) []string {
	rightSet := map[string]bool{}
	for _, value := range right {
		rightSet[value] = true
	}
	out := []string{}
	for _, value := range left {
		if !rightSet[value] {
			out = append(out, value)
		}
	}
	sort.Strings(out)
	return out
}

func sortedIntersection(left []string, right []string) []string {
	rightSet := map[string]bool{}
	for _, value := range right {
		rightSet[value] = true
	}
	out := []string{}
	for _, value := range left {
		if rightSet[value] {
			out = append(out, value)
		}
	}
	sort.Strings(out)
	return out
}

func sortedUnion(left []string, right []string) []string {
	set := map[string]bool{}
	for _, value := range left {
		set[value] = true
	}
	for _, value := range right {
		set[value] = true
	}
	out := make([]string, 0, len(set))
	for value := range set {
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func sortedMapKeys[V any](values map[string]V) []string {
	out := make([]string, 0, len(values))
	for key := range values {
		out = append(out, key)
	}
	sort.Strings(out)
	return out
}

func cloneProfile(profile Profile) Profile {
	out := make(Profile, len(profile))
	for key, value := range profile {
		out[key] = value
	}
	return out
}

func jsonEqual(left any, right any) bool {
	leftJSON, leftErr := json.Marshal(left)
	rightJSON, rightErr := json.Marshal(right)
	return leftErr == nil && rightErr == nil && string(leftJSON) == string(rightJSON)
}

func isTokenField(field string) bool {
	return field == "max_input_tokens" || field == "max_output_tokens"
}

func intValue(value any) (int64, bool) {
	switch v := value.(type) {
	case int:
		return int64(v), true
	case int64:
		return v, true
	case float64:
		if v == float64(int64(v)) {
			return int64(v), true
		}
	}
	return 0, false
}

func truthy(value any) bool {
	v, ok := value.(bool)
	return ok && v
}

func formatInt(value int64) string {
	sign := ""
	if value < 0 {
		sign = "-"
		value = -value
	}
	text := fmt.Sprintf("%d", value)
	if len(text) <= 3 {
		return sign + text
	}
	parts := []string{}
	for len(text) > 3 {
		parts = append([]string{text[len(text)-3:]}, parts...)
		text = text[:len(text)-3]
	}
	parts = append([]string{text}, parts...)
	return sign + strings.Join(parts, ",")
}

package modelprofiles

import (
	"maps"
	"slices"
)

// Declared ModelProfile field names.
//
// These mirror the fields of Python's ModelProfile TypedDict in
// langchain_core.language_models.model_profile. The TypedDict is total=False
// with extra="allow", so a profile may omit any field and may also carry
// additional keys. In Go we model a profile as an open map (Profile) and expose
// the declared field set plus helpers to detect unrecognized keys.
const (
	// Model metadata.
	FieldName        = "name"
	FieldStatus      = "status"
	FieldReleaseDate = "release_date"
	FieldLastUpdated = "last_updated"
	FieldOpenWeights = "open_weights"

	// Input constraints.
	FieldMaxInputTokens = "max_input_tokens"
	FieldTextInputs     = "text_inputs"
	FieldImageInputs    = "image_inputs"
	FieldImageURLInputs = "image_url_inputs"
	FieldPDFInputs      = "pdf_inputs"
	FieldAudioInputs    = "audio_inputs"
	FieldVideoInputs    = "video_inputs"
	FieldImageToolMsg   = "image_tool_message"
	FieldPDFToolMsg     = "pdf_tool_message"

	// Output constraints.
	FieldMaxOutputTokens = "max_output_tokens"
	FieldReasoningOutput = "reasoning_output"

	// Reasoning effort (langchain-core model_profile.py:94-108; anthropic
	// bundle data since the #41536-era augmentations).
	FieldReasoningEffortLevels  = "reasoning_effort_levels"
	FieldReasoningEffortDefault = "reasoning_effort_default"

	FieldTextOutputs   = "text_outputs"
	FieldImageOutputs  = "image_outputs"
	FieldAudioOutputs  = "audio_outputs"
	FieldVideoOutputs  = "video_outputs"

	// Tool calling.
	FieldToolCalling    = "tool_calling"
	FieldToolChoice     = "tool_choice"
	FieldToolCallStream = "tool_call_streaming"

	// Mid-conversation changes (Python #41180, langchain-core 1.6.10).
	FieldMidConversationSystemMessages = "mid_conversation_system_messages"
	FieldMidConversationTools          = "mid_conversation_tools"

	// Structured output.
	FieldStructuredOutput = "structured_output"

	// Other capabilities.
	FieldAttachment  = "attachment"
	FieldTemperature = "temperature"
)

// declaredProfileFields is the set of ModelProfile field names declared by the
// Python TypedDict. It backs UnknownProfileKeys and IsDeclaredProfileField.
var declaredProfileFields = map[string]struct{}{
	FieldName:                          {},
	FieldStatus:                        {},
	FieldReleaseDate:                   {},
	FieldLastUpdated:                   {},
	FieldOpenWeights:                   {},
	FieldMaxInputTokens:                {},
	FieldTextInputs:                    {},
	FieldImageInputs:                   {},
	FieldImageURLInputs:                {},
	FieldPDFInputs:                     {},
	FieldAudioInputs:                   {},
	FieldVideoInputs:                   {},
	FieldImageToolMsg:                  {},
	FieldPDFToolMsg:                    {},
	FieldMaxOutputTokens:               {},
	FieldReasoningOutput:               {},
	FieldReasoningEffortLevels:         {},
	FieldReasoningEffortDefault:        {},
	FieldTextOutputs:                   {},
	FieldImageOutputs:                  {},
	FieldAudioOutputs:                  {},
	FieldVideoOutputs:                  {},
	FieldToolCalling:                   {},
	FieldToolChoice:                    {},
	FieldToolCallStream:                {},
	FieldMidConversationSystemMessages: {},
	FieldMidConversationTools:          {},
	FieldStructuredOutput:              {},
	FieldAttachment:                    {},
	FieldTemperature:                   {},
}

// DeclaredProfileFields returns the sorted set of ModelProfile field names
// declared by the schema. Unknown keys outside this set are still permitted on
// a Profile (extra="allow" semantics) but can be surfaced via UnknownProfileKeys.
func DeclaredProfileFields() []string {
	return slices.Sorted(maps.Keys(declaredProfileFields))
}

// IsDeclaredProfileField reports whether name is a declared ModelProfile field.
func IsDeclaredProfileField(name string) bool {
	_, ok := declaredProfileFields[name]
	return ok
}

// Supports reports whether the boolean profile field key is present and true.
//
// Absent fields and explicit false both report false, matching the Python
// idiom `profile.get(key, False)`; use a direct map lookup on the Profile when
// the distinction between "absent" and "false" matters (upstream declares
// mid_conversation_tools = false for some models, e.g. claude-sonnet-5).
func (p Profile) Supports(key string) bool {
	value, ok := p[key].(bool)
	return ok && value
}

// ReasoningEffortLevels returns the reasoning-effort levels declared in the
// profile, mirroring Python's defensive _reasoning_effort_levels helper
// (langchain_anthropic chat_models.py:1140-1152): an absent key, a non-list
// value, or non-string entries read as "not declared" rather than erroring, so
// a malformed or future profile never breaks a lookup.
func (p Profile) ReasoningEffortLevels() []string {
	raw, ok := p[FieldReasoningEffortLevels].([]any)
	if !ok {
		return nil
	}
	var levels []string
	for _, item := range raw {
		if level, isString := item.(string); isString {
			levels = append(levels, level)
		}
	}
	return levels
}

// ReasoningEffortDefault returns the provider's documented default
// reasoning-effort level and whether the profile declares one. Python's
// ChatAnthropic never resolves this default client-side (the value documents
// API behavior, e.g. Opus 5.5 defaults to "medium"); the accessor exists for
// schema completeness and consumers that surface the documented default.
func (p Profile) ReasoningEffortDefault() (string, bool) {
	level, ok := p[FieldReasoningEffortDefault].(string)
	return level, ok
}

// UnknownProfileKeys returns the sorted profile keys that are not declared
// ModelProfile fields.
//
// This is the offline-testable Go equivalent of Python's
// _warn_unknown_profile_keys: rather than emitting a runtime warning, it returns
// the unrecognized keys so callers can decide how to report a version mismatch
// between core and a provider package. An empty result means the profile only
// uses declared fields.
func UnknownProfileKeys(profile Profile) []string {
	if profile == nil {
		return nil
	}
	var extra []string
	for key := range profile {
		if _, ok := declaredProfileFields[key]; !ok {
			extra = append(extra, key)
		}
	}
	slices.Sort(extra)
	return extra
}

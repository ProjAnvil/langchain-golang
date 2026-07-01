package modelprofiles

import "sort"

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
	FieldTextOutputs     = "text_outputs"
	FieldImageOutputs    = "image_outputs"
	FieldAudioOutputs    = "audio_outputs"
	FieldVideoOutputs    = "video_outputs"

	// Tool calling.
	FieldToolCalling      = "tool_calling"
	FieldToolChoice       = "tool_choice"
	FieldToolCallStream   = "tool_call_streaming"
	FieldStructuredOutput = "structured_output"

	// Other capabilities.
	FieldAttachment  = "attachment"
	FieldTemperature = "temperature"
)

// declaredProfileFields is the set of ModelProfile field names declared by the
// Python TypedDict. It backs UnknownProfileKeys and IsDeclaredProfileField.
var declaredProfileFields = map[string]struct{}{
	FieldName:             {},
	FieldStatus:           {},
	FieldReleaseDate:      {},
	FieldLastUpdated:      {},
	FieldOpenWeights:      {},
	FieldMaxInputTokens:   {},
	FieldTextInputs:       {},
	FieldImageInputs:      {},
	FieldImageURLInputs:   {},
	FieldPDFInputs:        {},
	FieldAudioInputs:      {},
	FieldVideoInputs:      {},
	FieldImageToolMsg:     {},
	FieldPDFToolMsg:       {},
	FieldMaxOutputTokens:  {},
	FieldReasoningOutput:  {},
	FieldTextOutputs:      {},
	FieldImageOutputs:     {},
	FieldAudioOutputs:     {},
	FieldVideoOutputs:     {},
	FieldToolCalling:      {},
	FieldToolChoice:       {},
	FieldToolCallStream:   {},
	FieldStructuredOutput: {},
	FieldAttachment:       {},
	FieldTemperature:      {},
}

// DeclaredProfileFields returns the sorted set of ModelProfile field names
// declared by the schema. Unknown keys outside this set are still permitted on
// a Profile (extra="allow" semantics) but can be surfaced via UnknownProfileKeys.
func DeclaredProfileFields() []string {
	fields := make([]string, 0, len(declaredProfileFields))
	for field := range declaredProfileFields {
		fields = append(fields, field)
	}
	sort.Strings(fields)
	return fields
}

// IsDeclaredProfileField reports whether name is a declared ModelProfile field.
func IsDeclaredProfileField(name string) bool {
	_, ok := declaredProfileFields[name]
	return ok
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
	sort.Strings(extra)
	return extra
}

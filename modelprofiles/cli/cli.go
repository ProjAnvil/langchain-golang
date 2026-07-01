// Package cli ports the "refresh" workflow of Python's
// langchain_model_profiles.cli module to Go: it downloads model capability
// data from models.dev, merges local overrides declared in
// `profile_augmentations.toml`, and writes a canonical `profiles.json` data
// file for a single provider.
//
// This intentionally writes JSON rather than a generated Python-style module,
// since Go partner packages are expected to load profile data via
// encoding/json (optionally with go:embed) rather than a generated .go source
// file. The data shape (a map of model ID to profile fields) is identical to
// Python's `_PROFILES` dict.
package cli

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/projanvil/langchain-golang/modelprofiles"
)

// DefaultAPIURL is the models.dev endpoint used by Refresh unless overridden.
const DefaultAPIURL = "https://models.dev/api.json"

const augmentationsFileName = "profile_augmentations.toml"
const profilesFileName = "profiles.json"

// Profile and Registry alias the shared model profile types so callers of
// this package do not need to import modelprofiles directly for basic use.
type Profile = modelprofiles.Profile
type Registry = modelprofiles.Registry

// RefreshOptions configures a Refresh invocation. HTTPClient, Stdout, Stderr,
// and Confirm all have safe defaults so only Provider and DataDir are
// required for typical use; the other fields exist primarily to make Refresh
// deterministically testable.
type RefreshOptions struct {
	// Provider is the models.dev provider ID, e.g. "anthropic" or "openai".
	Provider string
	// DataDir contains (or will contain) profile_augmentations.toml and is
	// where profiles.json is written.
	DataDir string
	// APIURL overrides the models.dev endpoint. Defaults to DefaultAPIURL.
	APIURL string
	// HTTPClient overrides the HTTP client used to fetch APIURL. Defaults to
	// http.DefaultClient.
	HTTPClient *http.Client
	// Stdout and Stderr receive progress and warning output. Default to
	// io.Discard when nil.
	Stdout io.Writer
	Stderr io.Writer
	// Confirm is invoked only when DataDir resolves outside the current
	// working directory. It should return true to continue writing there. A
	// nil Confirm treats an out-of-cwd data dir as declined (aborts).
	Confirm func() bool
	// Cwd overrides the working directory used to decide whether DataDir is
	// "outside the current directory". Defaults to os.Getwd().
	Cwd string
}

// Refresh downloads and merges model profile data for a single provider and
// writes the result to <data-dir>/profiles.json. It mirrors Python's
// `langchain_model_profiles.cli.refresh`.
func Refresh(opts RefreshOptions) error {
	stdout := opts.Stdout
	if stdout == nil {
		stdout = io.Discard
	}
	stderr := opts.Stderr
	if stderr == nil {
		stderr = io.Discard
	}

	cwd := opts.Cwd
	if cwd == "" {
		wd, err := os.Getwd()
		if err != nil {
			return fmt.Errorf("failed to determine current directory: %w", err)
		}
		cwd = wd
	}

	dataDir, needsConfirmation, err := ValidateDataDir(opts.DataDir, cwd)
	if err != nil {
		return err
	}
	if needsConfirmation {
		fmt.Fprintln(stderr, "WARNING: Writing outside current directory")
		fmt.Fprintf(stderr, "   Current directory: %s\n", cwd)
		fmt.Fprintf(stderr, "   Target directory:  %s\n", dataDir)
		if opts.Confirm == nil || !opts.Confirm() {
			return fmt.Errorf("aborted: refusing to write outside %s without confirmation", cwd)
		}
	}

	fmt.Fprintf(stdout, "Provider: %s\n", opts.Provider)
	fmt.Fprintf(stdout, "Data directory: %s\n", dataDir)

	apiURL := opts.APIURL
	if apiURL == "" {
		apiURL = DefaultAPIURL
	}
	client := opts.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	fmt.Fprintf(stdout, "Downloading data from %s...\n", apiURL)
	allData, err := fetchModelsDevData(client, apiURL)
	if err != nil {
		return err
	}

	providerCount := len(allData)
	modelCount := 0
	for _, p := range allData {
		if pm, ok := p.(map[string]any); ok {
			if models, ok := pm["models"].(map[string]any); ok {
				modelCount += len(models)
			}
		}
	}
	fmt.Fprintf(stdout, "Downloaded %d providers with %d models\n", providerCount, modelCount)

	providerData, ok := allData[opts.Provider].(map[string]any)
	if !ok {
		return fmt.Errorf("provider %q not found in models.dev data", opts.Provider)
	}
	models, _ := providerData["models"].(map[string]any)
	fmt.Fprintf(stdout, "Extracted %d models for %s\n", len(models), opts.Provider)

	fmt.Fprintln(stdout, "Loading augmentations...")
	providerAug, modelAugs, err := LoadAugmentations(dataDir)
	if err != nil {
		return err
	}

	profiles := Registry{}
	for modelID, rawModelData := range models {
		modelData, _ := rawModelData.(map[string]any)
		base := ModelDataToProfile(modelData)
		profiles[modelID] = ApplyOverrides(base, providerAug, modelAugs[modelID])
	}

	var extraModels []string
	for modelID := range modelAugs {
		if _, exists := models[modelID]; !exists {
			extraModels = append(extraModels, modelID)
		}
	}
	sort.Strings(extraModels)
	if len(extraModels) > 0 {
		fmt.Fprintf(stdout, "Adding %d models from augmentations only...\n", len(extraModels))
	}
	for _, modelID := range extraModels {
		profiles[modelID] = ApplyOverrides(Profile{}, providerAug, modelAugs[modelID])
	}

	if unknown := unknownKeysAcross(profiles); len(unknown) > 0 {
		fmt.Fprintf(stderr,
			"warning: profile keys not declared in modelprofiles.ModelProfile: %s. "+
				"Add these fields to modelprofiles field declarations before publishing "+
				"partner packages that use these profiles.\n",
			strings.Join(unknown, ", "))
	}

	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		return fmt.Errorf("failed to create directory %s: %w", dataDir, err)
	}

	outputFile := filepath.Join(dataDir, profilesFileName)
	fmt.Fprintf(stdout, "Writing to %s...\n", outputFile)
	contents, err := BuildProfilesJSON(profiles)
	if err != nil {
		return fmt.Errorf("failed to encode profiles: %w", err)
	}
	if err := writeProfilesFileAtomic(dataDir, outputFile, contents); err != nil {
		return err
	}

	fmt.Fprintf(stdout, "Successfully refreshed %d model profiles (%d bytes)\n", len(profiles), len(contents))
	return nil
}

func fetchModelsDevData(client *http.Client, apiURL string) (map[string]any, error) {
	resp, err := client.Get(apiURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to %s: %w", apiURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("HTTP error %d from %s", resp.StatusCode, apiURL)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response from %s: %w", apiURL, err)
	}

	var allData map[string]any
	if err := json.Unmarshal(body, &allData); err != nil {
		return nil, fmt.Errorf("invalid JSON response from API: %w", err)
	}
	return allData, nil
}

// ModelDataToProfile converts a single raw models.dev model entry into the
// canonical profile structure. It mirrors Python's
// `langchain_model_profiles.cli._model_data_to_profile`: fields that resolve
// to nil (Python `None`) are omitted, but explicit `false`/zero values are
// kept.
func ModelDataToProfile(modelData map[string]any) Profile {
	limit, _ := modelData["limit"].(map[string]any)
	modalities, _ := modelData["modalities"].(map[string]any)
	inputModalities := toStringSet(modalities["input"])
	outputModalities := toStringSet(modalities["output"])

	var pdfInputs any
	if inputModalities["pdf"] {
		pdfInputs = true
	} else {
		pdfInputs = modelData["pdf_inputs"]
	}

	raw := Profile{
		modelprofiles.FieldName:             modelData["name"],
		modelprofiles.FieldStatus:           modelData["status"],
		modelprofiles.FieldReleaseDate:      modelData["release_date"],
		modelprofiles.FieldLastUpdated:      modelData["last_updated"],
		modelprofiles.FieldOpenWeights:      modelData["open_weights"],
		modelprofiles.FieldMaxInputTokens:   limit["context"],
		modelprofiles.FieldMaxOutputTokens:  limit["output"],
		modelprofiles.FieldTextInputs:       inputModalities["text"],
		modelprofiles.FieldImageInputs:      inputModalities["image"],
		modelprofiles.FieldAudioInputs:      inputModalities["audio"],
		modelprofiles.FieldPDFInputs:        pdfInputs,
		modelprofiles.FieldVideoInputs:      inputModalities["video"],
		modelprofiles.FieldTextOutputs:      outputModalities["text"],
		modelprofiles.FieldImageOutputs:     outputModalities["image"],
		modelprofiles.FieldAudioOutputs:     outputModalities["audio"],
		modelprofiles.FieldVideoOutputs:     outputModalities["video"],
		modelprofiles.FieldReasoningOutput:  modelData["reasoning"],
		modelprofiles.FieldToolCalling:      modelData["tool_call"],
		modelprofiles.FieldToolChoice:       modelData["tool_choice"],
		modelprofiles.FieldToolCallStream:   modelData["tool_call_streaming"],
		modelprofiles.FieldStructuredOutput: modelData["structured_output"],
		modelprofiles.FieldAttachment:       modelData["attachment"],
		modelprofiles.FieldTemperature:      modelData["temperature"],
		modelprofiles.FieldImageURLInputs:   modelData["image_url_inputs"],
		modelprofiles.FieldImageToolMsg:     modelData["image_tool_message"],
		modelprofiles.FieldPDFToolMsg:       modelData["pdf_tool_message"],
	}

	profile := Profile{}
	for key, value := range raw {
		if value != nil {
			profile[key] = value
		}
	}
	return profile
}

// ApplyOverrides merges provider- and model-level overrides onto profile,
// mirroring Python's `_apply_overrides`. Later overrides win; nil override
// values are skipped so they never blank out a base value. profile itself is
// not mutated.
func ApplyOverrides(profile Profile, overrides ...Profile) Profile {
	merged := make(Profile, len(profile))
	for key, value := range profile {
		merged[key] = value
	}
	for _, override := range overrides {
		for key, value := range override {
			if value != nil {
				merged[key] = value
			}
		}
	}
	return merged
}

// LoadAugmentations reads and parses <dataDir>/profile_augmentations.toml,
// mirroring Python's `_load_augmentations`. A missing file is not an error:
// it returns empty results, matching Python's `if not aug_file.exists()`
// short-circuit.
func LoadAugmentations(dataDir string) (providerAug Profile, modelAugs map[string]Profile, err error) {
	path := filepath.Join(dataDir, augmentationsFileName)
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return Profile{}, map[string]Profile{}, nil
		}
		return nil, nil, fmt.Errorf("failed to read augmentations file: %w", err)
	}
	return ParseAugmentations(data)
}

// ParseAugmentations parses the contents of a profile_augmentations.toml
// file. Top-level scalar entries under `[overrides]` become provider-level
// overrides; sub-tables under `[overrides."<model-id>"]` become per-model
// overrides. All other top-level keys (e.g. `provider = "..."`) are ignored,
// matching Python behavior.
func ParseAugmentations(data []byte) (providerAug Profile, modelAugs map[string]Profile, err error) {
	doc, err := parseTOMLSubset(data)
	if err != nil {
		return nil, nil, fmt.Errorf("invalid TOML syntax in augmentations file: %w", err)
	}

	providerAug = Profile{}
	modelAugs = map[string]Profile{}

	overridesRaw, ok := doc["overrides"]
	if !ok {
		return providerAug, modelAugs, nil
	}
	overrides, ok := overridesRaw.(map[string]any)
	if !ok {
		return nil, nil, fmt.Errorf("[overrides] must be a table")
	}

	for key, value := range overrides {
		if table, isTable := value.(map[string]any); isTable {
			profile := Profile{}
			for k, v := range table {
				profile[k] = v
			}
			modelAugs[key] = profile
		} else {
			providerAug[key] = value
		}
	}
	return providerAug, modelAugs, nil
}

// ValidateDataDir resolves dataDir to an absolute, cleaned path and reports
// whether it falls outside cwd, mirroring Python's `_validate_data_dir`
// (minus the interactive prompt, which callers implement via
// RefreshOptions.Confirm).
func ValidateDataDir(dataDir, cwd string) (resolved string, needsConfirmation bool, err error) {
	if dataDir == "" {
		return "", false, fmt.Errorf("data directory must not be empty")
	}
	absDataDir, err := filepath.Abs(dataDir)
	if err != nil {
		return "", false, fmt.Errorf("invalid data directory path: %w", err)
	}
	absDataDir = filepath.Clean(absDataDir)

	absCwd, err := filepath.Abs(cwd)
	if err != nil {
		return "", false, fmt.Errorf("invalid current directory path: %w", err)
	}
	absCwd = filepath.Clean(absCwd)

	rel, err := filepath.Rel(absCwd, absDataDir)
	outside := err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator))
	return absDataDir, outside, nil
}

// BuildProfilesJSON renders profiles as indented, deterministically ordered
// JSON (Go's encoding/json sorts map keys), analogous to Python's
// `json.dumps(dict(sorted(profiles.items())), indent=4)`.
func BuildProfilesJSON(profiles Registry) ([]byte, error) {
	out, err := json.MarshalIndent(profiles, "", "    ")
	if err != nil {
		return nil, err
	}
	return append(out, '\n'), nil
}

func writeProfilesFileAtomic(dataDir, outputFile string, contents []byte) error {
	if err := ensureSafeOutputPath(dataDir, outputFile); err != nil {
		return err
	}

	tmp, err := os.CreateTemp(dataDir, ".profiles-*.tmp")
	if err != nil {
		return fmt.Errorf("failed to create temp file: %w", err)
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)

	if _, err := tmp.Write(contents); err != nil {
		tmp.Close()
		return fmt.Errorf("failed to write file: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}
	if err := os.Rename(tmpPath, outputFile); err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}
	return nil
}

// ensureSafeOutputPath rejects symlinked data directories/output files and
// output paths that escape dataDir, mirroring Python's
// `_ensure_safe_output_path`.
func ensureSafeOutputPath(dataDir, outputFile string) error {
	if info, err := os.Lstat(dataDir); err == nil && info.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("data directory %s is a symlink; refusing to write profiles", dataDir)
	}
	if info, err := os.Lstat(outputFile); err == nil && info.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("%s is a symlink; refusing to overwrite it", outputFile)
	}

	absDataDir, err := filepath.Abs(dataDir)
	if err != nil {
		return fmt.Errorf("failed to resolve data directory: %w", err)
	}
	absOutputFile, err := filepath.Abs(outputFile)
	if err != nil {
		return fmt.Errorf("failed to resolve output path: %w", err)
	}
	rel, err := filepath.Rel(absDataDir, absOutputFile)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return fmt.Errorf("refusing to write outside of data directory: %s", outputFile)
	}
	return nil
}

func unknownKeysAcross(profiles Registry) []string {
	seen := map[string]struct{}{}
	for _, profile := range profiles {
		for _, key := range modelprofiles.UnknownProfileKeys(profile) {
			seen[key] = struct{}{}
		}
	}
	out := make([]string, 0, len(seen))
	for key := range seen {
		out = append(out, key)
	}
	sort.Strings(out)
	return out
}

func toStringSet(v any) map[string]bool {
	set := map[string]bool{}
	items, _ := v.([]any)
	for _, item := range items {
		if s, ok := item.(string); ok {
			set[s] = true
		}
	}
	return set
}

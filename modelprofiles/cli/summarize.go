// This file ports the "summarize" workflow of Python's
// langchain_model_profiles._summary module to Go.
//
// Divergence from Python: Python's `summarize` compares a generated
// `_profiles.py` module at a git ref (`_verify_ref`/`_git_show`) against the
// working-tree copy, extracting the `_PROFILES` dict via `ast.literal_eval`
// (`extract_profiles`). Go stores profiles as JSON, so `Summarize` and
// `RunSummarize` read before/after `profiles.json` snapshots directly from
// disk. No git or Python-AST extraction is needed, so `_verify_ref`,
// `_git_show`, and `extract_profiles` have no Go equivalent here.
package cli

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/projanvil/langchain-golang/modelprofiles"
)

// Summarize renders the Markdown summary of model profile changes between two
// registries for a single provider, mirroring Python's
// `langchain_model_profiles._summary.build_summary`. It is the pure, testable
// entry point: it takes already-loaded before/after registries and returns the
// rendered summary without touching the filesystem or git.
//
// The provider argument names the section heading (Python's per-provider
// `provider` key). The diff and rendering reuse the already-ported
// `DiffProfiles`/`RenderProviderSection`/`BuildSummary` and therefore keep the
// existing ASCII rendering (`+`/`-`/`~`, `...and N more`) rather than Python's
// emoji/`<details>` rendering.
func Summarize(provider string, before, after Registry) string {
	return modelprofiles.BuildSummary(map[string]modelprofiles.Diff{
		provider: modelprofiles.DiffProfiles(before, after),
	})
}

// LoadProfiles reads and decodes a profiles.json snapshot written by Refresh.
// It is the Go analogue of Python's `extract_profiles`, with JSON decoding
// replacing Python-AST extraction.
func LoadProfiles(path string) (Registry, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read profiles snapshot %s: %w", path, err)
	}
	var profiles Registry
	if err := json.Unmarshal(data, &profiles); err != nil {
		return nil, fmt.Errorf("invalid profiles JSON in %s: %w", path, err)
	}
	return profiles, nil
}

// SummarizeOptions configures a RunSummarize invocation. Stdout and Stderr
// have safe defaults (io.Discard) so only Provider, Before, and either After
// or DataDir are required for typical use.
type SummarizeOptions struct {
	// Provider is the provider ID used as the summary section heading, e.g.
	// "anthropic" or "openai".
	Provider string
	// Before is the path to the "before" profiles.json snapshot.
	Before string
	// After is the path to the "after" profiles.json snapshot. When empty, it
	// defaults to <DataDir>/profiles.json (the path Refresh writes).
	After string
	// DataDir provides the default "after" path when After is empty.
	DataDir string
	// Stdout receives the rendered summary. Defaults to io.Discard when nil.
	Stdout io.Writer
	// Stderr receives warning/error context. Defaults to io.Discard when nil.
	Stderr io.Writer
}

// RunSummarize loads the before/after profiles.json snapshots, computes the
// diff, and writes the rendered Markdown summary to opts.Stdout. It mirrors the
// data flow of Python's `langchain_model_profiles._summary.summarize` minus the
// git ref comparison (see the package-level divergence note).
func RunSummarize(opts SummarizeOptions) error {
	stdout := opts.Stdout
	if stdout == nil {
		stdout = io.Discard
	}

	if opts.Provider == "" {
		return fmt.Errorf("provider must not be empty")
	}
	if opts.Before == "" {
		return fmt.Errorf("before snapshot path must not be empty")
	}
	afterPath := opts.After
	if afterPath == "" {
		if opts.DataDir == "" {
			return fmt.Errorf("after snapshot path (or data-dir) must not be empty")
		}
		afterPath = filepath.Join(opts.DataDir, profilesFileName)
	}

	before, err := LoadProfiles(opts.Before)
	if err != nil {
		return err
	}
	after, err := LoadProfiles(afterPath)
	if err != nil {
		return err
	}

	fmt.Fprintln(stdout, Summarize(opts.Provider, before, after))
	return nil
}

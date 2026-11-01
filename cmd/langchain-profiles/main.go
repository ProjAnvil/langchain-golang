// Command langchain-profiles refreshes model profile data from models.dev and
// summarizes profile changes, the Go equivalent of Python's
// `langchain-profiles` CLI (langchain_model_profiles.cli).
//
// Usage:
//
//	langchain-profiles refresh --provider anthropic --data-dir ./data
//	langchain-profiles summarize --provider anthropic --data-dir ./data --before /tmp/before.json
//	langchain-profiles materialize [--providers anthropic,openai] [--data-root modelprofiles/data] [--out modelprofiles/profiles.json]
//
// materialize fetches every listed provider, merges each provider's
// augmentations from <data-root>/<provider>/profile_augmentations.toml, and
// writes one combined bundle to --out. That bundle is what the modelprofiles
// package embeds for Lookup, so regenerate it (from the repo root) after
// changing an augmentations file:
//
//	go run ./cmd/langchain-profiles materialize
package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/projanvil/langchain-golang/modelprofiles/cli"
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr))
}

func run(args []string, stdin *os.File, stdout, stderr *os.File) int {
	if len(args) == 0 {
		_, _ = fmt.Fprintln(stderr, "expected a subcommand, e.g. \"refresh\"")
		printUsage(stderr)
		return 1
	}

	switch args[0] {
	case "refresh":
		return runRefresh(args[1:], stdin, stdout, stderr)
	case "summarize":
		return runSummarize(args[1:], stdin, stdout, stderr)
	case "materialize":
		return runMaterialize(args[1:], stdout, stderr)
	case "-h", "--help", "help":
		printUsage(stdout)
		return 0
	default:
		_, _ = fmt.Fprintf(stderr, "unknown command %q\n", args[0])
		printUsage(stderr)
		return 1
	}
}

func runRefresh(args []string, stdin *os.File, stdout, stderr *os.File) int {
	fs := flag.NewFlagSet("refresh", flag.ContinueOnError)
	fs.SetOutput(stderr)
	provider := fs.String("provider", "", "Provider ID from models.dev (e.g. 'anthropic', 'openai', 'google')")
	dataDir := fs.String("data-dir", "", "Data directory containing profile_augmentations.toml")
	if err := fs.Parse(args); err != nil {
		return 1
	}
	if *provider == "" || *dataDir == "" {
		_, _ = fmt.Fprintln(stderr, "both --provider and --data-dir are required")
		fs.Usage()
		return 1
	}

	reader := bufio.NewReader(stdin)
	err := cli.Refresh(cli.RefreshOptions{
		Provider: *provider,
		DataDir:  *dataDir,
		Stdout:   stdout,
		Stderr:   stderr,
		Confirm: func() bool {
			_, _ = fmt.Fprint(stderr, "Continue? (y/N): ")
			line, _ := reader.ReadString('\n')
			return strings.EqualFold(strings.TrimSpace(line), "y")
		},
	})
	if err != nil {
		_, _ = fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}
	return 0
}

func runSummarize(args []string, stdin *os.File, stdout, stderr *os.File) int {
	fs := flag.NewFlagSet("summarize", flag.ContinueOnError)
	fs.SetOutput(stderr)
	provider := fs.String("provider", "", "Provider ID used as the summary section heading (e.g. 'anthropic', 'openai')")
	dataDir := fs.String("data-dir", "", "Data directory; --after defaults to <data-dir>/profiles.json")
	before := fs.String("before", "", "Path to the 'before' profiles.json snapshot")
	after := fs.String("after", "", "Path to the 'after' profiles.json snapshot (defaults to <data-dir>/profiles.json)")
	if err := fs.Parse(args); err != nil {
		return 1
	}
	if *provider == "" || *dataDir == "" || *before == "" {
		_, _ = fmt.Fprintln(stderr, "--provider, --data-dir, and --before are required")
		fs.Usage()
		return 1
	}

	err := cli.RunSummarize(cli.SummarizeOptions{
		Provider: *provider,
		DataDir:  *dataDir,
		Before:   *before,
		After:    *after,
		Stdout:   stdout,
		Stderr:   stderr,
	})
	if err != nil {
		_, _ = fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}
	return 0
}

// runMaterialize fetches profile data for several providers and writes one
// combined provider-to-registry bundle, the data file the modelprofiles
// package embeds. Unlike refresh it never writes inside the data dirs, so no
// out-of-cwd confirmation is needed; --out is written exactly where asked.
func runMaterialize(args []string, stdout, stderr *os.File) int {
	fs := flag.NewFlagSet("materialize", flag.ContinueOnError)
	fs.SetOutput(stderr)
	providers := fs.String("providers", "anthropic,openai", "Comma-separated provider IDs from models.dev")
	dataRoot := fs.String("data-root", "modelprofiles/data", "Root directory holding <provider>/profile_augmentations.toml")
	out := fs.String("out", "modelprofiles/profiles.json", "Output path for the combined profiles bundle")
	apiURL := fs.String("api-url", "", "Override the models.dev endpoint (defaults to "+cli.DefaultAPIURL+")")
	if err := fs.Parse(args); err != nil {
		return 1
	}

	var providerIDs []string
	for _, provider := range strings.Split(*providers, ",") {
		if id := strings.TrimSpace(provider); id != "" {
			providerIDs = append(providerIDs, id)
		}
	}
	if len(providerIDs) == 0 || *dataRoot == "" || *out == "" {
		_, _ = fmt.Fprintln(stderr, "--providers, --data-root, and --out must not be empty")
		fs.Usage()
		return 1
	}

	bundle := map[string]cli.Registry{}
	for _, provider := range providerIDs {
		profiles, err := cli.FetchProfiles(cli.RefreshOptions{
			Provider: provider,
			DataDir:  filepath.Join(*dataRoot, provider),
			APIURL:   *apiURL,
			Stdout:   stdout,
			Stderr:   stderr,
		})
		if err != nil {
			_, _ = fmt.Fprintf(stderr, "error: %v\n", err)
			return 1
		}
		bundle[provider] = profiles
	}

	contents, err := cli.BuildRegistriesJSON(bundle)
	if err != nil {
		_, _ = fmt.Fprintf(stderr, "error: failed to encode bundle: %v\n", err)
		return 1
	}
	if dir := filepath.Dir(*out); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			_, _ = fmt.Fprintf(stderr, "error: failed to create output directory: %v\n", err)
			return 1
		}
	}
	if err := os.WriteFile(*out, contents, 0o644); err != nil {
		_, _ = fmt.Fprintf(stderr, "error: failed to write %s: %v\n", *out, err)
		return 1
	}
	_, _ = fmt.Fprintf(stdout, "Materialized %d provider(s) into %s (%d bytes)\n", len(bundle), *out, len(contents))
	return 0
}

func printUsage(w *os.File) {
	_, _ = fmt.Fprintln(w, "langchain-profiles refreshes model profile data from models.dev")
	_, _ = fmt.Fprintln(w, "")
	_, _ = fmt.Fprintln(w, "Usage:")
	_, _ = fmt.Fprintln(w, "  langchain-profiles refresh --provider <id> --data-dir <dir>")
	_, _ = fmt.Fprintln(w, "  langchain-profiles summarize --provider <id> --data-dir <dir> --before <path> [--after <path>]")
	_, _ = fmt.Fprintln(w, "  langchain-profiles materialize [--providers <ids>] [--data-root <dir>] [--out <path>]")
}

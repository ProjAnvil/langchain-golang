// Command langchain-profiles refreshes model profile data from models.dev,
// the Go equivalent of Python's `langchain-profiles` CLI
// (langchain_model_profiles.cli).
//
// Usage:
//
//	langchain-profiles refresh --provider anthropic --data-dir ./data
package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/projanvil/langchain-golang/modelprofiles/cli"
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr))
}

func run(args []string, stdin *os.File, stdout, stderr *os.File) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, "expected a subcommand, e.g. \"refresh\"")
		printUsage(stderr)
		return 1
	}

	switch args[0] {
	case "refresh":
		return runRefresh(args[1:], stdin, stdout, stderr)
	case "-h", "--help", "help":
		printUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "unknown command %q\n", args[0])
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
		fmt.Fprintln(stderr, "both --provider and --data-dir are required")
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
			fmt.Fprint(stderr, "Continue? (y/N): ")
			line, _ := reader.ReadString('\n')
			return strings.EqualFold(strings.TrimSpace(line), "y")
		},
	})
	if err != nil {
		fmt.Fprintf(stderr, "error: %v\n", err)
		return 1
	}
	return 0
}

func printUsage(w *os.File) {
	fmt.Fprintln(w, "langchain-profiles refreshes model profile data from models.dev")
	fmt.Fprintln(w, "")
	fmt.Fprintln(w, "Usage:")
	fmt.Fprintln(w, "  langchain-profiles refresh --provider <id> --data-dir <dir>")
}

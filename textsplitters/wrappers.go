package textsplitters

import (
	"regexp"
)

// NewPythonCode creates a Python syntax-aware recursive splitter.
func NewPythonCode(cfg Config) (*RecursiveCharacterTextSplitter, error) {
	return NewRecursiveCharacterFromLanguage(LanguagePython, cfg)
}

// NewLatex creates a LaTeX syntax-aware recursive splitter.
func NewLatex(cfg Config) (*RecursiveCharacterTextSplitter, error) {
	return NewRecursiveCharacterFromLanguage(LanguageLatex, cfg)
}

// NewHTML creates an HTML syntax-aware recursive splitter.
func NewHTML(cfg Config) (*RecursiveCharacterTextSplitter, error) {
	return NewRecursiveCharacterFromLanguage(LanguageHTML, cfg)
}

// JSFrameworkTextSplitter handles JSX/Vue/Svelte-like component code.
type JSFrameworkTextSplitter struct {
	Config     Config
	Separators []string
}

// NewJSFramework creates a JS framework splitter.
func NewJSFramework(separators []string, cfg Config) JSFrameworkTextSplitter {
	if cfg.ChunkSize == 0 {
		cfg.ChunkSize = 2000
	}
	return JSFrameworkTextSplitter{
		Config:     cfg,
		Separators: append([]string(nil), separators...),
	}
}

// SplitText extracts component tags and delegates to recursive splitting.
func (s JSFrameworkTextSplitter) SplitText(text string) ([]string, error) {
	separators := append([]string(nil), s.Separators...)
	separators = append(separators, jsFrameworkSeparators...)
	separators = append(separators, componentSeparators(text)...)
	separators = append(separators, "<>", "\n\n", "&&\n", "||\n")
	splitter, err := NewRecursiveCharacter(separators, false, s.Config)
	if err != nil {
		return nil, err
	}
	return splitter.SplitText(text), nil
}

var jsFrameworkSeparators = []string{
	"\nexport ",
	" export ",
	"\nfunction ",
	"\nasync function ",
	" async function ",
	"\nconst ",
	"\nlet ",
	"\nvar ",
	"\nclass ",
	" class ",
	"\nif ",
	" if ",
	"\nfor ",
	" for ",
	"\nwhile ",
	" while ",
	"\nswitch ",
	" switch ",
	"\ncase ",
	" case ",
	"\ndefault ",
	" default ",
}

func componentSeparators(text string) []string {
	re := regexp.MustCompile(`<\s*([a-zA-Z0-9]+)[^>]*>`)
	matches := re.FindAllStringSubmatch(text, -1)
	seen := map[string]bool{}
	out := []string{}
	for _, match := range matches {
		if len(match) < 2 || seen[match[1]] {
			continue
		}
		seen[match[1]] = true
		out = append(out, "<"+match[1])
	}
	return out
}

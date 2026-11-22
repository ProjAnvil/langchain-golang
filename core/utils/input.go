package utils

import (
	"fmt"
	"io"
)

// smallBTextColorMapping mirrors the ANSI SGR parameter strings used by
// LangChain for named text colors. Keys are color names; values are the SGR
// parameters without the surrounding escape prefix and the trailing "m".
var smallBTextColorMapping = map[string]string{
	"blue":   "36;1",
	"yellow": "33;1",
	"pink":   "38;5;200",
	"green":  "32;1",
	"red":    "31;1",
}

// GetColorMapping maps each item to a color by rotating through the colors
// slice. The rotation is deterministic: item i is assigned
// colors[i % len(colors)]. When colors is empty the returned map is empty, and
// when items is empty the returned map is empty. Duplicate items keep the color
// assigned at their last position, which matches a map built by iteration.
func GetColorMapping(items []string, colors []string) map[string]string {
	out := make(map[string]string, len(items))
	if len(colors) == 0 {
		return out
	}
	for i, item := range items {
		out[item] = colors[i%len(colors)]
	}
	return out
}

// GetColoredText wraps text in ANSI escape sequences for the given color. A
// known color name (for example "red") is resolved to its SGR parameters; any
// other string is used directly as the SGR parameters, which lets callers pass
// raw codes such as "31;1". Unknown or empty colors are used verbatim and
// never cause a panic. The output matches LangChain's get_colored_text.
func GetColoredText(text string, color string) string {
	code, ok := smallBTextColorMapping[color]
	if !ok {
		code = color
	}
	return "\x1b[" + code + "m\x1b[1;3m" + text + "\x1b[0m"
}

// GetBoldedText wraps text in ANSI bold escape sequences.
func GetBoldedText(text string) string {
	return "\x1b[1m" + text + "\x1b[0m"
}

// PrintText writes text to writer, optionally coloring and/or bolding it. No
// trailing newline or other suffix is appended. When color is non-empty the
// text is colored via GetColoredText; when bold is true the text is wrapped via
// GetBoldedText. Bold is applied first, then color. A nil writer or a write
// failure is returned as an error; this function never panics.
func PrintText(text string, color string, bold bool, writer io.Writer) error {
	if writer == nil {
		return fmt.Errorf("print text: writer must not be nil")
	}
	out := text
	if bold {
		out = GetBoldedText(out)
	}
	if color != "" {
		out = GetColoredText(out, color)
	}
	if _, err := io.WriteString(writer, out); err != nil {
		return fmt.Errorf("print text: %w", err)
	}
	return nil
}

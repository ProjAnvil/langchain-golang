package messages

import "testing"

type markedOutput struct{}

func (markedOutput) IsToolOutput() bool { return true }

func TestToolOutputMarkerRecognition(t *testing.T) {
	var _ ToolOutput = markedOutput{}
	var out any = markedOutput{}
	if _, ok := out.(ToolOutput); !ok {
		t.Fatal("markedOutput does not satisfy messages.ToolOutput")
	}
	if _, ok := any("plain string").(ToolOutput); ok {
		t.Fatal("a plain string must not satisfy messages.ToolOutput")
	}
}

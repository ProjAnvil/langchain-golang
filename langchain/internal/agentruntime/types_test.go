package agentruntime

import (
	"errors"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/types"
)

// TestSentinelConstants verifies the shim re-exports the exact sentinel
// values of langgraph/types: the graph executor compares node names against
// langgraph/types.START/END/ParentGraph, so any drift here would silently
// break graph wiring for consumers going through this package.
func TestSentinelConstants(t *testing.T) {
	cases := []struct {
		name string
		got  string
		want string
	}{
		{"START", START, "__start__"},
		{"END", END, "__end__"},
		{"ParentGraph", ParentGraph, "__parent__"},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("%s = %q, want %q", c.name, c.got, c.want)
		}
	}

	// The shim constants must be the langgraph/types ones, not copies.
	if START != types.START || END != types.END || ParentGraph != types.ParentGraph {
		t.Errorf("shim constants diverged from langgraph/types: START=%q END=%q ParentGraph=%q",
			START, END, ParentGraph)
	}
}

// TestTypeAliasesAreIdentical verifies the declared aliases remain true
// aliases of the langgraph/types structs: values constructed through the shim
// must be assignable to the langgraph/types types (and vice versa) with no
// conversion. This fails to compile if an alias ever drifts into a defined
// type or loses a field.
func TestTypeAliasesAreIdentical(t *testing.T) {
	send := Send{Node: "node-a", Arg: map[string]any{"k": "v"}}
	var langgraphSend types.Send = send
	if langgraphSend.Node != "node-a" || langgraphSend.Arg["k"] != "v" {
		t.Errorf("Send round trip = %+v", langgraphSend)
	}

	cmd := Command{
		Graph:  ParentGraph,
		Update: map[string]any{"a": 1},
		Resume: "resume-value",
		Goto:   []any{"node-b", &send},
	}
	var langgraphCmd types.Command = cmd
	if langgraphCmd.Graph != types.ParentGraph || langgraphCmd.Resume != "resume-value" || len(langgraphCmd.Goto) != 2 {
		t.Errorf("Command round trip = %+v", langgraphCmd)
	}

	intr := Interrupt{Value: "what is your name?", ID: "int-1"}
	var langgraphIntr types.Interrupt = intr
	if langgraphIntr.Value != "what is your name?" || langgraphIntr.ID != "int-1" {
		t.Errorf("Interrupt round trip = %+v", langgraphIntr)
	}
}

// TestGraphInterrupt verifies the GraphInterrupt alias keeps its error
// behavior: it must format the interrupt value and ID, and errors.As must
// match it through both the shim and the langgraph/types spelling.
func TestGraphInterrupt(t *testing.T) {
	err := error(&GraphInterrupt{Interrupt: Interrupt{Value: "question", ID: "int-7"}})

	var shimGI *GraphInterrupt
	if !errors.As(err, &shimGI) {
		t.Fatal("errors.As did not match *agentruntime.GraphInterrupt")
	}
	var langgraphGI *types.GraphInterrupt
	if !errors.As(err, &langgraphGI) {
		t.Fatal("errors.As did not match *types.GraphInterrupt")
	}
	if shimGI != (*GraphInterrupt)(langgraphGI) {
		t.Error("shim and langgraph/types GraphInterrupt matches are not the same value")
	}

	msg := shimGI.Error()
	if !strings.Contains(msg, "question") || !strings.Contains(msg, "int-7") {
		t.Errorf("Error() = %q, want it to mention the interrupt value and ID", msg)
	}

	// A non-GraphInterrupt error must not match.
	if errors.As(errors.New("boom"), &shimGI) {
		t.Error("errors.As matched an unrelated error")
	}
}

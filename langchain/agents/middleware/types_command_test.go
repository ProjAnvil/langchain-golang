package middleware

import (
	"reflect"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

// TestNormalizeModelCallResultSurfacesCommand pins that the
// ExtendedModelResponse unwrap no longer drops the Command: the embedded
// ModelResponse is returned for the inner composition boundary AND the
// Command is surfaced so the model node can accumulate and apply it
// (mirroring factory._to_composed_result's command accumulation,
// factory.py:258-275).
func TestNormalizeModelCallResultSurfacesCommand(t *testing.T) {
	in := ModelResponse{Result: []messages.Message{messages.AI("x")}}
	cmd := &Command{Update: map[string]any{"foo": "bar"}}

	resp, gotCmd, err := NormalizeModelCallResult(ExtendedModelResponse{ModelResponse: in, Command: cmd})
	if err != nil || !reflect.DeepEqual(resp, in) || gotCmd != cmd {
		t.Fatalf("value form: resp=%#v cmd=%#v err=%v", resp, gotCmd, err)
	}

	resp, gotCmd, err = NormalizeModelCallResult(&ExtendedModelResponse{ModelResponse: in, Command: cmd})
	if err != nil || !reflect.DeepEqual(resp, in) || gotCmd != cmd {
		t.Fatalf("pointer form: resp=%#v cmd=%#v err=%v", resp, gotCmd, err)
	}

	// An ExtendedModelResponse without a Command surfaces a nil command.
	resp, gotCmd, err = NormalizeModelCallResult(ExtendedModelResponse{ModelResponse: in})
	if err != nil || !reflect.DeepEqual(resp, in) || gotCmd != nil {
		t.Fatalf("no command: resp=%#v cmd=%#v err=%v", resp, gotCmd, err)
	}

	// Non-extended results never carry a command.
	resp, gotCmd, err = NormalizeModelCallResult(messages.AI("hello"))
	if err != nil || len(resp.Result) != 1 || gotCmd != nil {
		t.Fatalf("AI short form: resp=%#v cmd=%#v err=%v", resp, gotCmd, err)
	}
	resp, gotCmd, err = NormalizeModelCallResult(in)
	if err != nil || !reflect.DeepEqual(resp, in) || gotCmd != nil {
		t.Fatalf("ModelResponse: resp=%#v cmd=%#v err=%v", resp, gotCmd, err)
	}
}

// TestValidateForWrapModelCall pins the update-only rule for Commands
// returned from wrap_model_call middleware: goto/resume/graph raise (mirroring
// factory._build_commands, factory.py:210-216) and update-only Commands pass.
func TestValidateForWrapModelCall(t *testing.T) {
	cases := []struct {
		name    string
		command Command
		wantSub string
	}{
		{"goto", Command{Goto: "tools"}, "Command goto is not supported in wrap_model_call"},
		{"resume", Command{Resume: "anything"}, "Command resume is not supported in wrap_model_call"},
		{"graph", Command{Graph: "parent"}, "Command graph is not supported in wrap_model_call"},
	}
	for _, tc := range cases {
		err := tc.command.ValidateForWrapModelCall()
		if err == nil || !strings.Contains(err.Error(), tc.wantSub) {
			t.Fatalf("%s: expected error containing %q, got %v", tc.name, tc.wantSub, err)
		}
	}
	for _, cmd := range []Command{
		{},
		{Update: map[string]any{"foo": "bar"}},
	} {
		if err := cmd.ValidateForWrapModelCall(); err != nil {
			t.Fatalf("update-only command %#v must validate, got %v", cmd, err)
		}
	}
}

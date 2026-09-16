package agents

// T16 PR2: interrupt-mode HumanInTheLoopMiddleware wired into CreateAgent as
// a dedicated "hitl" graph node between the model node and the tools node.
// The pause happens AFTER the model node's update commits, so a resume
// re-runs only the hitl node and never re-invokes the model (test 4's
// exactly-once guard). Mirrors Python's per-middleware after_model nodes.

import (
	"context"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langchain/agents/middleware"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
)

// newHitlTestTool builds an "echo"-like tool that counts its executions, so
// reject/respond tests can assert the tool never ran.
func newHitlTestTool(t *testing.T, executions *int) coretools.Tool {
	t.Helper()
	tool, err := coretools.NewSimple("echo", "echoes its input",
		func(_ context.Context, input string) (coretools.Result, error) {
			*executions++
			return coretools.Result{Content: "echo:" + input}, nil
		})
	if err != nil {
		t.Fatalf("new hitl test tool: %v", err)
	}
	return tool
}

func hitlEchoToolCall(id, input string) messages.ToolCall {
	return messages.ToolCall{ID: id, Name: "echo", Args: map[string]any{"tool_input": input}}
}

// TestCreateAgentHITLPauseShape (design §6.4): with a checkpointer, a
// reviewable tool call pauses the run as exactly one interrupt whose value
// decodes to the HITLRequest; the pending AI message is already committed at
// the pause; and the model has been invoked EXACTLY ONCE (the gatekeeping
// assertion: pausing must not replay the model node on resume).
func TestCreateAgentHITLPauseShape(t *testing.T) {
	var toolRuns int
	model := &sequenceModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{hitlEchoToolCall("call_1", "hi")}},
		messages.AI("done"),
	}}
	saver := checkpoint.NewMemorySaver()

	agent, err := CreateAgent(model, []coretools.Tool{newHitlTestTool(t, &toolRuns)},
		WithAgentMiddleware(middleware.NewInterruptHumanInTheLoopMiddleware(map[string]middleware.InterruptConfig{
			"echo": {AllowedDecisions: []middleware.DecisionType{middleware.DecisionApprove, middleware.DecisionReject}},
		})),
		WithAgentCheckpointer(saver),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	values, interrupts, err := agent.InvokeWithStateOptions(t.Context(),
		[]messages.Message{messages.Human("hi")}, graphpkg.Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("first invoke: %v", err)
	}
	if len(interrupts) != 1 {
		t.Fatalf("expected exactly one pending interrupt, got %+v", interrupts)
	}
	intr := interrupts[0]
	if !strings.HasPrefix(intr.ID, "hitl-") {
		t.Fatalf("expected a hitl-node interrupt ID, got %q", intr.ID)
	}
	if !strings.HasPrefix(intr.NS, "hitl:") {
		t.Fatalf("expected a hitl-node interrupt NS, got %q", intr.NS)
	}
	request, err := middleware.HITLRequestFromInterrupt(intr)
	if err != nil {
		t.Fatalf("decode interrupt value: %v", err)
	}
	if len(request.ActionRequests) != 1 || request.ActionRequests[0].Name != "echo" {
		t.Fatalf("action requests mismatch: %#v", request.ActionRequests)
	}
	if request.ActionRequests[0].Args["tool_input"] != "hi" {
		t.Fatalf("action request args mismatch: %#v", request.ActionRequests[0])
	}
	if !strings.Contains(request.ActionRequests[0].Description, "Tool execution requires approval") ||
		!strings.Contains(request.ActionRequests[0].Description, "Tool: echo") {
		t.Fatalf("action request description mismatch: %q", request.ActionRequests[0].Description)
	}
	if len(request.ReviewConfigs) != 1 ||
		len(request.ReviewConfigs[0].AllowedDecisions) != 2 ||
		request.ReviewConfigs[0].ActionName != "echo" {
		t.Fatalf("review configs mismatch: %#v", request.ReviewConfigs)
	}

	// The pending AI message is committed at the pause (the model node's
	// superstep finished before the hitl node paused).
	pausedMsgs, _ := values["messages"].([]messages.Message)
	if len(pausedMsgs) != 2 || pausedMsgs[1].Role != messages.RoleAI || len(pausedMsgs[1].ToolCalls) != 1 {
		t.Fatalf("expected human+AI(tool_call) committed at pause, got %#v", pausedMsgs)
	}
	// GATEKEEPER: the model ran exactly once for the paused call.
	if len(model.invocations) != 1 {
		t.Fatalf("expected model invoked exactly once at pause, got %d", len(model.invocations))
	}
	if toolRuns != 0 {
		t.Fatalf("tool must not run before approval, ran %d times", toolRuns)
	}

	// Approve through Agent.Resume: only the hitl node re-runs, the tool
	// executes, and the model's SECOND call is a fresh one after the tool.
	values, interrupts, err = agent.Resume(t.Context(), graphpkg.Options{
		ThreadID: "t1",
		Resume:   middleware.HITLResponse{Decisions: []middleware.Decision{{Type: middleware.DecisionApprove}}},
	})
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	if len(interrupts) != 0 {
		t.Fatalf("expected no interrupts after resume, got %+v", interrupts)
	}
	if len(model.invocations) != 2 {
		t.Fatalf("expected exactly two model invocations total (no replay of the first), got %d", len(model.invocations))
	}
	if toolRuns != 1 {
		t.Fatalf("expected tool to run once after approval, ran %d times", toolRuns)
	}
	out, _ := values["messages"].([]messages.Message)
	if len(out) != 4 || out[2].Role != messages.RoleTool || out[2].Content != "echo:hi" || out[3].Content != "done" {
		t.Fatalf("final messages mismatch: %#v", out)
	}
}

// TestCreateAgentHitlMintedIDsUniqueForIdenticalModelCalls pins the monotonic
// factor of mintAIMessageIDs: two model-node executions whose local prompt is
// byte-identical and whose model output content is byte-identical (two fresh
// threads, same input — also what keep-last-K history trimming produces
// in-thread once the trimmed window repeats) must still mint DISTINCT
// AI-message IDs. With the pre-monotonic sha256(prompt+content) hash the two
// IDs collided and MessagesReducer's ID match silently REPLACED the earlier
// message instead of appending.
func TestCreateAgentHitlMintedIDsUniqueForIdenticalModelCalls(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{
		messages.AI("same answer"),
		messages.AI("same answer"),
	}}
	saver := checkpoint.NewMemorySaver()
	agent, err := CreateAgent(model, []coretools.Tool{newHitlTestTool(t, new(int))},
		WithAgentMiddleware(middleware.NewInterruptHumanInTheLoopMiddleware(map[string]middleware.InterruptConfig{
			"echo": {AllowedDecisions: []middleware.DecisionType{middleware.DecisionApprove}},
		})),
		WithAgentCheckpointer(saver),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	first, _, err := agent.InvokeWithStateOptions(t.Context(),
		[]messages.Message{messages.Human("hi")}, graphpkg.Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("first invoke: %v", err)
	}
	second, _, err := agent.InvokeWithStateOptions(t.Context(),
		[]messages.Message{messages.Human("hi")}, graphpkg.Options{ThreadID: "t2"})
	if err != nil {
		t.Fatalf("second invoke: %v", err)
	}
	firstMsgs, _ := first["messages"].([]messages.Message)
	secondMsgs, _ := second["messages"].([]messages.Message)
	if len(firstMsgs) != 2 || len(secondMsgs) != 2 {
		t.Fatalf("message counts: first=%d second=%d", len(firstMsgs), len(secondMsgs))
	}
	firstID, secondID := firstMsgs[1].ID, secondMsgs[1].ID
	if firstID == "" || secondID == "" {
		t.Fatalf("hitl-wired agent must mint AI ids: first=%q second=%q", firstID, secondID)
	}
	if firstID == secondID {
		t.Fatalf("identical (prompt, content) model calls minted the same id %q; "+
			"a monotonic factor must keep distinct model-node outputs distinct", firstID)
	}
}

// TestMintAIMessageIDsMonotonicFactor is the unit-level pin: the mint must
// never produce the same ID for two distinct messages, whether across calls
// with identical (prompt, content) or within one response whose messages
// repeat content.
func TestMintAIMessageIDsMonotonicFactor(t *testing.T) {
	first := []messages.Message{messages.AI("dup")}
	second := []messages.Message{messages.AI("dup")}
	mintAIMessageIDs(first, "same prompt")
	mintAIMessageIDs(second, "same prompt")
	if first[0].ID == "" || first[0].ID == second[0].ID {
		t.Fatalf("ids must differ across identical calls: %q vs %q", first[0].ID, second[0].ID)
	}
	twin := []messages.Message{messages.AI("twin"), messages.AI("twin")}
	mintAIMessageIDs(twin, "p")
	if twin[0].ID == "" || twin[0].ID == twin[1].ID {
		t.Fatalf("ids must differ within one response: %q vs %q", twin[0].ID, twin[1].ID)
	}
}

// TestCreateAgentHITLDecisionBranches (design §6.5): approve, edit, reject,
// and respond each applied to the committed AI message on resume.
func TestCreateAgentHITLDecisionBranches(t *testing.T) {
	t.Run("edit rewrites args keeping the call id", func(t *testing.T) {
		var toolRuns int
		model := &sequenceModel{responses: []messages.Message{
			{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{hitlEchoToolCall("call_1", "original")}},
			messages.AI("done"),
		}}
		saver := checkpoint.NewMemorySaver()
		agent, err := CreateAgent(model, []coretools.Tool{newHitlTestTool(t, &toolRuns)},
			WithAgentMiddleware(middleware.NewInterruptHumanInTheLoopMiddleware(map[string]middleware.InterruptConfig{
				"echo": {AllowedDecisions: []middleware.DecisionType{middleware.DecisionEdit}},
			})),
			WithAgentCheckpointer(saver),
		)
		if err != nil {
			t.Fatalf("create agent: %v", err)
		}
		_, interrupts, err := agent.InvokeWithStateOptions(t.Context(),
			[]messages.Message{messages.Human("hi")}, graphpkg.Options{ThreadID: "t1"})
		if err != nil || len(interrupts) != 1 {
			t.Fatalf("first invoke: %v interrupts=%+v", err, interrupts)
		}
		values, _, err := agent.Resume(t.Context(), graphpkg.Options{
			ThreadID: "t1",
			Resume: middleware.HITLResponse{Decisions: []middleware.Decision{{
				Type:         middleware.DecisionEdit,
				EditedAction: &middleware.ToolCall{Name: "echo", Args: map[string]any{"tool_input": "edited"}},
			}}},
		})
		if err != nil {
			t.Fatalf("resume: %v", err)
		}
		out, _ := values["messages"].([]messages.Message)
		if len(out) != 4 {
			t.Fatalf("expected 4 final messages, got %d: %#v", len(out), out)
		}
		if out[1].ToolCalls[0].ID != "call_1" {
			t.Fatalf("edit must keep the tool call id, got %q", out[1].ToolCalls[0].ID)
		}
		if out[1].ToolCalls[0].Args["tool_input"] != "edited" {
			t.Fatalf("edit must rewrite the args, got %#v", out[1].ToolCalls[0].Args)
		}
		if out[2].Content != "echo:edited" {
			t.Fatalf("expected the edited call to execute, got %#v", out[2])
		}
		if toolRuns != 1 || len(model.invocations) != 2 {
			t.Fatalf("expected one tool run and two model calls, got %d/%d", toolRuns, len(model.invocations))
		}
	})

	t.Run("reject answers with an error tool message the model sees", func(t *testing.T) {
		var toolRuns int
		model := &sequenceModel{responses: []messages.Message{
			{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{hitlEchoToolCall("call_1", "hi")}},
			messages.AI("understood, not retrying"),
		}}
		saver := checkpoint.NewMemorySaver()
		agent, err := CreateAgent(model, []coretools.Tool{newHitlTestTool(t, &toolRuns)},
			WithAgentMiddleware(middleware.NewInterruptHumanInTheLoopMiddleware(map[string]middleware.InterruptConfig{
				"echo": {AllowedDecisions: []middleware.DecisionType{middleware.DecisionReject}},
			})),
			WithAgentCheckpointer(saver),
		)
		if err != nil {
			t.Fatalf("create agent: %v", err)
		}
		_, interrupts, err := agent.InvokeWithStateOptions(t.Context(),
			[]messages.Message{messages.Human("hi")}, graphpkg.Options{ThreadID: "t1"})
		if err != nil || len(interrupts) != 1 {
			t.Fatalf("first invoke: %v interrupts=%+v", err, interrupts)
		}
		values, _, err := agent.Resume(t.Context(), graphpkg.Options{
			ThreadID: "t1",
			Resume:   middleware.HITLResponse{Decisions: []middleware.Decision{{Type: middleware.DecisionReject, Message: "not allowed"}}},
		})
		if err != nil {
			t.Fatalf("resume: %v", err)
		}
		if toolRuns != 0 {
			t.Fatalf("rejected call must not execute, ran %d times", toolRuns)
		}
		out, _ := values["messages"].([]messages.Message)
		if len(out) != 4 || out[2].Role != messages.RoleTool {
			t.Fatalf("expected 4 final messages with a tool answer, got %#v", out)
		}
		if out[2].Content != "not allowed" || out[2].ResponseMetadata["status"] != "error" || out[2].ToolCallID != "call_1" {
			t.Fatalf("rejection tool message mismatch: %#v", out[2])
		}
		// The second model call saw the rejection copy in its input.
		secondInput := model.invocations[1]
		sawRejection := false
		for _, m := range secondInput {
			if m.Role == messages.RoleTool && strings.Contains(m.Content, "not allowed") {
				sawRejection = true
			}
		}
		if !sawRejection {
			t.Fatalf("second model call did not see the rejection: %#v", secondInput)
		}
		if out[3].Content != "understood, not retrying" {
			t.Fatalf("final answer mismatch: %#v", out[3])
		}
	})

	t.Run("respond answers on behalf of the tool", func(t *testing.T) {
		var toolRuns int
		model := &sequenceModel{responses: []messages.Message{
			{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{hitlEchoToolCall("call_1", "hi")}},
			messages.AI("done"),
		}}
		saver := checkpoint.NewMemorySaver()
		agent, err := CreateAgent(model, []coretools.Tool{newHitlTestTool(t, &toolRuns)},
			WithAgentMiddleware(middleware.NewInterruptHumanInTheLoopMiddleware(map[string]middleware.InterruptConfig{
				"echo": {AllowedDecisions: []middleware.DecisionType{middleware.DecisionRespond}},
			})),
			WithAgentCheckpointer(saver),
		)
		if err != nil {
			t.Fatalf("create agent: %v", err)
		}
		_, interrupts, err := agent.InvokeWithStateOptions(t.Context(),
			[]messages.Message{messages.Human("hi")}, graphpkg.Options{ThreadID: "t1"})
		if err != nil || len(interrupts) != 1 {
			t.Fatalf("first invoke: %v interrupts=%+v", err, interrupts)
		}
		values, _, err := agent.Resume(t.Context(), graphpkg.Options{
			ThreadID: "t1",
			Resume:   middleware.HITLResponse{Decisions: []middleware.Decision{{Type: middleware.DecisionRespond, Message: "human answer"}}},
		})
		if err != nil {
			t.Fatalf("resume: %v", err)
		}
		if toolRuns != 0 {
			t.Fatalf("responded call must not execute, ran %d times", toolRuns)
		}
		out, _ := values["messages"].([]messages.Message)
		if len(out) != 4 || out[2].Role != messages.RoleTool || out[2].Content != "human answer" ||
			out[2].ResponseMetadata["status"] != "success" {
			t.Fatalf("respond tool message mismatch: %#v", out)
		}
		if len(model.invocations) != 2 {
			t.Fatalf("expected the model to be re-consulted after respond, got %d calls", len(model.invocations))
		}
	})
}

// TestCreateAgentHITLMixedBatch (design §6.6): two reviewable calls in one
// AI message surface as ONE interrupt; per-call decisions apply in order
// (first approved and executed, second rejected and error-answered), and the
// rejected call is not re-executed by the tools node.
func TestCreateAgentHITLMixedBatch(t *testing.T) {
	var toolRuns int
	model := &sequenceModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
			hitlEchoToolCall("call_1", "a"),
			hitlEchoToolCall("call_2", "b"),
		}},
		messages.AI("done"),
	}}
	saver := checkpoint.NewMemorySaver()
	agent, err := CreateAgent(model, []coretools.Tool{newHitlTestTool(t, &toolRuns)},
		WithAgentMiddleware(middleware.NewInterruptHumanInTheLoopMiddleware(map[string]middleware.InterruptConfig{
			"echo": {AllowedDecisions: []middleware.DecisionType{middleware.DecisionApprove, middleware.DecisionReject}},
		})),
		WithAgentCheckpointer(saver),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	_, interrupts, err := agent.InvokeWithStateOptions(t.Context(),
		[]messages.Message{messages.Human("hi")}, graphpkg.Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("first invoke: %v", err)
	}
	if len(interrupts) != 1 {
		t.Fatalf("expected a single interrupt for the batch, got %+v", interrupts)
	}
	request, err := middleware.HITLRequestFromInterrupt(interrupts[0])
	if err != nil {
		t.Fatalf("decode interrupt: %v", err)
	}
	if len(request.ActionRequests) != 2 {
		t.Fatalf("expected both calls in the request, got %#v", request.ActionRequests)
	}
	values, _, err := agent.Resume(t.Context(), graphpkg.Options{
		ThreadID: "t1",
		Resume: middleware.HITLResponse{Decisions: []middleware.Decision{
			{Type: middleware.DecisionApprove},
			{Type: middleware.DecisionReject, Message: "second denied"},
		}},
	})
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	if toolRuns != 1 {
		t.Fatalf("expected exactly one tool execution (approved call only), ran %d times", toolRuns)
	}
	out, _ := values["messages"].([]messages.Message)
	// human, AI(2 calls), tool(error: second denied — the artificial answer
	// commits when the hitl node runs), tool(echo:a), AI(done). The artificial
	// message precedes the executed result because the hitl node's update
	// lands before the tools node's, matching Python's add_messages ordering.
	if len(out) != 5 {
		t.Fatalf("expected 5 final messages, got %d: %#v", len(out), out)
	}
	if out[2].Content != "second denied" || out[2].ResponseMetadata["status"] != "error" || out[2].ToolCallID != "call_2" {
		t.Fatalf("rejected call answer mismatch: %#v", out[2])
	}
	if out[3].Content != "echo:a" || out[3].ToolCallID != "call_1" {
		t.Fatalf("approved call result mismatch: %#v", out[3])
	}
	if len(model.invocations) != 2 {
		t.Fatalf("expected two model calls total, got %d", len(model.invocations))
	}
}

// TestCreateAgentHITLNilResumeRepauses (design §6.7): resuming without a
// value re-runs the hitl node, whose Interrupt() re-fires with the same ID.
func TestCreateAgentHITLNilResumeRepauses(t *testing.T) {
	var toolRuns int
	model := &sequenceModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{hitlEchoToolCall("call_1", "hi")}},
		messages.AI("done"),
	}}
	saver := checkpoint.NewMemorySaver()
	agent, err := CreateAgent(model, []coretools.Tool{newHitlTestTool(t, &toolRuns)},
		WithAgentMiddleware(middleware.NewInterruptHumanInTheLoopMiddleware(map[string]middleware.InterruptConfig{
			"echo": {AllowedDecisions: []middleware.DecisionType{middleware.DecisionApprove}},
		})),
		WithAgentCheckpointer(saver),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	_, interrupts, err := agent.InvokeWithStateOptions(t.Context(),
		[]messages.Message{messages.Human("hi")}, graphpkg.Options{ThreadID: "t1"})
	if err != nil || len(interrupts) != 1 {
		t.Fatalf("first invoke: %v interrupts=%+v", err, interrupts)
	}
	firstID := interrupts[0].ID

	values, interrupts, err := agent.Resume(t.Context(), graphpkg.Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("nil resume: %v", err)
	}
	if len(interrupts) != 1 || interrupts[0].ID != firstID {
		t.Fatalf("expected re-pause with the same interrupt id %q, got %+v", firstID, interrupts)
	}
	if toolRuns != 0 || len(model.invocations) != 1 {
		t.Fatalf("nil resume must not run tools or the model, got %d/%d", toolRuns, len(model.invocations))
	}
	// The state is unchanged apart from the re-pause.
	msgs, _ := values["messages"].([]messages.Message)
	if len(msgs) != 2 {
		t.Fatalf("expected the paused state to be unchanged, got %#v", msgs)
	}

	// A subsequent valued resume still completes.
	_, interrupts, err = agent.Resume(t.Context(), graphpkg.Options{
		ThreadID: "t1",
		Resume:   middleware.HITLResponse{Decisions: []middleware.Decision{{Type: middleware.DecisionApprove}}},
	})
	if err != nil || len(interrupts) != 0 {
		t.Fatalf("valued resume after re-pause: %v interrupts=%+v", err, interrupts)
	}
	if toolRuns != 1 || len(model.invocations) != 2 {
		t.Fatalf("expected completion after valued resume, got %d/%d", toolRuns, len(model.invocations))
	}
}

// TestCreateAgentHITLResumeValidation (design §6.8): a resume whose decision
// count does not match the interrupted calls, or whose decision type is not
// allowed for the tool, errors the run.
func TestCreateAgentHITLResumeValidation(t *testing.T) {
	newAgent := func(t *testing.T) *Agent {
		var toolRuns int
		model := &sequenceModel{responses: []messages.Message{
			{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{hitlEchoToolCall("call_1", "hi")}},
			messages.AI("done"), // unreachable on the error paths
			messages.AI("done"),
		}}
		saver := checkpoint.NewMemorySaver()
		agent, err := CreateAgent(model, []coretools.Tool{newHitlTestTool(t, &toolRuns)},
			WithAgentMiddleware(middleware.NewInterruptHumanInTheLoopMiddleware(map[string]middleware.InterruptConfig{
				"echo": {AllowedDecisions: []middleware.DecisionType{middleware.DecisionApprove}},
			})),
			WithAgentCheckpointer(saver),
		)
		if err != nil {
			t.Fatalf("create agent: %v", err)
		}
		return agent
	}

	t.Run("decision count mismatch", func(t *testing.T) {
		agent := newAgent(t)
		_, interrupts, err := agent.InvokeWithStateOptions(t.Context(),
			[]messages.Message{messages.Human("hi")}, graphpkg.Options{ThreadID: "t1"})
		if err != nil || len(interrupts) != 1 {
			t.Fatalf("first invoke: %v interrupts=%+v", err, interrupts)
		}
		_, _, err = agent.Resume(t.Context(), graphpkg.Options{
			ThreadID: "t1",
			Resume:   middleware.HITLResponse{}, // zero decisions for one interrupted call
		})
		if err == nil || !strings.Contains(err.Error(), "does not match") {
			t.Fatalf("expected count mismatch error, got %v", err)
		}
	})

	t.Run("disallowed decision type", func(t *testing.T) {
		agent := newAgent(t)
		_, interrupts, err := agent.InvokeWithStateOptions(t.Context(),
			[]messages.Message{messages.Human("hi")}, graphpkg.Options{ThreadID: "t1"})
		if err != nil || len(interrupts) != 1 {
			t.Fatalf("first invoke: %v interrupts=%+v", err, interrupts)
		}
		_, _, err = agent.Resume(t.Context(), graphpkg.Options{
			ThreadID: "t1",
			Resume:   middleware.HITLResponse{Decisions: []middleware.Decision{{Type: middleware.DecisionReject}}},
		})
		if err == nil || !strings.Contains(err.Error(), "is not allowed") {
			t.Fatalf("expected not-allowed error, got %v", err)
		}
	})

	t.Run("undecodable resume value", func(t *testing.T) {
		agent := newAgent(t)
		_, interrupts, err := agent.InvokeWithStateOptions(t.Context(),
			[]messages.Message{messages.Human("hi")}, graphpkg.Options{ThreadID: "t1"})
		if err != nil || len(interrupts) != 1 {
			t.Fatalf("first invoke: %v interrupts=%+v", err, interrupts)
		}
		_, _, err = agent.Resume(t.Context(), graphpkg.Options{ThreadID: "t1", Resume: "yes"})
		if err == nil || !strings.Contains(err.Error(), "cannot decode HITL response") {
			t.Fatalf("expected decode error, got %v", err)
		}
	})
}

// TestCreateAgentHITLMintsAIID (design §6.9): an ID-less AI message gets a
// deterministic ID minted when the hitl node is wired, so the revised AI
// message replaces the committed one by ID instead of duplicating it.
func TestCreateAgentHITLMintsAIID(t *testing.T) {
	var toolRuns int
	model := &sequenceModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{{Name: "echo", Args: map[string]any{"tool_input": "hi"}}}}, // no ID
		messages.AI("done"),
	}}
	saver := checkpoint.NewMemorySaver()
	agent, err := CreateAgent(model, []coretools.Tool{newHitlTestTool(t, &toolRuns)},
		WithAgentMiddleware(middleware.NewInterruptHumanInTheLoopMiddleware(map[string]middleware.InterruptConfig{
			"echo": {AllowedDecisions: []middleware.DecisionType{middleware.DecisionReject}},
		})),
		WithAgentCheckpointer(saver),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	values, interrupts, err := agent.InvokeWithStateOptions(t.Context(),
		[]messages.Message{messages.Human("hi")}, graphpkg.Options{ThreadID: "t1"})
	if err != nil || len(interrupts) != 1 {
		t.Fatalf("first invoke: %v interrupts=%+v", err, interrupts)
	}
	pausedMsgs, _ := values["messages"].([]messages.Message)
	if len(pausedMsgs) != 2 || pausedMsgs[1].ID == "" {
		t.Fatalf("expected the committed AI message to carry a minted id, got %#v", pausedMsgs)
	}
	mintedID := pausedMsgs[1].ID

	values, _, err = agent.Resume(t.Context(), graphpkg.Options{
		ThreadID: "t1",
		Resume:   middleware.HITLResponse{Decisions: []middleware.Decision{{Type: middleware.DecisionReject}}},
	})
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	out, _ := values["messages"].([]messages.Message)
	// human, AI(revised, same minted id), tool(rejection), AI(done) — exactly
	// one AI message with tool calls: no duplicate from the in-place revision.
	aiWithCalls := 0
	for _, m := range out {
		if m.Role == messages.RoleAI && len(m.ToolCalls) > 0 {
			aiWithCalls++
			if m.ID != mintedID {
				t.Fatalf("revised AI message id = %q, want the minted %q", m.ID, mintedID)
			}
		}
	}
	if aiWithCalls != 1 || len(out) != 4 {
		t.Fatalf("expected no duplicate AI message (1 with calls, 4 total), got %d/%d: %#v", aiWithCalls, len(out), out)
	}
	if toolRuns != 0 || len(model.invocations) != 2 {
		t.Fatalf("expected no tool run and two model calls, got %d/%d", toolRuns, len(model.invocations))
	}
}

// TestCreateAgentHITLReturnDirectAndStructured (design §6.10): a return_direct
// tool approved through HITL still ends the run after the tools node; a
// structured-output tool call never reaches the HITL review (the detection
// ends the run in the model node, before the after_model/hitl ordering).
func TestCreateAgentHITLReturnDirectAndStructured(t *testing.T) {
	t.Run("return_direct ends after approved tools", func(t *testing.T) {
		model := &sequenceModel{responses: []messages.Message{
			{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{{
				ID: "call_1", Name: "direct_echo", Args: map[string]any{"tool_input": "hi"},
			}}},
		}}
		saver := checkpoint.NewMemorySaver()
		agent, err := CreateAgent(model, []coretools.Tool{newDirectEchoTool(t)},
			WithAgentMiddleware(middleware.NewInterruptHumanInTheLoopMiddleware(map[string]middleware.InterruptConfig{
				"direct_echo": {AllowedDecisions: []middleware.DecisionType{middleware.DecisionApprove}},
			})),
			WithAgentCheckpointer(saver),
		)
		if err != nil {
			t.Fatalf("create agent: %v", err)
		}
		_, interrupts, err := agent.InvokeWithStateOptions(t.Context(),
			[]messages.Message{messages.Human("hi")}, graphpkg.Options{ThreadID: "t1"})
		if err != nil || len(interrupts) != 1 {
			t.Fatalf("first invoke: %v interrupts=%+v", err, interrupts)
		}
		values, interrupts, err := agent.Resume(t.Context(), graphpkg.Options{
			ThreadID: "t1",
			Resume:   middleware.HITLResponse{Decisions: []middleware.Decision{{Type: middleware.DecisionApprove}}},
		})
		if err != nil {
			t.Fatalf("resume: %v", err)
		}
		if len(interrupts) != 0 {
			t.Fatalf("expected completion, got %+v", interrupts)
		}
		if len(model.invocations) != 1 {
			t.Fatalf("return_direct must skip the second model call, got %d", len(model.invocations))
		}
		out, _ := values["messages"].([]messages.Message)
		if len(out) != 3 || out[2].Role != messages.RoleTool || out[2].Content != "direct:hi" {
			t.Fatalf("expected the run to end on the tool result, got %#v", out)
		}
	})

	t.Run("structured call bypasses hitl", func(t *testing.T) {
		strategy := NewToolStrategy(answerSchema())
		structuredTool := strategy.SchemaSpecs[0].Name
		var toolRuns int
		model := &sequenceModel{responses: []messages.Message{
			{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{{
				ID: "call_1", Name: structuredTool, Args: map[string]any{"text": "42"},
			}}},
		}}
		saver := checkpoint.NewMemorySaver()
		agent, err := CreateAgent(model, []coretools.Tool{newHitlTestTool(t, &toolRuns)},
			WithAgentResponseFormat(strategy),
			WithAgentMiddleware(middleware.NewInterruptHumanInTheLoopMiddleware(map[string]middleware.InterruptConfig{
				"echo": {AllowedDecisions: []middleware.DecisionType{middleware.DecisionApprove}},
			})),
			WithAgentCheckpointer(saver),
		)
		if err != nil {
			t.Fatalf("create agent: %v", err)
		}
		values, interrupts, err := agent.InvokeWithStateOptions(t.Context(),
			[]messages.Message{messages.Human("the answer?")}, graphpkg.Options{ThreadID: "t1"})
		if err != nil {
			t.Fatalf("invoke: %v", err)
		}
		if len(interrupts) != 0 {
			t.Fatalf("structured call must not pause for HITL, got %+v", interrupts)
		}
		structured, ok := values["structured_response"].(map[string]any)
		if !ok || structured["text"] != "42" {
			t.Fatalf("structured response mismatch: %#v", values["structured_response"])
		}
	})
}

// TestCreateAgentHITLWithInterruptBeforeTools (design §6.11): the hitl pause
// and a WithAgentInterruptBefore(ToolsNodeName) boundary interrupt compose —
// the HITL decision resumes first, then the boundary pause fires before the
// tools node, and a nil-input resume completes the run.
func TestCreateAgentHITLWithInterruptBeforeTools(t *testing.T) {
	var toolRuns int
	model := &sequenceModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{hitlEchoToolCall("call_1", "hi")}},
		messages.AI("done"),
	}}
	saver := checkpoint.NewMemorySaver()
	agent, err := CreateAgent(model, []coretools.Tool{newHitlTestTool(t, &toolRuns)},
		WithAgentMiddleware(middleware.NewInterruptHumanInTheLoopMiddleware(map[string]middleware.InterruptConfig{
			"echo": {AllowedDecisions: []middleware.DecisionType{middleware.DecisionApprove}},
		})),
		WithAgentCheckpointer(saver),
		WithAgentInterruptBefore(ToolsNodeName),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	_, interrupts, err := agent.InvokeWithStateOptions(t.Context(),
		[]messages.Message{messages.Human("hi")}, graphpkg.Options{ThreadID: "t1"})
	if err != nil || len(interrupts) != 1 || !strings.HasPrefix(interrupts[0].ID, "hitl-") {
		t.Fatalf("first invoke: %v interrupts=%+v", err, interrupts)
	}

	// Approve: the hitl node completes, then the run pauses AGAIN before the
	// tools node dispatches.
	_, interrupts, err = agent.Resume(t.Context(), graphpkg.Options{
		ThreadID: "t1",
		Resume:   middleware.HITLResponse{Decisions: []middleware.Decision{{Type: middleware.DecisionApprove}}},
	})
	if err != nil {
		t.Fatalf("approve resume: %v", err)
	}
	if len(interrupts) != 1 {
		t.Fatalf("expected the boundary interrupt before tools, got %+v", interrupts)
	}
	if !strings.HasPrefix(interrupts[0].ID, "interrupt-before-"+ToolsNodeName) {
		t.Fatalf("expected an interrupt-before-tools id, got %q", interrupts[0].ID)
	}
	if toolRuns != 0 {
		t.Fatalf("tool must not run before the boundary resume, ran %d times", toolRuns)
	}

	// Boundary interrupts resume with a nil value (no in-node interrupt to
	// answer).
	values, interrupts, err := agent.Resume(t.Context(), graphpkg.Options{ThreadID: "t1"})
	if err != nil || len(interrupts) != 0 {
		t.Fatalf("boundary resume: %v interrupts=%+v", err, interrupts)
	}
	if toolRuns != 1 || len(model.invocations) != 2 {
		t.Fatalf("expected completion, got %d tool runs / %d model calls", toolRuns, len(model.invocations))
	}
	out, _ := values["messages"].([]messages.Message)
	if len(out) != 4 || out[3].Content != "done" {
		t.Fatalf("final messages mismatch: %#v", out)
	}
}

// namedHitl wraps an interrupt-mode HITL middleware under a distinct
// middleware name, so two of them bypass the same-type duplicate-name check
// and reach the dedicated single-interrupt-HITL validation.
type namedHitl struct {
	*middleware.HumanInTheLoopMiddleware
	name string
}

func (n namedHitl) Name() string { return n.name }

// TestCreateAgentHITLMultipleInterruptMiddlewareRejected: a second
// interrupt-mode HITL middleware is rejected at build time (both would share
// the single fixed-name hitl node).
func TestCreateAgentHITLMultipleInterruptMiddlewareRejected(t *testing.T) {
	hitl := func() *middleware.HumanInTheLoopMiddleware {
		return middleware.NewInterruptHumanInTheLoopMiddleware(map[string]middleware.InterruptConfig{
			"echo": {AllowedDecisions: []middleware.DecisionType{middleware.DecisionApprove}},
		})
	}
	_, err := CreateAgent(&sequenceModel{}, []coretools.Tool{newEchoTool(t)},
		WithAgentMiddleware(
			namedHitl{HumanInTheLoopMiddleware: hitl(), name: "hitl-a"},
			namedHitl{HumanInTheLoopMiddleware: hitl(), name: "hitl-b"},
		),
	)
	if err == nil || !strings.Contains(err.Error(), "human-in-the-loop") {
		t.Fatalf("expected multiple interrupt-mode HITL rejection, got %v", err)
	}
	// Two instances of the same type are still caught by the generic
	// duplicate-middleware-name validation first — also a rejection.
	if _, err := CreateAgent(&sequenceModel{}, []coretools.Tool{newEchoTool(t)},
		WithAgentMiddleware(hitl(), hitl()),
	); err == nil || !strings.Contains(err.Error(), "duplicate middleware") {
		t.Fatalf("expected duplicate-middleware rejection, got %v", err)
	}

	// One interrupt-mode HITL alongside a Decide-mode HITL stays legal: the
	// Decide-mode middleware runs inline in the model node. (Both are the
	// same Go type, so the generic duplicate-name validation requires distinct
	// names — unrelated to the HITL single-node rule under test.)
	decide := middleware.NewHumanInTheLoopMiddleware(map[string]middleware.InterruptConfig{
		"echo": {AllowedDecisions: []middleware.DecisionType{middleware.DecisionApprove}},
	}, func(middleware.HITLRequest) ([]middleware.Decision, error) {
		return []middleware.Decision{{Type: middleware.DecisionApprove}}, nil
	})
	if _, err := CreateAgent(&sequenceModel{}, []coretools.Tool{newEchoTool(t)},
		WithAgentMiddleware(hitl(), namedHitl{HumanInTheLoopMiddleware: decide, name: "hitl-decide"}),
	); err != nil {
		t.Fatalf("interrupt + decide HITL must compose: %v", err)
	}
}

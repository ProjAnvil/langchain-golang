package agents

// T16 PR3 (design §6.12-§6.14): interrupt-mode HITL under nesting and across
// "processes".
//
//   - §6.12: an agent embedded as a subgraph (StateGraph.AddSubgraph) inside a
//     checkpointing parent — the supervisor shape — pauses the PARENT with the
//     child's hitl interrupt carrying a nested NS, and a parent-side
//     Options.Resume map addressed by that NS completes the child and merges
//     its values into the parent.
//   - §6.13: the same pause resumed through Options.Graph scoping (the child
//     task's namespace), including the error when the named graph matches no
//     pending interrupt.
//   - §6.14: a cross-process simulation — the paused thread's checkpoints
//     (including the ReservedInterrupt write carrying the HITLRequest) are
//     serialized through the JSON serde (the same typed split the sqlite/
//     postgres savers persist) and reloaded into a fresh MemorySaver; a
//     freshly built agent over that saver resumes to completion, and the
//     round-tripped interrupt value decodes through the map wire form.

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/messages"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langchain/agents/middleware"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/serde"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// newHITLWorkerAgent builds a child agent whose single "echo" tool call must
// pass an interrupt-mode HITL review before it executes. The child carries no
// checkpointer of its own: embedded via AddSubgraph it shares the PARENT
// run's checkpointer and thread, mirroring a supervisor deployment where only
// the parent persists.
func newHITLWorkerAgent(t *testing.T, toolRuns *int, model *sequenceModel) *Agent {
	t.Helper()
	agent, err := CreateAgent(model, []coretools.Tool{newHitlTestTool(t, toolRuns)},
		WithAgentMiddleware(middleware.NewInterruptHumanInTheLoopMiddleware(map[string]middleware.InterruptConfig{
			"echo": {AllowedDecisions: []middleware.DecisionType{middleware.DecisionApprove}},
		})),
	)
	if err != nil {
		t.Fatalf("create worker agent: %v", err)
	}
	return agent
}

// newSupervisorGraph embeds worker as the "worker" subgraph node, followed by
// a "report" node whose value proves the parent continued past the subgraph
// with the child's final values merged in.
func newSupervisorGraph(t *testing.T, worker *Agent, saver checkpoint.Saver, reportRuns *int) *graphpkg.CompiledGraph {
	t.Helper()
	parent := graphpkg.NewStateGraph()
	parent.AddSubgraph("worker", worker.Graph)
	parent.AddNode("report", func(_ runtime.Runtime, state map[string]any) (any, error) {
		*reportRuns++
		msgs, _ := state["messages"].([]messages.Message)
		return map[string]any{"report": fmt.Sprintf("%d messages", len(msgs))}, nil
	})
	parent.AddEdge(types.START, "worker")
	parent.AddEdge("worker", "report")
	parent.AddEdge("report", types.END)
	cg, err := parent.Compile(graphpkg.WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("compile supervisor graph: %v", err)
	}
	return cg
}

// TestCreateAgentHITLNestedSupervisorResume (design §6.12): the child agent's
// hitl pause surfaces on the PARENT run's Result.Interrupts with a nested NS
// ("<subgraph>:<task>/hitl:<task>"); a parent-side Options.Resume map keyed by
// that NS forwards the decision into the pinned child, which completes without
// re-invoking its model, and the parent merges the child's final values and
// runs on.
func TestCreateAgentHITLNestedSupervisorResume(t *testing.T) {
	var toolRuns, reportRuns int
	childModel := &sequenceModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{hitlEchoToolCall("call_1", "hi")}},
		messages.AI("done"),
	}}
	worker := newHITLWorkerAgent(t, &toolRuns, childModel)
	supervisor := newSupervisorGraph(t, worker, checkpoint.NewMemorySaver(), &reportRuns)

	res, err := supervisor.InvokeWithOptions(context.Background(),
		map[string]any{"messages": []messages.Message{messages.Human("hi")}},
		graphpkg.Options{ThreadID: "sup-1"})
	if err != nil {
		t.Fatalf("first supervisor invoke: %v", err)
	}
	if len(res.Interrupts) != 1 {
		t.Fatalf("expected the child's HITL pause to surface on the parent, got %+v", res.Interrupts)
	}
	intr := res.Interrupts[0]
	if !strings.HasPrefix(intr.NS, "worker:") || !strings.Contains(intr.NS, "/hitl:") {
		t.Fatalf("expected a nested hitl NS under the worker subgraph task, got %q", intr.NS)
	}
	if !strings.HasPrefix(intr.ID, "hitl-") {
		t.Fatalf("expected the child hitl node's interrupt id, got %q", intr.ID)
	}
	request, err := middleware.HITLRequestFromInterrupt(intr)
	if err != nil {
		t.Fatalf("decode nested interrupt: %v", err)
	}
	if len(request.ActionRequests) != 1 || request.ActionRequests[0].Name != "echo" ||
		request.ActionRequests[0].Args["tool_input"] != "hi" {
		t.Fatalf("nested interrupt request mismatch: %#v", request.ActionRequests)
	}
	// Gatekeeper (design §6.4, under nesting): the child model ran exactly
	// once at the pause and the tool has not executed.
	if len(childModel.invocations) != 1 {
		t.Fatalf("expected the child model invoked exactly once at pause, got %d", len(childModel.invocations))
	}
	if toolRuns != 0 || reportRuns != 0 {
		t.Fatalf("nothing may run before approval: tool=%d report=%d", toolRuns, reportRuns)
	}

	// Resume from the PARENT side, addressing the nested interrupt by NS.
	res, err = supervisor.InvokeWithOptions(context.Background(), nil, graphpkg.Options{
		ThreadID: "sup-1",
		Resume: map[string]any{intr.NS: middleware.HITLResponse{
			Decisions: []middleware.Decision{{Type: middleware.DecisionApprove}},
		}},
	})
	if err != nil {
		t.Fatalf("supervisor resume by NS: %v", err)
	}
	if len(res.Interrupts) != 0 {
		t.Fatalf("expected completion after the NS-addressed resume, got %+v", res.Interrupts)
	}
	// The child's hitl node alone re-ran: no model replay, one tool run.
	if len(childModel.invocations) != 2 {
		t.Fatalf("expected exactly two child model calls (no replay of the paused one), got %d", len(childModel.invocations))
	}
	if toolRuns != 1 {
		t.Fatalf("expected the approved call to execute once, ran %d times", toolRuns)
	}
	// The parent merged the child's final values and continued to "report".
	out, _ := res.Values["messages"].([]messages.Message)
	if len(out) != 4 || out[0].Content != "hi" || out[1].Role != messages.RoleAI ||
		out[2].Role != messages.RoleTool || out[2].Content != "echo:hi" || out[3].Content != "done" {
		t.Fatalf("parent did not merge the child's final messages: %#v", out)
	}
	if reportRuns != 1 || res.Values["report"] != "4 messages" {
		t.Fatalf("expected the parent to continue past the subgraph with merged values, got report=%v runs=%d",
			res.Values["report"], reportRuns)
	}
}

// TestCreateAgentHITLNestedSupervisorGraphNS (design §6.13): the same nested
// pause resumed through Options.Graph scoping — the child task's checkpoint
// namespace directs the scalar resume into the child, while a Graph naming no
// pending interrupt's namespace is a descriptive error.
func TestCreateAgentHITLNestedSupervisorGraphNS(t *testing.T) {
	var toolRuns, reportRuns int
	childModel := &sequenceModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{hitlEchoToolCall("call_1", "hi")}},
		messages.AI("done"),
	}}
	worker := newHITLWorkerAgent(t, &toolRuns, childModel)
	supervisor := newSupervisorGraph(t, worker, checkpoint.NewMemorySaver(), &reportRuns)

	res, err := supervisor.InvokeWithOptions(context.Background(),
		map[string]any{"messages": []messages.Message{messages.Human("hi")}},
		graphpkg.Options{ThreadID: "sup-2"})
	if err != nil || len(res.Interrupts) != 1 {
		t.Fatalf("first invoke: %v interrupts=%+v", err, res.Interrupts)
	}
	intr := res.Interrupts[0]
	// The child task's namespace is the interrupt NS minus its trailing
	// in-node segment: "worker:<taskID>" for ".../hitl:<taskID>".
	cut := strings.LastIndex(intr.NS, "/hitl:")
	if cut < 0 {
		t.Fatalf("interrupt NS %q carries no hitl segment", intr.NS)
	}
	childNS := intr.NS[:cut]
	if !strings.HasPrefix(childNS, "worker:") {
		t.Fatalf("child namespace %q is not under the worker subgraph", childNS)
	}

	// A Graph naming no pending interrupt's namespace errors descriptively.
	_, err = supervisor.InvokeWithOptions(context.Background(), nil, graphpkg.Options{
		ThreadID: "sup-2",
		Graph:    "no-such-namespace",
		Resume:   middleware.HITLResponse{Decisions: []middleware.Decision{{Type: middleware.DecisionApprove}}},
	})
	if err == nil || !strings.Contains(err.Error(), "matches no pending interrupt") {
		t.Fatalf("expected a no-pending-interrupt error for the disjoint Graph, got %v", err)
	}

	// The scoped resume: Graph = the child task's namespace, scalar answer.
	res, err = supervisor.InvokeWithOptions(context.Background(), nil, graphpkg.Options{
		ThreadID: "sup-2",
		Graph:    childNS,
		Resume:   middleware.HITLResponse{Decisions: []middleware.Decision{{Type: middleware.DecisionApprove}}},
	})
	if err != nil {
		t.Fatalf("graph-scoped resume: %v", err)
	}
	if len(res.Interrupts) != 0 {
		t.Fatalf("expected completion after the graph-scoped resume, got %+v", res.Interrupts)
	}
	if len(childModel.invocations) != 2 || toolRuns != 1 {
		t.Fatalf("expected two child model calls and one tool run, got %d/%d", len(childModel.invocations), toolRuns)
	}
	if reportRuns != 1 || res.Values["report"] != "4 messages" {
		t.Fatalf("expected the parent to complete with merged values, got report=%v runs=%d",
			res.Values["report"], reportRuns)
	}
}

// wireValue mirrors the sqlite/postgres savers' typed-value split: every `any`
// leaf (channel values, planned-task args, write values) crosses the boundary
// through the serde's type-tagged JSON instead of plain JSON, so registered
// Go types ([]messages.Message, types.Interrupt, ...) round-trip exactly.
type wireValue struct {
	Type string          `json:"type"`
	Data json.RawMessage `json:"data"`
}

type wireTask struct {
	ID   string               `json:"id"`
	Node string               `json:"node"`
	Arg  map[string]wireValue `json:"arg,omitempty"`
}

type wireCheckpoint struct {
	V               int                         `json:"v"`
	ID              string                      `json:"id"`
	TS              time.Time                   `json:"ts"`
	ChannelValues   map[string]wireValue       `json:"channel_values,omitempty"`
	ChannelVersions map[string]int64            `json:"channel_versions,omitempty"`
	VersionsSeen    map[string]map[string]int64 `json:"versions_seen,omitempty"`
	Next            []wireTask                  `json:"next,omitempty"`
}

type wireWrite struct {
	TaskID   string    `json:"task_id"`
	TaskPath string    `json:"task_path"`
	Channel  string    `json:"channel"`
	Value    wireValue `json:"value"`
}

// TestCreateAgentHITLCrossProcessCheckpointRoundTrip (design §6.14): the
// paused thread survives a serialization boundary. The thread's checkpoints —
// including the ReservedInterrupt pending write whose types.Interrupt.Value
// is the snake_case map form of the HITLRequest — are marshaled through the
// JSON serde (the typed split the durable savers persist), unmarshaled, and
// re-Put into a fresh MemorySaver; an agent rebuilt over that saver resumes
// to completion, answering through the wire-form decisions map that a
// non-Go producer (or a JSON-decoded gateway payload) would send.
func TestCreateAgentHITLCrossProcessCheckpointRoundTrip(t *testing.T) {
	ctx := context.Background()
	ser := serde.NewJSONSerializer()

	dumpValue := func(v any) wireValue {
		typ, data, err := ser.DumpsTyped(v)
		if err != nil {
			t.Fatalf("serialize %T: %v", v, err)
		}
		return wireValue{Type: typ, Data: data}
	}
	loadValue := func(w wireValue) any {
		v, err := ser.LoadsTyped(w.Type, w.Data)
		if err != nil {
			t.Fatalf("deserialize %q: %v", w.Type, err)
		}
		return v
	}

	// Process 1: pause the run.
	var toolRuns int
	model1 := &sequenceModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{hitlEchoToolCall("call_1", "hi")}},
	}}
	saver1 := checkpoint.NewMemorySaver()
	agent1, err := CreateAgent(model1, []coretools.Tool{newHitlTestTool(t, &toolRuns)},
		WithAgentMiddleware(middleware.NewInterruptHumanInTheLoopMiddleware(map[string]middleware.InterruptConfig{
			"echo": {AllowedDecisions: []middleware.DecisionType{middleware.DecisionApprove}},
		})),
		WithAgentCheckpointer(saver1),
	)
	if err != nil {
		t.Fatalf("create agent 1: %v", err)
	}
	_, interrupts, err := agent1.InvokeWithStateOptions(ctx,
		[]messages.Message{messages.Human("hi")}, graphpkg.Options{ThreadID: "t1"})
	if err != nil || len(interrupts) != 1 {
		t.Fatalf("first process invoke: %v interrupts=%+v", err, interrupts)
	}
	if len(model1.invocations) != 1 || toolRuns != 0 {
		t.Fatalf("process 1 paused before approval: model=%d tool=%d", len(model1.invocations), toolRuns)
	}

	// Serialize the whole thread out of process 1.
	tuples, err := saver1.List(ctx, checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("list thread: %v", err)
	}
	if len(tuples) == 0 {
		t.Fatalf("the paused thread persisted no checkpoints")
	}
	type wireTuple struct {
		Config checkpoint.Config `json:"config"`
		Parent *checkpoint.Config `json:"parent,omitempty"`
		MD     checkpoint.Metadata `json:"md"`
		CP     wireCheckpoint     `json:"cp"`
		Writes []wireWrite        `json:"writes,omitempty"`
	}
	var wire []wireTuple
	for _, tup := range tuples { // newest first; marshaling order is irrelevant
		wcp := wireCheckpoint{
			V:               tup.Checkpoint.V,
			ID:              tup.Checkpoint.ID,
			TS:              tup.Checkpoint.TS,
			ChannelVersions: tup.Checkpoint.ChannelVersions,
			VersionsSeen:    tup.Checkpoint.VersionsSeen,
		}
		if tup.Checkpoint.ChannelValues != nil {
			wcp.ChannelValues = make(map[string]wireValue, len(tup.Checkpoint.ChannelValues))
			for ch, v := range tup.Checkpoint.ChannelValues {
				wcp.ChannelValues[ch] = dumpValue(v)
			}
		}
		for _, task := range tup.Checkpoint.Next {
			wt := wireTask{ID: task.ID, Node: task.Node}
			if task.Arg != nil {
				wt.Arg = make(map[string]wireValue, len(task.Arg))
				for k, v := range task.Arg {
					wt.Arg[k] = dumpValue(v)
				}
			}
			wcp.Next = append(wcp.Next, wt)
		}
		wt := wireTuple{Config: tup.Config, Parent: tup.ParentConfig, MD: tup.Metadata, CP: wcp}
		for _, w := range tup.PendingWrites {
			wt.Writes = append(wt.Writes, wireWrite{
				TaskID: w.TaskID, TaskPath: w.TaskPath, Channel: w.Channel, Value: dumpValue(w.Value),
			})
		}
		wire = append(wire, wt)
	}
	blob, err := json.Marshal(wire)
	if err != nil {
		t.Fatalf("marshal thread: %v", err)
	}
	var restored []wireTuple
	if err := json.Unmarshal(blob, &restored); err != nil {
		t.Fatalf("unmarshal thread: %v", err)
	}

	// Process 2: pour the thread into a fresh MemorySaver, oldest first so
	// each Put's parent link resolves.
	saver2 := checkpoint.NewMemorySaver()
	var roundTripped types.Interrupt
	for i := len(restored) - 1; i >= 0; i-- {
		wt := restored[i]
		cp := checkpoint.Checkpoint{
			V:               wt.CP.V,
			ID:              wt.CP.ID,
			TS:              wt.CP.TS,
			ChannelVersions: wt.CP.ChannelVersions,
			VersionsSeen:    wt.CP.VersionsSeen,
		}
		if wt.CP.ChannelValues != nil {
			cp.ChannelValues = make(map[string]any, len(wt.CP.ChannelValues))
			for ch, v := range wt.CP.ChannelValues {
				cp.ChannelValues[ch] = loadValue(v)
			}
		}
		for _, task := range wt.CP.Next {
			pt := checkpoint.PlannedTask{ID: task.ID, Node: task.Node}
			if task.Arg != nil {
				pt.Arg = make(map[string]any, len(task.Arg))
				for k, v := range task.Arg {
					pt.Arg[k] = loadValue(v)
				}
			}
			cp.Next = append(cp.Next, pt)
		}
		cfg := checkpoint.Config{ThreadID: wt.Config.ThreadID, CheckpointNS: wt.Config.CheckpointNS}
		if wt.Parent != nil {
			cfg.CheckpointID = wt.Parent.CheckpointID
		}
		put, err := saver2.Put(ctx, cfg, cp, wt.MD, nil)
		if err != nil {
			t.Fatalf("put checkpoint %s: %v", cp.ID, err)
		}
		for _, w := range wt.Writes {
			value := loadValue(w.Value)
			if w.Channel == checkpoint.ReservedInterrupt {
				if intr, ok := value.(types.Interrupt); ok {
					roundTripped = intr
				}
			}
			if err := saver2.PutWrites(ctx, put, []checkpoint.Write{{
				TaskID: w.TaskID, TaskPath: w.TaskPath, Channel: w.Channel, Value: value,
			}}, w.TaskID, w.TaskPath); err != nil {
				t.Fatalf("put write %q/%q: %v", w.TaskID, w.Channel, err)
			}
		}
	}

	// Behavior lock: the round-tripped pending interrupt kept its identity
	// and its Value stayed the snake_case map wire form, which
	// HITLRequestFromInterrupt restores (PR1's always-map payload).
	if roundTripped.ID != interrupts[0].ID || roundTripped.NS != interrupts[0].NS {
		t.Fatalf("round-tripped interrupt identity mismatch: %+v vs %+v", roundTripped, interrupts[0])
	}
	request, err := middleware.HITLRequestFromInterrupt(roundTripped)
	if err != nil {
		t.Fatalf("decode round-tripped interrupt: %v", err)
	}
	if len(request.ActionRequests) != 1 || request.ActionRequests[0].Name != "echo" ||
		request.ActionRequests[0].Args["tool_input"] != "hi" {
		t.Fatalf("round-tripped request mismatch: %#v", request.ActionRequests)
	}

	// A fresh agent over the restored saver — only the persisted thread
	// state carries over, nothing in-memory from process 1.
	model2 := &sequenceModel{responses: []messages.Message{messages.AI("done")}}
	agent2, err := CreateAgent(model2, []coretools.Tool{newHitlTestTool(t, &toolRuns)},
		WithAgentMiddleware(middleware.NewInterruptHumanInTheLoopMiddleware(map[string]middleware.InterruptConfig{
			"echo": {AllowedDecisions: []middleware.DecisionType{middleware.DecisionApprove}},
		})),
		WithAgentCheckpointer(saver2),
	)
	if err != nil {
		t.Fatalf("create agent 2: %v", err)
	}
	// The answer arrives in the JSON wire form (decisions as maps), addressed
	// by the interrupt NS the way a gateway that persisted the pause would.
	values, interrupts, err := agent2.Resume(ctx, graphpkg.Options{
		ThreadID: "t1",
		Resume: map[string]any{roundTripped.NS: map[string]any{
			"decisions": []any{map[string]any{"type": "approve"}},
		}},
	})
	if err != nil {
		t.Fatalf("second process resume: %v", err)
	}
	if len(interrupts) != 0 {
		t.Fatalf("expected completion in the second process, got %+v", interrupts)
	}
	if len(model1.invocations) != 1 || len(model2.invocations) != 1 {
		t.Fatalf("expected exactly one model call per process (no replay), got %d/%d",
			len(model1.invocations), len(model2.invocations))
	}
	if toolRuns != 1 {
		t.Fatalf("expected the approved call to execute once in process 2, ran %d times", toolRuns)
	}
	out, _ := values["messages"].([]messages.Message)
	if len(out) != 4 || out[2].Role != messages.RoleTool || out[2].Content != "echo:hi" || out[3].Content != "done" {
		t.Fatalf("second-process final messages mismatch: %#v", out)
	}
}

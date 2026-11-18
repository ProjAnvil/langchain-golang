// Command llm-compiler_LLMCompiler is the Go port of
// langgraph/examples/llm-compiler/LLMCompiler.ipynb.
//
// The notebook has three moving parts: a planner emits a DAG of tool tasks
// whose arguments reference earlier task results ($1, ${2}, ...), a scheduler
// runs ready tasks concurrently and resolves those references, and a joiner
// either returns the final answer or asks for a replan. The Go port models
// the DAG with StateGraph conditional edges that fan out ready tasks via
// types.Send; each task node writes its observation into a shared map. The
// joiner drives the same plan/schedule -> join -> END-or-replan loop.
//
// All planning, tool execution, and joining are deterministic offline stubs.
// The "calculate" tool uses a small arithmetic parser, never code execution,
// and the "search" tool returns canned evidence. No API key or network.
package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"regexp"
	"strconv"
	"strings"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// compileTask is one task of the generated DAG. Args may contain $N or ${N}
// references to the observation of task N; Deps lists those task indices.
type compileTask struct {
	Idx  int
	Name string
	Tool string
	Args string
	Deps []int
}

var refPattern = regexp.MustCompile(`\$\{?(\d+)\}?`)

// resolveArgs replaces $N / ${N} references with string observations,
// mirroring the notebook's _resolve_arg.
func resolveArgs(args string, observations map[int]string) string {
	return refPattern.ReplaceAllStringFunc(args, func(match string) string {
		groups := refPattern.FindStringSubmatch(match)
		n, _ := strconv.Atoi(groups[1])
		if value, ok := observations[n]; ok {
			return value
		}
		return match
	})
}

// --- A deliberately small, safe arithmetic evaluator (no eval / exec). ---

type arithParser struct {
	s string
	i int
}

func (p *arithParser) skip() {
	for p.i < len(p.s) && p.s[p.i] == ' ' {
		p.i++
	}
}

func (p *arithParser) parse() (float64, error) {
	v, err := p.parseAddSub()
	if err != nil {
		return 0, err
	}
	p.skip()
	if p.i != len(p.s) {
		return 0, fmt.Errorf("unexpected input %q", p.s[p.i:])
	}
	return v, nil
}

func (p *arithParser) parseAddSub() (float64, error) {
	left, err := p.parseMulDiv()
	if err != nil {
		return 0, err
	}
	for {
		p.skip()
		if p.i >= len(p.s) || (p.s[p.i] != '+' && p.s[p.i] != '-') {
			break
		}
		op := p.s[p.i]
		p.i++
		right, err := p.parseMulDiv()
		if err != nil {
			return 0, err
		}
		if op == '+' {
			left += right
		} else {
			left -= right
		}
	}
	return left, nil
}

func (p *arithParser) parseMulDiv() (float64, error) {
	left, err := p.parsePow()
	if err != nil {
		return 0, err
	}
	for {
		p.skip()
		if p.i >= len(p.s) || (p.s[p.i] != '*' && p.s[p.i] != '/') {
			break
		}
		op := p.s[p.i]
		p.i++
		right, err := p.parsePow()
		if err != nil {
			return 0, err
		}
		if op == '*' {
			left *= right
		} else {
			if right == 0 {
				return 0, fmt.Errorf("division by zero")
			}
			left /= right
		}
	}
	return left, nil
}

func (p *arithParser) parsePow() (float64, error) {
	base, err := p.parseUnary()
	if err != nil {
		return 0, err
	}
	p.skip()
	if p.i < len(p.s) && p.s[p.i] == '^' {
		p.i++
		exp, err := p.parsePow() // right-associative, like Python's **
		if err != nil {
			return 0, err
		}
		return math.Pow(base, exp), nil
	}
	return base, nil
}

func (p *arithParser) parseUnary() (float64, error) {
	p.skip()
	if p.i < len(p.s) && p.s[p.i] == '-' {
		p.i++
		v, err := p.parseUnary()
		return -v, err
	}
	if p.i < len(p.s) && p.s[p.i] == '+' {
		p.i++
		return p.parseUnary()
	}
	return p.parsePrimary()
}

func (p *arithParser) parsePrimary() (float64, error) {
	p.skip()
	if p.i < len(p.s) && p.s[p.i] == '(' {
		p.i++
		v, err := p.parseAddSub()
		if err != nil {
			return 0, err
		}
		p.skip()
		if p.i >= len(p.s) || p.s[p.i] != ')' {
			return 0, fmt.Errorf("missing )")
		}
		p.i++
		return v, nil
	}
	start := p.i
	for p.i < len(p.s) && ((p.s[p.i] >= '0' && p.s[p.i] <= '9') || p.s[p.i] == '.') {
		p.i++
	}
	if start == p.i {
		return 0, fmt.Errorf("expected a number at %d", p.i)
	}
	return strconv.ParseFloat(p.s[start:p.i], 64)
}

func safeCalculate(expr string) string {
	v, err := (&arithParser{s: expr}).parse()
	if err != nil {
		return "ERROR: " + err.Error()
	}
	return strconv.FormatFloat(v, 'f', -1, 64)
}

// searchStub is the offline stand-in for TavilySearchResults.
func searchStub(query string) string {
	switch {
	case strings.Contains(strings.ToLower(query), "temperature"):
		return "18"
	case strings.Contains(strings.ToLower(query), "gdp"):
		return "1200"
	default:
		return "evidence for " + query
	}
}

// mergeObservations is the reducer for the shared observation map. Several
// task nodes may write in the same superstep, so writes must be merged.
func mergeObservations(existing any, update any) (any, error) {
	out := map[int]string{}
	if m, ok := existing.(map[int]string); ok {
		for k, v := range m {
			out[k] = v
		}
	}
	if m, ok := update.(map[int]string); ok {
		for k, v := range m {
			out[k] = v
		}
	}
	return out, nil
}

// planner stands in for create_planner: round 0 emits the initial DAG; later
// rounds emit a small follow-up task that depends on an earlier observation
// (the notebook's "begin counting at N" replan behavior).
func planner(_ runtime.Runtime, state map[string]any) (any, error) {
	round, _ := state["round"].(int)
	var tasks []compileTask
	if round == 0 {
		tasks = []compileTask{
			{Idx: 1, Name: "$1", Tool: "search", Args: "current temperature in San Francisco"},
			{Idx: 2, Name: "$2", Tool: "search", Args: "GDP of New York"},
			{Idx: 3, Name: "$3", Tool: "calculate", Args: "$1 ^ 3", Deps: []int{1}},
		}
		fmt.Println("planner: initial DAG with 3 tasks (1 and 2 independent, 3 depends on 1)")
	} else {
		tasks = []compileTask{
			{Idx: 4, Name: "$4", Tool: "calculate", Args: "$2 / 4", Deps: []int{2}},
		}
		fmt.Println("planner: replan DAG with 1 follow-up task dependent on task 2")
	}
	return map[string]any{"tasks": tasks, "round": round + 1}, nil
}

// dispatchTasks is the scheduler's conditional edge. It dispatches the next
// ready task via types.Send, or routes to the joiner when the DAG is drained.
// Only one task is dispatched per superstep: the router is attached to both
// the planner and the task node, and firing it for several concurrent tasks
// would send duplicates. Dispatching one task at a time keeps the offline
// trace deterministic while still using Send for dependency-aware scheduling.
func dispatchTasks(_ runtime.Runtime, state map[string]any) ([]any, error) {
	tasks, _ := state["tasks"].([]compileTask)
	observations, _ := state["observations"].(map[int]string)
	for _, t := range tasks {
		if _, done := observations[t.Idx]; done {
			continue
		}
		ready := true
		for _, dep := range t.Deps {
			if _, ok := observations[dep]; !ok {
				ready = false
				break
			}
		}
		if !ready {
			continue
		}
		// A Send-driven node receives exactly Arg as its input, so pass the
		// observations the task needs to resolve its $N references.
		fmt.Printf("scheduler: dispatching task %d (%s)\n", t.Idx, t.Tool)
		return []any{&types.Send{Node: "task", Arg: map[string]any{
			"task":         t,
			"observations": observations,
		}}}, nil
	}
	return graphpkg.To("join"), nil
}

// taskNode executes one dispatched task with a safe tool and records its
// observation into the shared map.
func taskNode(_ runtime.Runtime, arg map[string]any) (any, error) {
	t, ok := arg["task"].(compileTask)
	if !ok {
		return nil, fmt.Errorf("task: Send arg missing task")
	}
	observations, _ := arg["observations"].(map[int]string)
	resolved := resolveArgs(t.Args, observations)
	var result string
	switch t.Tool {
	case "search":
		result = searchStub(resolved)
	case "calculate":
		result = safeCalculate(resolved)
	default:
		result = "ERROR: unknown tool " + t.Tool
	}
	fmt.Printf("  task %d %s[%s] -> %s\n", t.Idx, t.Tool, resolved, result)
	return map[string]any{"observations": map[int]string{t.Idx: result}}, nil
}

// joiner stands in for the LLMCompiler joiner: after the first round it asks
// for a replan (a SystemMessage makes shouldContinue loop), after the second
// it emits the final answer as an AI message.
func joiner(_ runtime.Runtime, state map[string]any) (any, error) {
	round, _ := state["round"].(int)
	if round <= 1 {
		fmt.Println("joiner: needs one more calculation, requesting replan")
		return map[string]any{"messages": []messages.Message{
			messages.AI("Thought: initial observations gathered; one more calculation is needed."),
			messages.System("Context from last attempt: verify the final calculation."),
		}}, nil
	}
	obs, _ := state["observations"].(map[int]string)
	answer := fmt.Sprintf("Final answer: SF temperature cubed = %s; one quarter of NY GDP = %s.",
		obs[3], obs[4])
	fmt.Printf("joiner: %s\n", answer)
	return map[string]any{"messages": []messages.Message{
		messages.AI("Thought: all tasks resolved."),
		messages.AI(answer),
	}}, nil
}

// shouldContinue mirrors the notebook: an AI message last ends the run,
// otherwise go back to the planner.
func shouldContinue(_ runtime.Runtime, state map[string]any) ([]any, error) {
	msgs, _ := state["messages"].([]messages.Message)
	if len(msgs) > 0 && msgs[len(msgs)-1].Role == messages.RoleAI {
		return graphpkg.To(types.END), nil
	}
	return graphpkg.To("planner"), nil
}

func main() {
	ctx := context.Background()

	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddReducer("observations", mergeObservations)
	g.AddNode("planner", planner)
	g.AddNode("task", taskNode)
	g.AddNode("join", joiner)
	g.AddEdge(types.START, "planner")
	// The scheduler fans out ready tasks from both the planner and from each
	// completed task; when none are ready it routes to the joiner.
	g.AddConditionalEdges("planner", dispatchTasks)
	g.AddConditionalEdges("task", dispatchTasks)
	g.AddConditionalEdges("join", shouldContinue)
	cg, err := g.Compile(graphpkg.WithRecursionLimit(100))
	if err != nil {
		log.Fatalf("compile: %v", err)
	}

	question := "What's the temperature in SF raised to the 3rd power?"
	result, err := cg.Invoke(ctx, map[string]any{
		"messages":     []messages.Message{messages.Human(question)},
		"round":        0,
		"observations": map[int]string{},
	})
	if err != nil {
		log.Fatalf("invoke: %v", err)
	}
	fmt.Println("--- final answer ---")
	msgs, _ := result.Values["messages"].([]messages.Message)
	if len(msgs) > 0 {
		fmt.Println(messages.Text(msgs[len(msgs)-1]))
	}
}

// End-to-end test for the StreamEvents auto-attach hook (design t18 PR3):
// with LangSmith env vars pointing at an httptest server, the
// runnables.StreamEvents driver appends a LangChainTracer before its
// collector and closes it at iteration end, so a full run tree
// (root sequence + step children, dotted-order chains, inputs/outputs)
// reaches /runs/batch without the caller wiring anything.
//
// This file is deliberately an external test package (tracers_test): it must
// import core/runnables, which itself imports core/tracers — an in-package
// test file would create an import cycle.
package tracers_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tracers"
)

func e2eFunc(transform func(string) string) runnables.Func[string, string] {
	return runnables.NewFunc(
		func(_ context.Context, input string, _ ...runnables.Option) (string, error) {
			return transform(input), nil
		},
		schema.String(""), schema.String(""),
	)
}

func TestStreamEventsAutoAttachesLangSmithTracer(t *testing.T) {
	var mu sync.Mutex
	var bodies []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var decoded map[string]any
		_ = json.Unmarshal(raw, &decoded)
		mu.Lock()
		bodies = append(bodies, decoded)
		mu.Unlock()
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	t.Setenv("LANGSMITH_TRACING", "true")
	t.Setenv("LANGSMITH_API_KEY", "e2e-key")
	t.Setenv("LANGSMITH_ENDPOINT", server.URL)
	t.Setenv("LANGSMITH_PROJECT", "e2e-proj")
	if !tracers.EnabledFromEnv() {
		t.Fatal("EnabledFromEnv must be true with tracing flag plus key")
	}

	chain := runnables.Pipe(
		e2eFunc(func(s string) string { return s + s }),
		e2eFunc(strings.ToUpper),
	)
	var sawEvents int
	for event, err := range runnables.StreamEvents(
		context.Background(), chain, "go", runnables.StreamEventOptions{},
	) {
		if err != nil {
			t.Fatalf("stream event error: %v", err)
		}
		if event.Event == "" {
			t.Fatalf("empty event at index %d", sawEvents)
		}
		sawEvents++
	}
	if sawEvents == 0 {
		t.Fatal("no stream events observed")
	}

	// The driver closes the auto-attached tracer at iteration end, which
	// flushes; poll briefly anyway to tolerate scheduling jitter.
	deadline := time.Now().Add(5 * time.Second)
	for {
		mu.Lock()
		count := len(bodies)
		mu.Unlock()
		if count > 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("auto-attached tracer never posted to /runs/batch")
		}
		time.Sleep(10 * time.Millisecond)
	}

	mu.Lock()
	defer mu.Unlock()
	postsByID := map[string]map[string]any{}
	patchesByID := map[string]map[string]any{}
	posts, patches := 0, 0
	for _, body := range bodies {
		for _, entry := range []struct {
			key  string
			sink map[string]map[string]any
		}{{"post", postsByID}, {"patch", patchesByID}} {
			items, _ := body[entry.key].([]any)
			for _, item := range items {
				run := item.(map[string]any)
				entry.sink[run["id"].(string)] = run
				if entry.key == "post" {
					posts++
				} else {
					patches++
				}
			}
		}
	}
	if posts < 3 || patches < 3 {
		t.Fatalf("auto tracer posted %d creates / %d updates, want >= 3/3 (sequence + 2 steps)", posts, patches)
	}
	// Root run: the sequence chain with the streamed input.
	var root map[string]any
	for _, run := range postsByID {
		if run["name"] == "sequence" && run["run_type"] == "chain" {
			root = run
			break
		}
	}
	if root == nil {
		t.Fatalf("no sequence run posted: %#v", postsByID)
	}
	rootID := root["id"].(string)
	if root["trace_id"] != rootID {
		t.Fatalf("root trace_id %v", root["trace_id"])
	}
	if root["session_name"] != "e2e-proj" || root["session_id"] != rootID {
		t.Fatalf("root session fields: %#v", root)
	}
	rootDO, _ := root["dotted_order"].(string)
	if !strings.HasSuffix(rootDO, rootID) || !strings.HasPrefix(rootDO, "20") {
		t.Fatalf("root dotted_order %q", rootDO)
	}
	// Step children: parent chain and dotted-order chaining.
	children := map[string]map[string]any{}
	for id, run := range postsByID {
		if run["parent_run_id"] == rootID {
			children[id] = run
			do, _ := run["dotted_order"].(string)
			if !strings.HasPrefix(do, rootDO+".") || !strings.HasSuffix(do, id) {
				t.Fatalf("child dotted_order %q does not extend root %q", do, rootDO)
			}
			if run["trace_id"] != rootID {
				t.Fatalf("child trace_id %v", run["trace_id"])
			}
		}
	}
	if len(children) != 2 {
		t.Fatalf("sequence children %d, want 2", len(children))
	}
	// Step patches carry their outputs; the streamed root's end patch has no
	// output (Go chain streams end with nil output — chunk aggregation is
	// the consumer's job, a documented PR1 divergence).
	stepOutputs := map[string]bool{"gogo": false, "GOGO": false}
	for id := range children {
		if out, ok := patchesByID[id]["outputs"].(string); ok {
			stepOutputs[out] = true
		}
	}
	for output, seen := range stepOutputs {
		if !seen {
			t.Fatalf("step patch outputs missing %q: %v", output, patchesByID)
		}
	}
	if _, has := patchesByID[rootID]["outputs"]; has {
		t.Fatalf("streamed root patch outputs %#v, want none", patchesByID[rootID]["outputs"])
	}
	if _, has := patchesByID[rootID]["end_time"]; !has {
		t.Fatal("root patch missing end_time")
	}
}

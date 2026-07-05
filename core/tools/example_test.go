package tools

import (
	"context"
	"fmt"
	"sort"
)

// exampleSearchArgs is the argument struct FromFunc reflects into a JSON
// schema in ExampleFromFunc_struct.
type exampleSearchArgs struct {
	Query string `json:"query"`
	Limit int    `json:"limit"`
}

// ExampleFromFunc_struct builds a tool from an ordinary Go func via FromFunc.
// FromFunc reflects the argument struct's `json` tags into the tool's JSON
// schema, so callers get schema-driven dispatch without hand-writing a schema.
// The example prints the reflected schema's property names and required fields
// (both deterministic), then invokes the tool.
func ExampleFromFunc_struct() {
	tool, err := FromFunc("search", "search the index",
		func(ctx context.Context, a exampleSearchArgs) (Result, error) {
			return Result{Content: fmt.Sprintf("query=%s limit=%d", a.Query, a.Limit)}, nil
		})
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println(tool.Name())
	fmt.Println(tool.Description())

	// The reflected schema: an object with one property per exported struct
	// field. Property names come from the json tags; required fields are those
	// that are neither pointers nor marked omitempty (here: both).
	sch := tool.ArgsSchema()
	props, _ := sch["properties"].(map[string]any)
	names := make([]string, 0, len(props))
	for name := range props {
		names = append(names, name)
	}
	sort.Strings(names)
	fmt.Println(names)
	if required, ok := sch["required"].([]string); ok {
		fmt.Println(required)
	}

	res, err := tool.Invoke(context.Background(), map[string]any{"query": "go", "limit": 3})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(res.Content)
	// Output:
	// search
	// search the index
	// [limit query]
	// [query limit]
	// query=go limit=3
}

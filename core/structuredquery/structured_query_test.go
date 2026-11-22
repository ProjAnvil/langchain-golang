package structuredquery

import (
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

func TestOperatorAndComparatorValues(t *testing.T) {
	operators := map[Operator]string{
		OperatorAnd: "and",
		OperatorOr:  "or",
		OperatorNot: "not",
	}
	for op, want := range operators {
		if string(op) != want {
			t.Errorf("operator = %q, want %q", op, want)
		}
	}
	comparators := map[Comparator]string{
		ComparatorEq: "eq", ComparatorNe: "ne", ComparatorGt: "gt",
		ComparatorGte: "gte", ComparatorLt: "lt", ComparatorLte: "lte",
		ComparatorIn: "in", ComparatorNin: "nin", ComparatorLike: "like",
		ComparatorContain: "contain",
	}
	for cmp, want := range comparators {
		if string(cmp) != want {
			t.Errorf("comparator = %q, want %q", cmp, want)
		}
	}
}

// dispatchVisitor records which Visit* method the Accept dispatch selected.
type dispatchVisitor struct {
	seen []string
}

func (v *dispatchVisitor) VisitOperation(*Operation) (any, error) {
	v.seen = append(v.seen, "operation")
	return "op", nil
}
func (v *dispatchVisitor) VisitComparison(*Comparison) (any, error) {
	v.seen = append(v.seen, "comparison")
	return "cmp", nil
}
func (v *dispatchVisitor) VisitStructuredQuery(*StructuredQuery) (any, error) {
	v.seen = append(v.seen, "structured_query")
	return "sq", nil
}

func TestAcceptDispatch(t *testing.T) {
	cases := []struct {
		name string
		expr Expr
		want string
	}{
		{"comparison", NewComparison(ComparatorEq, "a", 1), "comparison"},
		{"operation", NewOperation(OperatorAnd, NewComparison(ComparatorEq, "a", 1)), "operation"},
		{"structured_query", NewStructuredQuery("q", nil, 3), "structured_query"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v := &dispatchVisitor{}
			result, err := tc.expr.Accept(v)
			if err != nil {
				t.Fatalf("Accept: %v", err)
			}
			if result == nil {
				t.Fatalf("Accept returned nil result")
			}
			if len(v.seen) != 1 || v.seen[0] != tc.want {
				t.Fatalf("dispatch = %v, want single %q", v.seen, tc.want)
			}
		})
	}
}

func TestValidateStructuredQueryValid(t *testing.T) {
	q := NewStructuredQuery("find docs",
		NewOperation(OperatorAnd,
			NewComparison(ComparatorGte, "age", 18),
			NewComparison(ComparatorIn, "tag", []any{"a", "b"}),
			NewOperation(OperatorNot, NewComparison(ComparatorEq, "hidden", true)),
		),
		10,
	)
	if err := ValidateStructuredQuery(q); err != nil {
		t.Fatalf("ValidateStructuredQuery: %v", err)
	}
}

func TestValidateStructuredQueryErrors(t *testing.T) {
	cases := []struct {
		name    string
		query   *StructuredQuery
		wantErr string
	}{
		{"nil query", nil, "query is nil"},
		{"unknown comparator", NewStructuredQuery("q", NewComparison(Comparator("bogus"), "a", 1), 0), "invalid comparator"},
		{"unknown operator", NewStructuredQuery("q", NewOperation(Operator("bogus"), NewComparison(ComparatorEq, "a", 1)), 0), "invalid operator"},
		{"empty attribute", NewStructuredQuery("q", NewComparison(ComparatorEq, "", 1), 0), "attribute must not be empty"},
		{"empty operation", NewStructuredQuery("q", NewOperation(OperatorAnd), 0), "at least one argument"},
		{"not arity", NewStructuredQuery("q", NewOperation(OperatorNot,
			NewComparison(ComparatorEq, "a", 1), NewComparison(ComparatorEq, "b", 2)), 0), "exactly one argument"},
		{"negative limit", NewStructuredQuery("q", nil, -1), "non-negative"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateStructuredQuery(tc.query)
			if err == nil {
				t.Fatalf("expected an error containing %q", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("error = %q, want it to contain %q", err.Error(), tc.wantErr)
			}
		})
	}
}

func TestValidateAllowed(t *testing.T) {
	expr := NewOperation(OperatorOr,
		NewComparison(ComparatorEq, "a", 1),
		NewComparison(ComparatorLike, "b", "%x%"),
	)

	// Both the OR operator and the LIKE comparator are disallowed.
	err := ValidateAllowed(expr, []Operator{OperatorAnd}, []Comparator{ComparatorEq})
	if err == nil {
		t.Fatalf("expected a disallowed-operator error")
	}
	if !strings.Contains(err.Error(), "disallowed operator") {
		t.Fatalf("error = %q, want a disallowed-operator error", err.Error())
	}

	err = ValidateAllowed(expr, []Operator{OperatorAnd, OperatorOr}, []Comparator{ComparatorEq})
	if err == nil || !strings.Contains(err.Error(), "disallowed comparator") {
		t.Fatalf("error = %v, want a disallowed-comparator error", err)
	}

	// nil allow lists permit everything.
	if err := ValidateAllowed(expr, nil, nil); err != nil {
		t.Fatalf("ValidateAllowed with nil allow lists: %v", err)
	}
}

// filterMap runs ToFilter and asserts the result is a map[string]any.
func filterMap(t *testing.T, expr Expr, visitor Visitor) map[string]any {
	t.Helper()
	result, err := ToFilter(expr, visitor)
	if err != nil {
		t.Fatalf("ToFilter: %v", err)
	}
	m, ok := result.(map[string]any)
	if !ok {
		t.Fatalf("result type = %T, want map[string]any", result)
	}
	return m
}

func TestToFilterDefaultComparison(t *testing.T) {
	cases := []struct {
		cmp    Comparator
		value  any
		wantOp string
	}{
		{ComparatorEq, "x", vectorstores.FilterEq},
		{ComparatorNe, "x", vectorstores.FilterNe},
		{ComparatorGt, 5, vectorstores.FilterGt},
		{ComparatorGte, 5, vectorstores.FilterGte},
		{ComparatorLt, 5, vectorstores.FilterLt},
		{ComparatorLte, 5, vectorstores.FilterLte},
		{ComparatorIn, []any{"a", "b"}, vectorstores.FilterIn},
		{ComparatorNin, []any{"a", "b"}, vectorstores.FilterNin},
		{ComparatorLike, "%x%", vectorstores.FilterLike},
	}
	for _, tc := range cases {
		t.Run(string(tc.cmp), func(t *testing.T) {
			m := filterMap(t, NewComparison(tc.cmp, "field", tc.value), nil)
			if len(m) != 1 {
				t.Fatalf("filter = %#v, want a single field", m)
			}
			condition, ok := m["field"].(map[string]any)
			if !ok || len(condition) != 1 {
				t.Fatalf("condition = %#v, want a single-operator map", m["field"])
			}
			got, present := condition[tc.wantOp]
			if !present {
				t.Fatalf("condition = %#v, want operator %q", condition, tc.wantOp)
			}
			if err := vectorstores.ValidateFilter(m); err != nil {
				t.Fatalf("generated filter is invalid: %v", err)
			}
			_ = got
		})
	}
}

func TestToFilterContainUnsupported(t *testing.T) {
	_, err := ToFilter(NewComparison(ComparatorContain, "field", "x"), nil)
	if err == nil || !strings.Contains(err.Error(), "cannot be represented") {
		t.Fatalf("error = %v, want an unsupported-comparator error", err)
	}
}

func TestToFilterAndMergesFields(t *testing.T) {
	expr := NewOperation(OperatorAnd,
		NewComparison(ComparatorGte, "age", 18),
		NewComparison(ComparatorEq, "name", "alice"),
	)
	m := filterMap(t, expr, nil)
	if len(m) != 2 {
		t.Fatalf("filter = %#v, want two merged fields", m)
	}
	if _, ok := m["age"]; !ok {
		t.Fatalf("filter missing age: %#v", m)
	}
	if _, ok := m["name"]; !ok {
		t.Fatalf("filter missing name: %#v", m)
	}
	if err := vectorstores.ValidateFilter(m); err != nil {
		t.Fatalf("generated filter is invalid: %v", err)
	}
}

func TestToFilterAndDuplicateFieldError(t *testing.T) {
	expr := NewOperation(OperatorAnd,
		NewComparison(ComparatorGt, "age", 10),
		NewComparison(ComparatorLt, "age", 20),
	)
	_, err := ToFilter(expr, nil)
	if err == nil || !strings.Contains(err.Error(), "duplicate field") {
		t.Fatalf("error = %v, want a duplicate-field error", err)
	}
}

func TestToFilterOrNotUnsupported(t *testing.T) {
	for _, op := range []Operator{OperatorOr, OperatorNot} {
		expr := NewOperation(op, NewComparison(ComparatorEq, "a", 1))
		_, err := ToFilter(expr, nil)
		if err == nil || !strings.Contains(err.Error(), "cannot be represented") {
			t.Fatalf("operator %q: error = %v, want an unsupported error", op, err)
		}
	}
}

func TestToFilterStructuredQuery(t *testing.T) {
	q := NewStructuredQuery("find", NewComparison(ComparatorEq, "status", "active"), 5)
	m := filterMap(t, q, nil)
	condition, ok := m["status"].(map[string]any)
	if !ok || condition[vectorstores.FilterEq] != "active" {
		t.Fatalf("filter = %#v, want a status eq condition", m)
	}

	empty := filterMap(t, NewStructuredQuery("find", nil, 0), nil)
	if len(empty) != 0 {
		t.Fatalf("filter = %#v, want an empty map for a query with no filter", empty)
	}
}

func TestToFilterConstrainedVisitor(t *testing.T) {
	expr := NewComparison(ComparatorGt, "age", 18)
	visitor := &DefaultVisitor{
		AllowedOps:  []Operator{OperatorAnd},
		AllowedCmps: []Comparator{ComparatorEq, ComparatorNe},
	}
	_, err := ToFilter(expr, visitor)
	if err == nil || !strings.Contains(err.Error(), "disallowed comparator") {
		t.Fatalf("error = %v, want a disallowed-comparator error", err)
	}

	// A permitting allow list lets the same expression through.
	visitor.AllowedCmps = append(visitor.AllowedCmps, ComparatorGt)
	m := filterMap(t, expr, visitor)
	if err := vectorstores.ValidateFilter(m); err != nil {
		t.Fatalf("generated filter is invalid: %v", err)
	}
}

func TestGeneratedFilterMatchesDocument(t *testing.T) {
	expr := NewOperation(OperatorAnd,
		NewComparison(ComparatorGte, "age", 18),
		NewComparison(ComparatorEq, "name", "alice"),
	)
	m := filterMap(t, expr, nil)

	matching := documents.Document{
		PageContent: "doc",
		Metadata:    map[string]any{"age": 30, "name": "alice"},
	}
	nonMatching := documents.Document{
		PageContent: "doc",
		Metadata:    map[string]any{"age": 12, "name": "alice"},
	}
	if !vectorstores.MatchFilter(matching, m) {
		t.Fatalf("expected matching document to match filter %#v", m)
	}
	if vectorstores.MatchFilter(nonMatching, m) {
		t.Fatalf("expected non-matching document not to match filter %#v", m)
	}
}

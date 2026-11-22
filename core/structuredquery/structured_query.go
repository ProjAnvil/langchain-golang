// Package structuredquery implements the internal representation of a
// structured query language, mirroring langchain_core.structured_query. It
// provides an expression tree (Comparison/Operation/StructuredQuery), a
// visitor-based IR translation hook, validation of operators and comparators,
// and a default visitor that emits the declarative metadata filter DSL
// understood by core/vectorstores.
package structuredquery

import (
	"fmt"
	"slices"
	"strings"

	"github.com/projanvil/langchain-golang/core/vectorstores"
)

// Operator is a logical operation over filter directives. Values match the
// Python langchain_core.structured_query.Operator enum.
type Operator string

const (
	// OperatorAnd joins filter directives with a logical AND.
	OperatorAnd Operator = "and"
	// OperatorOr joins filter directives with a logical OR.
	OperatorOr Operator = "or"
	// OperatorNot negates a filter directive.
	OperatorNot Operator = "not"
)

// Comparator is a comparison operator. Values match the Python
// langchain_core.structured_query.Comparator enum.
type Comparator string

const (
	// ComparatorEq matches equality.
	ComparatorEq Comparator = "eq"
	// ComparatorNe matches inequality.
	ComparatorNe Comparator = "ne"
	// ComparatorGt matches greater-than.
	ComparatorGt Comparator = "gt"
	// ComparatorGte matches greater-than-or-equal.
	ComparatorGte Comparator = "gte"
	// ComparatorLt matches less-than.
	ComparatorLt Comparator = "lt"
	// ComparatorLte matches less-than-or-equal.
	ComparatorLte Comparator = "lte"
	// ComparatorIn matches membership in a list.
	ComparatorIn Comparator = "in"
	// ComparatorNin matches absence from a list.
	ComparatorNin Comparator = "nin"
	// ComparatorLike matches a SQL LIKE pattern.
	ComparatorLike Comparator = "like"
	// ComparatorContain matches containment.
	ComparatorContain Comparator = "contain"
)

// allOperators and allComparators are the full sets of recognized enum
// values. They are used by structural validation and by error messages.
var allOperators = []Operator{OperatorAnd, OperatorOr, OperatorNot}

var allComparators = []Comparator{
	ComparatorEq, ComparatorNe, ComparatorGt, ComparatorGte, ComparatorLt,
	ComparatorLte, ComparatorIn, ComparatorNin, ComparatorLike, ComparatorContain,
}

func isKnownOperator(op Operator) bool {
	return slices.Contains(allOperators, op)
}

func isKnownComparator(cmp Comparator) bool {
	return slices.Contains(allComparators, cmp)
}

func operatorList() string {
	parts := make([]string, len(allOperators))
	for i, op := range allOperators {
		parts[i] = string(op)
	}
	return strings.Join(parts, ", ")
}

func comparatorList() string {
	parts := make([]string, len(allComparators))
	for i, cmp := range allComparators {
		parts[i] = string(cmp)
	}
	return strings.Join(parts, ", ")
}

// Expr is a node in a structured query expression tree. Each concrete node
// dispatches to the matching Visitor method through Accept, mirroring the
// snake_case visit_* dispatch of langchain_core.structured_query.Expr.accept
// (Comparison -> VisitComparison, Operation -> VisitOperation,
// StructuredQuery -> VisitStructuredQuery).
type Expr interface {
	Accept(visitor Visitor) (any, error)
}

// Visitor is the IR translation interface. Method names correspond to the
// snake_case visit_operation / visit_comparison / visit_structured_query
// methods of the Python Visitor base class.
type Visitor interface {
	VisitOperation(*Operation) (any, error)
	VisitComparison(*Comparison) (any, error)
	VisitStructuredQuery(*StructuredQuery) (any, error)
}

// ConstrainedVisitor is optionally implemented by visitors that only support
// a subset of operators and comparators. When a visitor implements it, ToFilter
// validates the expression against the returned allow lists before dispatch.
// A nil or empty list means "allow all". This mirrors the
// allowed_operators / allowed_comparators attributes of the Python Visitor.
type ConstrainedVisitor interface {
	Visitor
	AllowedOperators() []Operator
	AllowedComparators() []Comparator
}

// FilterDirective is a filtering expression: a Comparison or an Operation.
// A StructuredQuery's Filter is a FilterDirective. The unexported marker
// method keeps the set of directives closed to the two types in this package.
type FilterDirective interface {
	Expr
	filterDirective()
}

// Comparison compares an attribute to a value.
type Comparison struct {
	Comparator Comparator
	Attribute  string
	Value      any
}

// NewComparison builds a Comparison.
func NewComparison(comparator Comparator, attribute string, value any) *Comparison {
	return &Comparison{Comparator: comparator, Attribute: attribute, Value: value}
}

// Accept dispatches to Visitor.VisitComparison.
func (c *Comparison) Accept(visitor Visitor) (any, error) {
	return visitor.VisitComparison(c)
}

func (c *Comparison) filterDirective() {}

// Operation combines filter directives with a logical operator.
type Operation struct {
	Operator  Operator
	Arguments []FilterDirective
}

// NewOperation builds an Operation.
func NewOperation(operator Operator, arguments ...FilterDirective) *Operation {
	return &Operation{Operator: operator, Arguments: arguments}
}

// Accept dispatches to Visitor.VisitOperation.
func (o *Operation) Accept(visitor Visitor) (any, error) {
	return visitor.VisitOperation(o)
}

func (o *Operation) filterDirective() {}

// StructuredQuery is a full structured query: a free-text query string, an
// optional filter directive, and an optional result limit.
type StructuredQuery struct {
	Query  string
	Filter FilterDirective
	Limit  int
}

// NewStructuredQuery builds a StructuredQuery.
func NewStructuredQuery(query string, filter FilterDirective, limit int) *StructuredQuery {
	return &StructuredQuery{Query: query, Filter: filter, Limit: limit}
}

// Accept dispatches to Visitor.VisitStructuredQuery.
func (q *StructuredQuery) Accept(visitor Visitor) (any, error) {
	return visitor.VisitStructuredQuery(q)
}

// validateComparator checks a single comparator against an allow list and
// returns a clear error when it is disallowed. A nil allow list allows all.
// It mirrors langchain_core.structured_query.Visitor._validate_func.
func validateComparator(cmp Comparator, allowed []Comparator) error {
	if allowed == nil || slices.Contains(allowed, cmp) {
		return nil
	}
	return fmt.Errorf(
		"structuredquery: received disallowed comparator %q; allowed comparators are %s",
		cmp, formatComparators(allowed),
	)
}

// validateOperator checks a single operator against an allow list and returns
// a clear error when it is disallowed. A nil allow list allows all.
func validateOperator(op Operator, allowed []Operator) error {
	if allowed == nil || slices.Contains(allowed, op) {
		return nil
	}
	return fmt.Errorf(
		"structuredquery: received disallowed operator %q; allowed operators are %s",
		op, formatOperators(allowed),
	)
}

func formatOperators(ops []Operator) string {
	parts := make([]string, len(ops))
	for i, op := range ops {
		parts[i] = string(op)
	}
	return strings.Join(parts, ", ")
}

func formatComparators(cmps []Comparator) string {
	parts := make([]string, len(cmps))
	for i, cmp := range cmps {
		parts[i] = string(cmp)
	}
	return strings.Join(parts, ", ")
}

// validateExpr walks an expression tree, performing structural validation
// (recognized enum values, non-nil nodes, non-empty attributes, valid arity,
// non-negative limit) and, when allow lists are supplied, allowed operator /
// comparator validation. It is the single workhorse behind
// ValidateStructuredQuery, ValidateAllowed, and ToFilter.
func validateExpr(expr Expr, allowedOperators []Operator, allowedComparators []Comparator) error {
	switch node := expr.(type) {
	case nil:
		return fmt.Errorf("structuredquery: expression is nil")
	case *Comparison:
		if node == nil {
			return fmt.Errorf("structuredquery: comparison is nil")
		}
		if node.Attribute == "" {
			return fmt.Errorf("structuredquery: comparison attribute must not be empty")
		}
		if !isKnownComparator(node.Comparator) {
			return fmt.Errorf(
				"structuredquery: invalid comparator %q; expected one of %s",
				node.Comparator, comparatorList(),
			)
		}
		return validateComparator(node.Comparator, allowedComparators)
	case *Operation:
		if node == nil {
			return fmt.Errorf("structuredquery: operation is nil")
		}
		if !isKnownOperator(node.Operator) {
			return fmt.Errorf(
				"structuredquery: invalid operator %q; expected one of %s",
				node.Operator, operatorList(),
			)
		}
		if err := validateOperator(node.Operator, allowedOperators); err != nil {
			return err
		}
		if len(node.Arguments) == 0 {
			return fmt.Errorf("structuredquery: operation %q requires at least one argument", node.Operator)
		}
		if node.Operator == OperatorNot && len(node.Arguments) != 1 {
			return fmt.Errorf(
				"structuredquery: operation %q requires exactly one argument, got %d",
				node.Operator, len(node.Arguments),
			)
		}
		for i, arg := range node.Arguments {
			if err := validateExpr(arg, allowedOperators, allowedComparators); err != nil {
				return fmt.Errorf("structuredquery: operation %q argument %d: %w", node.Operator, i, err)
			}
		}
		return nil
	case *StructuredQuery:
		if node == nil {
			return fmt.Errorf("structuredquery: query is nil")
		}
		if node.Limit < 0 {
			return fmt.Errorf("structuredquery: limit must be non-negative, got %d", node.Limit)
		}
		if node.Filter != nil {
			return validateExpr(node.Filter, allowedOperators, allowedComparators)
		}
		return nil
	default:
		return fmt.Errorf("structuredquery: unsupported expression type %T", expr)
	}
}

// ValidateStructuredQuery reports whether q is a well-formed structured
// query: every operator and comparator is a recognized value, operations have
// valid arity, comparison attributes are non-empty, and the limit is
// non-negative. It returns a descriptive error on the first problem.
func ValidateStructuredQuery(q *StructuredQuery) error {
	if q == nil {
		return fmt.Errorf("structuredquery: query is nil")
	}
	return validateExpr(q, nil, nil)
}

// ValidateAllowed reports whether every operator and comparator used in expr
// is present in the provided allow lists. A nil list allows all values. It
// also performs the structural checks of ValidateStructuredQuery. This mirrors
// the allowed_operators / allowed_comparators validation of the Python
// Visitor and is useful for visitors that declare a supported subset.
func ValidateAllowed(expr Expr, allowedOperators []Operator, allowedComparators []Comparator) error {
	return validateExpr(expr, allowedOperators, allowedComparators)
}

// DefaultVisitor translates a structured query expression into the
// declarative metadata filter DSL consumed by core/vectorstores
// (ValidateFilter / MatchFilter). Its Visit methods return map[string]any.
//
// Translation is best-effort ("where possible"): comparators map to the
// matching "$"-prefixed filter operators, and AND operations merge their
// argument filters into a single map (top-level fields are implicitly ANDed).
// Comparators or operators with no DSL equivalent (contain, or, not), and
// AND branches that collide on a field name, produce a clear error instead of
// an invalid filter. The free-text Query and the Limit of a StructuredQuery
// are not representable in the filter DSL and are ignored.
type DefaultVisitor struct {
	// AllowedOps optionally restricts which logical operators are translated.
	// A nil or empty slice allows all operators.
	AllowedOps []Operator
	// AllowedCmps optionally restricts which comparators are translated.
	// A nil or empty slice allows all comparators.
	AllowedCmps []Comparator
}

// AllowedOperators implements ConstrainedVisitor.
func (v *DefaultVisitor) AllowedOperators() []Operator { return v.AllowedOps }

// AllowedComparators implements ConstrainedVisitor.
func (v *DefaultVisitor) AllowedComparators() []Comparator { return v.AllowedCmps }

// VisitComparison emits a single-field filter map for a comparison.
func (v *DefaultVisitor) VisitComparison(c *Comparison) (any, error) {
	if c == nil {
		return nil, fmt.Errorf("structuredquery: comparison is nil")
	}
	if err := validateComparator(c.Comparator, v.AllowedCmps); err != nil {
		return nil, err
	}
	filterOp, ok := comparatorToFilterOperator(c.Comparator)
	if !ok {
		return nil, fmt.Errorf(
			"structuredquery: comparator %q cannot be represented in the vectorstores filter DSL",
			c.Comparator,
		)
	}
	return validatedFilter(map[string]any{
		c.Attribute: map[string]any{filterOp: c.Value},
	})
}

// VisitOperation emits a merged filter map for AND, and an error for OR / NOT
// which have no representation in the vectorstores filter DSL.
func (v *DefaultVisitor) VisitOperation(o *Operation) (any, error) {
	if o == nil {
		return nil, fmt.Errorf("structuredquery: operation is nil")
	}
	if err := validateOperator(o.Operator, v.AllowedOps); err != nil {
		return nil, err
	}
	switch o.Operator {
	case OperatorAnd:
		merged := map[string]any{}
		for i, arg := range o.Arguments {
			result, err := arg.Accept(v)
			if err != nil {
				return nil, fmt.Errorf("structuredquery: and argument %d: %w", i, err)
			}
			partial, ok := result.(map[string]any)
			if !ok {
				return nil, fmt.Errorf(
					"structuredquery: and argument %d produced %T, want map[string]any", i, result,
				)
			}
			for field, condition := range partial {
				if _, exists := merged[field]; exists {
					return nil, fmt.Errorf(
						"structuredquery: cannot merge AND branches: duplicate field %q requires an unsupported $and operator",
						field,
					)
				}
				merged[field] = condition
			}
		}
		return validatedFilter(merged)
	case OperatorOr, OperatorNot:
		return nil, fmt.Errorf(
			"structuredquery: operator %q cannot be represented in the vectorstores filter DSL",
			o.Operator,
		)
	default:
		return nil, fmt.Errorf("structuredquery: invalid operator %q", o.Operator)
	}
}

// VisitStructuredQuery emits the filter map for q's filter directive, or an
// empty filter map when q has no filter. Query and Limit are ignored.
func (v *DefaultVisitor) VisitStructuredQuery(q *StructuredQuery) (any, error) {
	if q == nil {
		return nil, fmt.Errorf("structuredquery: query is nil")
	}
	if q.Filter == nil {
		return map[string]any{}, nil
	}
	return q.Filter.Accept(v)
}

// comparatorToFilterOperator maps a Comparator to the matching
// core/vectorstores filter operator constant. ComparatorContain has no
// equivalent and returns ok=false.
func comparatorToFilterOperator(cmp Comparator) (string, bool) {
	switch cmp {
	case ComparatorEq:
		return vectorstores.FilterEq, true
	case ComparatorNe:
		return vectorstores.FilterNe, true
	case ComparatorGt:
		return vectorstores.FilterGt, true
	case ComparatorGte:
		return vectorstores.FilterGte, true
	case ComparatorLt:
		return vectorstores.FilterLt, true
	case ComparatorLte:
		return vectorstores.FilterLte, true
	case ComparatorIn:
		return vectorstores.FilterIn, true
	case ComparatorNin:
		return vectorstores.FilterNin, true
	case ComparatorLike:
		return vectorstores.FilterLike, true
	default:
		return "", false
	}
}

// validatedFilter guarantees the produced map is a valid vectorstores filter.
func validatedFilter(filter map[string]any) (map[string]any, error) {
	if err := vectorstores.ValidateFilter(filter); err != nil {
		return nil, fmt.Errorf("structuredquery: generated filter is invalid: %w", err)
	}
	return filter, nil
}

// ToFilter is the conversion hook: it validates expr and translates it with
// visitor, returning the visitor's output. If visitor is nil, a DefaultVisitor
// is used. When visitor implements ConstrainedVisitor, its allow lists are
// enforced before dispatch. expr may be a StructuredQuery, a Comparison, or
// an Operation.
func ToFilter(expr Expr, visitor Visitor) (any, error) {
	if expr == nil {
		return nil, fmt.Errorf("structuredquery: expression is nil")
	}
	if visitor == nil {
		visitor = &DefaultVisitor{}
	}
	var allowedOperators []Operator
	var allowedComparators []Comparator
	if constrained, ok := visitor.(ConstrainedVisitor); ok {
		allowedOperators = constrained.AllowedOperators()
		allowedComparators = constrained.AllowedComparators()
	}
	if err := validateExpr(expr, allowedOperators, allowedComparators); err != nil {
		return nil, err
	}
	return expr.Accept(visitor)
}

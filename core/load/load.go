package load

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Serializable is implemented by values with a LangChain namespace and stable
// constructor payload.
type Serializable interface {
	LCNamespace() []string
	LCID() []string
	LCAttributes() map[string]any
}

// Serialized is the JSON-compatible LangChain serialization shape.
type Serialized struct {
	LC      int            `json:"lc"`
	Type    string         `json:"type"`
	ID      []string       `json:"id"`
	Attrs   map[string]any `json:"kwargs,omitempty"`
	Secrets map[string]any `json:"secrets,omitempty"`
}

// Dump converts a Serializable into a JSON-compatible payload.
func Dump(value Serializable) (Serialized, error) {
	if value == nil {
		return Serialized{}, fmt.Errorf("serializable value is nil")
	}
	id := value.LCID()
	if len(id) == 0 {
		id = value.LCNamespace()
	}
	if len(id) == 0 {
		return Serialized{}, fmt.Errorf("serializable id is required")
	}
	return Serialized{LC: 1, Type: "constructor", ID: cloneStrings(id), Attrs: cloneMap(value.LCAttributes())}, nil
}

// Dumps serializes a Serializable to JSON.
func Dumps(value Serializable) ([]byte, error) {
	payload, err := Dump(value)
	if err != nil {
		return nil, err
	}
	return json.Marshal(payload)
}

// Loader constructs a Go value from serialized attributes.
type Loader func(map[string]any) (any, error)

// Registry maps serialized IDs to loaders.
type Registry struct {
	loaders map[string]Loader
}

func NewRegistry() *Registry {
	return &Registry{loaders: map[string]Loader{}}
}

func (r *Registry) Register(id []string, loader Loader) error {
	if len(id) == 0 {
		return fmt.Errorf("id is required")
	}
	if loader == nil {
		return fmt.Errorf("loader is required")
	}
	r.loaders[strings.Join(id, "/")] = loader
	return nil
}

func (r *Registry) Load(payload Serialized) (any, error) {
	if payload.LC != 1 || payload.Type != "constructor" {
		return nil, fmt.Errorf("unsupported serialized payload")
	}
	loader := r.loaders[strings.Join(payload.ID, "/")]
	if loader == nil {
		return nil, fmt.Errorf("unknown serialized id: %s", strings.Join(payload.ID, "/"))
	}
	return loader(cloneMap(payload.Attrs))
}

func cloneStrings(values []string) []string {
	out := make([]string, len(values))
	copy(out, values)
	return out
}

func cloneMap(values map[string]any) map[string]any {
	if values == nil {
		return nil
	}
	out := make(map[string]any, len(values))
	for key, value := range values {
		out[key] = value
	}
	return out
}

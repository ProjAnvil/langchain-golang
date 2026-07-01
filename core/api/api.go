package api

import (
	"fmt"
	"strings"
)

// Status marks the support level of a public API surface.
type Status string

const (
	StatusStable     Status = "stable"
	StatusBeta       Status = "beta"
	StatusDeprecated Status = "deprecated"
)

// Deprecation describes a deprecated symbol and how callers should migrate.
type Deprecation struct {
	Name        string `json:"name"`
	Since       string `json:"since,omitempty"`
	Removal     string `json:"removal,omitempty"`
	Alternative string `json:"alternative,omitempty"`
	Message     string `json:"message,omitempty"`
}

// Error returns the formatted deprecation message.
func (d Deprecation) Error() string {
	return d.Format()
}

// Format returns a deterministic user-facing deprecation message.
func (d Deprecation) Format() string {
	name := d.Name
	if name == "" {
		name = "This API"
	}
	parts := []string{fmt.Sprintf("%s is deprecated", name)}
	if d.Since != "" {
		parts = append(parts, "since "+d.Since)
	}
	if d.Removal != "" {
		parts = append(parts, "and will be removed in "+d.Removal)
	}
	msg := strings.Join(parts, " ")
	if d.Alternative != "" {
		msg += "; use " + d.Alternative + " instead"
	}
	if d.Message != "" {
		msg += ". " + d.Message
	}
	return msg
}

// Metadata is lightweight public metadata attached to migrated APIs.
type Metadata struct {
	Status      Status       `json:"status"`
	Name        string       `json:"name,omitempty"`
	Deprecation *Deprecation `json:"deprecation,omitempty"`
	Message     string       `json:"message,omitempty"`
}

// Stable returns stable API metadata.
func Stable(name string) Metadata {
	return Metadata{Status: StatusStable, Name: name}
}

// Beta returns beta API metadata.
func Beta(name string, message string) Metadata {
	return Metadata{Status: StatusBeta, Name: name, Message: message}
}

// Deprecated returns deprecated API metadata.
func Deprecated(deprecation Deprecation) Metadata {
	return Metadata{
		Status:      StatusDeprecated,
		Name:        deprecation.Name,
		Deprecation: &deprecation,
		Message:     deprecation.Format(),
	}
}

// IsInternalPath reports whether a LangChain import path is internal/private.
func IsInternalPath(path string) bool {
	for _, part := range strings.Split(path, "/") {
		if strings.HasPrefix(part, "_") && part != "_" {
			return true
		}
	}
	return false
}

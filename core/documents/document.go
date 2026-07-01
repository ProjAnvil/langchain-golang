package documents

// Document is the shared representation for retrievers, vector stores, loaders,
// and transformers.
type Document struct {
	ID          string         `json:"id,omitempty"`
	PageContent string         `json:"page_content"`
	Metadata    map[string]any `json:"metadata,omitempty"`
}

// New creates a document with page content and optional metadata.
func New(pageContent string, metadata map[string]any) Document {
	return Document{
		PageContent: pageContent,
		Metadata:    cloneMetadata(metadata),
	}
}

// WithID returns a copy of the document with an ID.
func (d Document) WithID(id string) Document {
	d.ID = id
	return d
}

// WithMetadata returns a copy of the document with defensive-copied metadata.
func (d Document) WithMetadata(metadata map[string]any) Document {
	d.Metadata = cloneMetadata(metadata)
	return d
}

// Source returns metadata["source"] when it is a string.
func (d Document) Source() string {
	source, _ := d.Metadata["source"].(string)
	return source
}

// MetadataValue returns one metadata value.
func (d Document) MetadataValue(key string) (any, bool) {
	value, ok := d.Metadata[key]
	return value, ok
}

// Clone returns a defensive copy of the document.
func (d Document) Clone() Document {
	d.Metadata = cloneMetadata(d.Metadata)
	return d
}

func cloneMetadata(metadata map[string]any) map[string]any {
	if metadata == nil {
		return map[string]any{}
	}
	out := make(map[string]any, len(metadata))
	for key, value := range metadata {
		out[key] = value
	}
	return out
}

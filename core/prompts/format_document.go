package prompts

import (
	"fmt"
	"sort"

	"github.com/projanvil/langchain-golang/core/documents"
)

// DocumentPrompt is the minimal prompt interface needed to format a document.
// It is satisfied by PromptTemplate: input variables come from the template
// and Format renders the resolved values. Using an interface keeps the
// formatting helpers usable with any prompt type exposing the same methods.
type DocumentPrompt interface {
	InputVariables() []string
	Format(values map[string]any) (string, error)
}

// GetDocumentInfo extracts the values a document prompt needs from a document,
// mirroring langchain_core.prompts.base._get_document_info. The document's
// page content is exposed as "page_content" and its metadata keys are exposed
// under their own names; metadata keys may override "page_content".
//
// If any of the prompt's input variables cannot be resolved from the document,
// GetDocumentInfo returns a deterministic error listing the required and
// missing metadata. The returned map contains exactly the prompt's input
// variables.
func GetDocumentInfo(doc documents.Document, prompt DocumentPrompt) (map[string]any, error) {
	if prompt == nil {
		return nil, fmt.Errorf("document prompt is required")
	}
	baseInfo := map[string]any{"page_content": doc.PageContent}
	for key, value := range doc.Metadata {
		baseInfo[key] = value
	}

	inputVariables := prompt.InputVariables()
	missing := []string{}
	for _, variable := range inputVariables {
		if _, ok := baseInfo[variable]; !ok {
			missing = append(missing, variable)
		}
	}
	if len(missing) > 0 {
		sort.Strings(missing)
		required := make([]string, 0, len(inputVariables))
		for _, variable := range inputVariables {
			if variable != "page_content" {
				required = append(required, variable)
			}
		}
		sort.Strings(required)
		return nil, fmt.Errorf(
			"document prompt requires documents to have metadata variables: %v. Received document with missing metadata: %v",
			required, missing,
		)
	}

	info := make(map[string]any, len(inputVariables))
	for _, variable := range inputVariables {
		info[variable] = baseInfo[variable]
	}
	return info, nil
}

// FormatDocument renders a document into a string using a prompt template,
// mirroring langchain_core.prompts.base.format_document. It resolves the
// prompt's input variables from the document's page content and metadata and
// then formats the prompt. Missing metadata returns a deterministic error.
func FormatDocument(doc documents.Document, prompt DocumentPrompt) (string, error) {
	info, err := GetDocumentInfo(doc, prompt)
	if err != nil {
		return "", err
	}
	return prompt.Format(info)
}

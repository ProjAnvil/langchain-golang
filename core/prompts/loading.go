package prompts

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// TextPrompt formats a text prompt from input variables.
type TextPrompt interface {
	Format(values map[string]any) (string, error)
}

// LoadPrompt reads a local JSON prompt config and constructs a text prompt.
func LoadPrompt(path string, allowDangerousPaths bool) (TextPrompt, error) {
	if strings.HasPrefix(path, "lc://") {
		return nil, fmt.Errorf("loading deprecated lc:// hub prompts is not supported")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var config map[string]any
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("load prompt JSON: %w", err)
	}
	baseDir := filepath.Dir(path)
	return LoadPromptFromConfig(config, LoadPromptOptions{
		BaseDir:             baseDir,
		AllowDangerousPaths: allowDangerousPaths,
	})
}

// LoadPromptOptions configures LoadPromptFromConfig.
type LoadPromptOptions struct {
	BaseDir             string
	AllowDangerousPaths bool
}

// LoadPromptFromConfig constructs a prompt from a JSON-shaped config.
func LoadPromptFromConfig(config map[string]any, opts LoadPromptOptions) (TextPrompt, error) {
	copied := cloneMapAny(config)
	typ, _ := copied["_type"].(string)
	if typ == "" {
		typ = "prompt"
	}
	switch typ {
	case "prompt":
		return loadPromptTemplate(copied, opts)
	case "few_shot":
		return loadFewShotPrompt(copied, opts)
	default:
		return nil, fmt.Errorf("loading %s prompt not supported", typ)
	}
}

func loadPromptTemplate(config map[string]any, opts LoadPromptOptions) (PromptTemplate, error) {
	if err := loadTemplateValue(config, "template", opts); err != nil {
		return PromptTemplate{}, err
	}
	if format, _ := config["template_format"].(string); format == "jinja2" {
		return PromptTemplate{}, fmt.Errorf("loading templates with jinja2 is not supported")
	}
	templateText, _ := config["template"].(string)
	if templateText == "" {
		return PromptTemplate{}, fmt.Errorf("prompt template is required")
	}
	name, _ := config["name"].(string)
	partials, _ := config["partial_variables"].(map[string]any)
	return NewPromptTemplateWithPartials(name, templateText, partials)
}

func loadFewShotPrompt(config map[string]any, opts LoadPromptOptions) (FewShotPromptTemplate, error) {
	if err := loadTemplateValue(config, "prefix", opts); err != nil {
		return FewShotPromptTemplate{}, err
	}
	if err := loadTemplateValue(config, "suffix", opts); err != nil {
		return FewShotPromptTemplate{}, err
	}

	rawExamplePrompt, ok := config["example_prompt"].(map[string]any)
	if !ok {
		return FewShotPromptTemplate{}, fmt.Errorf("few-shot example_prompt is required")
	}
	loadedExamplePrompt, err := LoadPromptFromConfig(rawExamplePrompt, opts)
	if err != nil {
		return FewShotPromptTemplate{}, err
	}
	examplePrompt, ok := loadedExamplePrompt.(PromptTemplate)
	if !ok {
		return FewShotPromptTemplate{}, fmt.Errorf("few-shot example_prompt must be a prompt template")
	}

	examples, err := loadExamples(config["examples"], opts)
	if err != nil {
		return FewShotPromptTemplate{}, err
	}
	prefix, _ := config["prefix"].(string)
	suffix, _ := config["suffix"].(string)
	separator, _ := config["example_separator"].(string)
	return NewFewShotPromptTemplate(examples, nil, examplePrompt, prefix, suffix, separator)
}

func loadTemplateValue(config map[string]any, name string, opts LoadPromptOptions) error {
	pathValue, hasPath := config[name+"_path"].(string)
	if !hasPath || pathValue == "" {
		return nil
	}
	if _, exists := config[name]; exists {
		return fmt.Errorf("both %s_path and %s cannot be provided", name, name)
	}
	path, err := resolvePromptPath(pathValue, opts)
	if err != nil {
		return err
	}
	if filepath.Ext(path) != ".txt" {
		return fmt.Errorf("unsupported template file format %q", filepath.Ext(path))
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	config[name] = string(data)
	return nil
}

func loadExamples(value any, opts LoadPromptOptions) ([]map[string]any, error) {
	switch typed := value.(type) {
	case []map[string]any:
		return cloneExamples(typed), nil
	case []any:
		out := make([]map[string]any, len(typed))
		for i, item := range typed {
			example, ok := item.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("example %d must be an object", i)
			}
			out[i] = cloneMapAny(example)
		}
		return out, nil
	case string:
		path, err := resolvePromptPath(typed, opts)
		if err != nil {
			return nil, err
		}
		if filepath.Ext(path) != ".json" {
			return nil, fmt.Errorf("invalid examples file format; only json is supported")
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		var examples []map[string]any
		if err := json.Unmarshal(data, &examples); err != nil {
			return nil, err
		}
		return cloneExamples(examples), nil
	default:
		return nil, fmt.Errorf("invalid examples format")
	}
}

func resolvePromptPath(path string, opts LoadPromptOptions) (string, error) {
	clean := filepath.Clean(path)
	if !opts.AllowDangerousPaths {
		if filepath.IsAbs(clean) {
			return "", fmt.Errorf("absolute prompt paths are not allowed")
		}
		for _, part := range strings.Split(clean, string(filepath.Separator)) {
			if part == ".." {
				return "", fmt.Errorf("prompt path traversal is not allowed")
			}
		}
	}
	if !filepath.IsAbs(clean) && opts.BaseDir != "" {
		clean = filepath.Join(opts.BaseDir, clean)
	}
	return clean, nil
}

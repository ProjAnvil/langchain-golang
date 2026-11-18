// Command tutorials_tnt-llm_tnt-llm is the Go port of
// examples/tutorials/tnt-llm/tnt-llm.ipynb.
//
// TNT-LLM builds a taxonomy of user intents through a plan/search/reflect
// cycle: summarize documents, split them into minibatches, generate a
// taxonomy, iteratively update it against more minibatches, and review the
// result. The notebook pulls prompts from LangSmith Hub, calls Anthropic
// models, samples real traces and fits a scikit-learn classifier. This port
// keeps the generate -> update -> (review|update) graph and replaces every
// external dependency with a deterministic in-memory corpus and an offline
// ChatModel double that emits XML taxonomy text.
package main

import (
	"context"
	"fmt"
	"log"
	"regexp"
	"strings"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	checkpoint "github.com/projanvil/langchain-golang/langgraph/checkpoint"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// scriptedModel is a deterministic offline ChatModel double.
type scriptedModel struct {
	answer func(ctx context.Context, input []messages.Message) (messages.Message, error)
}

func (m *scriptedModel) Invoke(ctx context.Context, input []messages.Message, _ ...runnables.Option) (messages.Message, error) {
	return m.answer(ctx, input)
}
func (m *scriptedModel) Batch(ctx context.Context, inputs [][]messages.Message, opts ...runnables.Option) ([]messages.Message, error) {
	out := make([]messages.Message, len(inputs))
	for i, in := range inputs {
		msg, err := m.Invoke(ctx, in, opts...)
		if err != nil {
			return nil, err
		}
		out[i] = msg
	}
	return out, nil
}
func (m *scriptedModel) Stream(ctx context.Context, input []messages.Message, opts ...runnables.Option) (runnables.Stream[messages.Message], error) {
	msg, err := m.Invoke(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	return runnables.NewSliceStream([]messages.Message{msg}), nil
}
func (m *scriptedModel) InputSchema() schema.Schema { return schema.Object(map[string]schema.Schema{}) }
func (m *scriptedModel) OutputSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{})
}
func (m *scriptedModel) BindTools(_ []coretools.Tool) (language.ChatModel, error) {
	return m, nil
}
func (m *scriptedModel) Capabilities() language.ChatModelCapabilities {
	return language.ChatModelCapabilities{StructuredOutput: true}
}

type doc struct {
	ID          string
	Content     string
	Summary     string
	Explanation string
}

type cluster struct {
	ID          string
	Name        string
	Description string
}

func (c cluster) String() string {
	return fmt.Sprintf("%s | %s | %s", c.ID, c.Name, c.Description)
}

// corpus is the deterministic stand-in for LangSmith trace documents.
var corpus = []doc{
	{ID: "d1", Content: "How do I retrieve documents for a RAG chain?"},
	{ID: "d2", Content: "What is the best chunking strategy for retrieval?"},
	{ID: "d3", Content: "Why does my long context prompt get truncated?"},
	{ID: "d4", Content: "How should I evaluate answer quality?"},
	{ID: "d5", Content: "Can I fine-tune a model on my own data?"},
	{ID: "d6", Content: "How do I keep model output safe and grounded?"},
}

var summaryByKeyword = []struct {
	Keyword string
	Summary string
}{
	{"retriev", "question about document retrieval"},
	{"chunk", "question about text chunking"},
	{"context", "question about context-window limits"},
	{"evaluat", "question about evaluation"},
	{"fine-tun", "question about fine-tuning"},
	{"safe", "question about safety and grounding"},
}

// summarizeDoc is the offline replacement for the LangSmith-hosted summary
// prompt plus Anthropic haiku call.
func summarizeDoc(d doc) doc {
	for _, rule := range summaryByKeyword {
		if strings.Contains(strings.ToLower(d.Content), rule.Keyword) {
			d.Summary = rule.Summary
			d.Explanation = "Identified by keyword '" + rule.Keyword + "' in the offline corpus."
			return d
		}
	}
	d.Summary = "general question about language-model applications"
	d.Explanation = "No specific keyword matched; assigned the general category."
	return d
}

var categoryByName = []struct {
	Keyword  string
	Name     string
	Scenario string
}{
	{"retriev", "Document retrieval", "finding relevant documents"},
	{"chunk", "Text chunking", "splitting source text"},
	{"context", "Context management", "managing token limits"},
	{"evaluat", "Evaluation", "measuring answer quality"},
	{"fine-tun", "Fine tuning", "adapting a model"},
	{"safe", "Safety", "grounding and moderation"},
}

func categoryFor(text string) (string, string) {
	for _, rule := range categoryByName {
		if strings.Contains(strings.ToLower(text), rule.Keyword) {
			return rule.Name, rule.Scenario
		}
	}
	return "Other", "general assistance"
}

var taxaPattern = regexp.MustCompile(`(?s)<id>(.*?)</id>\s*<name>(.*?)</name>\s*<description>(.*?)</description>`)

// parseTaxa mirrors the notebook's regex-based XML taxonomy parser.
func parseTaxa(text string) []cluster {
	matches := taxaPattern.FindAllStringSubmatch(text, -1)
	out := make([]cluster, 0, len(matches))
	for _, m := range matches {
		out = append(out, cluster{
			ID:          strings.TrimSpace(m[1]),
			Name:        strings.TrimSpace(m[2]),
			Description: strings.TrimSpace(m[3]),
		})
	}
	return out
}

// taxonomyChain is the offline analogue of the prompt -> LLM -> parse chain.
// stage is one of "generate", "update", or "review"; it only prefixes IDs.
func taxonomyChain(stage string) *scriptedModel {
	return &scriptedModel{answer: func(_ context.Context, input []messages.Message) (messages.Message, error) {
		text := ""
		if len(input) > 0 {
			text = messages.Text(input[len(input)-1])
		}
		seen := map[string]bool{}
		var b strings.Builder
		idx := 0
		for _, rule := range categoryByName {
			if !strings.Contains(strings.ToLower(text), rule.Keyword) || seen[rule.Name] {
				continue
			}
			seen[rule.Name] = true
			idx++
			fmt.Fprintf(&b, "<id>%s-%d</id>\n<name>%s</name>\n<description>Offline cluster for %s.</description>\n",
				stage, idx, rule.Name, rule.Scenario)
		}
		if idx == 0 {
			b.WriteString("<id>" + stage + "-1</id>\n<name>Other</name>\n<description>General offline intent.</description>\n")
		}
		return messages.AI(b.String()), nil
	}}
}

func docsOf(state map[string]any) []doc {
	docs, _ := state["documents"].([]doc)
	return docs
}

func clustersOf(state map[string]any) [][]cluster {
	c, _ := state["clusters"].([][]cluster)
	return c
}

func batchesOf(state map[string]any) [][]int {
	b, _ := state["minibatches"].([][]int)
	return b
}

// formatDocs mirrors the notebook's <conversations> XML builder.
func formatDocs(docs []doc) string {
	var b strings.Builder
	b.WriteString("<conversations>\n")
	for _, d := range docs {
		fmt.Fprintf(&b, "<conv_summ id=%s>%s</conv_summ>\n", d.ID, d.Summary)
	}
	b.WriteString("</conversations>")
	return b.String()
}

// formatTaxonomy mirrors the notebook's <cluster_table> XML builder.
func formatTaxonomy(clusters []cluster) string {
	var b strings.Builder
	b.WriteString("<cluster_table>\n")
	for _, c := range clusters {
		fmt.Fprintf(&b, "  <cluster>\n    <id>%s</id>\n    <name>%s</name>\n    <description>%s</description>\n  </cluster>\n", c.ID, c.Name, c.Description)
	}
	b.WriteString("</cluster_table>")
	return b.String()
}

// invokeTaxonomy builds the prompt XML, runs the offline chain and returns
// one new taxonomy version to be appended to the `clusters` channel.
func invokeTaxonomy(ctx context.Context, model *scriptedModel, state map[string]any, indices []int) ([][]cluster, error) {
	docs := docsOf(state)
	selected := make([]doc, 0, len(indices))
	for _, i := range indices {
		if i >= 0 && i < len(docs) {
			selected = append(selected, docs[i])
		}
	}
	previous := []cluster{}
	versions := clustersOf(state)
	if len(versions) > 0 {
		previous = versions[len(versions)-1]
	}
	prompt := formatDocs(selected) + "\n" + formatTaxonomy(previous)
	resp, err := model.Invoke(ctx, []messages.Message{messages.Human(prompt)})
	if err != nil {
		return nil, err
	}
	taxonomy := parseTaxa(messages.Text(resp))
	return [][]cluster{taxonomy}, nil
}

func buildGraph(batchSize int) *graphpkg.CompiledGraph {
	generateModel := taxonomyChain("generate")
	updateModel := taxonomyChain("update")
	reviewModel := taxonomyChain("review")

	g := graphpkg.NewStateGraph()
	// operator.add on clusters in Python -> AppendSliceReducer in Go.
	g.AddReducer("clusters", channels.AppendSliceReducer)

	g.AddNode("summarize", func(_ runtime.Runtime, state map[string]any) (any, error) {
		fmt.Println("---SUMMARIZE (offline map-reduce)---")
		docs := docsOf(state)
		out := make([]doc, len(docs))
		for i, d := range docs {
			out[i] = summarizeDoc(d)
			fmt.Printf("  %s: %s\n", out[i].ID, out[i].Summary)
		}
		return map[string]any{"documents": out}, nil
	})

	g.AddNode("get_minibatches", func(_ runtime.Runtime, state map[string]any) (any, error) {
		fmt.Println("---GET MINIBATCHES---")
		n := len(docsOf(state))
		batches := [][]int{}
		for start := 0; start < n; start += batchSize {
			end := start + batchSize
			if end > n {
				end = n
			}
			indices := make([]int, 0, end-start)
			for i := start; i < end; i++ {
				indices = append(indices, i)
			}
			batches = append(batches, indices)
		}
		fmt.Printf("  %d minibatches of size <= %d\n", len(batches), batchSize)
		return map[string]any{"minibatches": batches}, nil
	})

	g.AddNode("generate_taxonomy", func(rt runtime.Runtime, state map[string]any) (any, error) {
		fmt.Println("---GENERATE TAXONOMY---")
		batches := batchesOf(state)
		if len(batches) == 0 {
			return nil, fmt.Errorf("generate_taxonomy: no minibatches")
		}
		taxonomy, err := invokeTaxonomy(rt, generateModel, state, batches[0])
		if err != nil {
			return nil, err
		}
		fmt.Printf("  %d clusters\n", len(taxonomy[0]))
		return map[string]any{"clusters": taxonomy}, nil
	})

	g.AddNode("update_taxonomy", func(rt runtime.Runtime, state map[string]any) (any, error) {
		fmt.Println("---UPDATE TAXONOMY---")
		batches := batchesOf(state)
		versions := clustersOf(state)
		which := len(versions) % len(batches)
		taxonomy, err := invokeTaxonomy(rt, updateModel, state, batches[which])
		if err != nil {
			return nil, err
		}
		fmt.Printf("  minibatch %d -> %d clusters (version %d)\n", which, len(taxonomy[0]), len(versions)+1)
		return map[string]any{"clusters": taxonomy}, nil
	})

	g.AddNode("review_taxonomy", func(rt runtime.Runtime, state map[string]any) (any, error) {
		fmt.Println("---REVIEW TAXONOMY---")
		docs := docsOf(state)
		indices := make([]int, len(docs))
		for i := range indices {
			indices[i] = i
		}
		taxonomy, err := invokeTaxonomy(rt, reviewModel, state, indices)
		if err != nil {
			return nil, err
		}
		fmt.Printf("  reviewed %d clusters\n", len(taxonomy[0]))
		return map[string]any{"clusters": taxonomy}, nil
	})

	g.AddEdge(types.START, "summarize")
	g.AddEdge("summarize", "get_minibatches")
	g.AddEdge("get_minibatches", "generate_taxonomy")
	g.AddEdge("generate_taxonomy", "update_taxonomy")
	g.AddConditionalEdges("update_taxonomy", func(_ runtime.Runtime, state map[string]any) ([]any, error) {
		if len(clustersOf(state)) < len(batchesOf(state)) {
			return graphpkg.To("update_taxonomy"), nil
		}
		return graphpkg.To("review_taxonomy"), nil
	})
	g.AddEdge("review_taxonomy", types.END)

	cg, err := g.Compile(graphpkg.WithCheckpointer(checkpoint.NewMemorySaver()))
	if err != nil {
		log.Fatalf("compile tnt-llm graph: %v", err)
	}
	return cg
}

func main() {
	ctx := context.Background()
	cg := buildGraph(2)
	result, err := cg.InvokeWithOptions(ctx, map[string]any{
		"documents": corpus,
	}, graphpkg.Options{ThreadID: "tnt-thread"})
	if err != nil {
		log.Fatalf("invoke tnt-llm: %v", err)
	}
	versions := clustersOf(result.Values)
	if len(versions) == 0 {
		log.Fatal("no taxonomy produced")
	}
	final := versions[len(versions)-1]
	fmt.Println("\n## Final Taxonomy")
	fmt.Println("| ID | Name | Description |")
	fmt.Println("|----|------|-------------|")
	for _, c := range final {
		fmt.Printf("| %s | %s | %s |\n", c.ID, c.Name, c.Description)
	}
	fmt.Printf("\ntaxonomy versions: %d\n", len(versions))
}

// Command storm_storm is the Go port of examples/storm/storm.ipynb.
//
// STORM (Structured Topic-based Open-ended Research Methodology) builds a
// Wikipedia article by generating an outline, simulating expert interviews,
// refining the outline from those interviews, indexing references, drafting
// sections, and assembling the final article. The notebook chains several
// hosted models, Wikipedia/DuckDuckGo search, a vector store and an interview
// subgraph. This port keeps the graph shape:
//
//	init_research -> conduct_interviews (nested interview graph with a
//	conditional question/answer loop) -> refine_outline -> index_references
//	-> write_sections -> write_article
//
// Every external resource is replaced by a deterministic in-memory corpus.
// NO network call, search, embedding service, or hosted model is used.
package main

import (
	"context"
	"fmt"
	"log"
	"sort"
	"strings"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	checkpoint "github.com/projanvil/langchain-golang/langgraph/checkpoint"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// corpus is the offline stand-in for WikipediaRetriever and the vector store.
var corpus = []struct {
	URL      string
	Title    string
	Summary  string
	Category string
}{
	{"https://en.wikipedia.org/wiki/Large_language_model", "Large language model", "A large language model is a neural network trained on vast text corpora to predict and generate text.", "Machine learning"},
	{"https://en.wikipedia.org/wiki/Retrieval-augmented_generation", "Retrieval-augmented generation", "RAG grounds model answers in documents retrieved from an external index before generation.", "Natural language processing"},
	{"https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)", "Transformer", "The transformer architecture relies on self-attention to relate distant tokens in a sequence.", "Deep learning"},
	{"https://en.wikipedia.org/wiki/Context_window", "Context window", "The context window is the maximum number of tokens a model can consider in one pass.", "Machine learning"},
}

// subsection, section and outline mirror the notebook's Pydantic models.
type subsection struct {
	Title       string
	Description string
}

func (s subsection) String() string {
	return fmt.Sprintf("### %s\n\n%s", s.Title, s.Description)
}

type section struct {
	Title       string
	Description string
	Subsections []subsection
}

func (s section) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "## %s\n\n%s", s.Title, s.Description)
	for _, sub := range s.Subsections {
		b.WriteString("\n\n" + sub.String())
	}
	return b.String()
}

type outline struct {
	PageTitle string
	Sections  []section
}

func (o outline) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "# %s\n", o.PageTitle)
	for _, s := range o.Sections {
		b.WriteString("\n\n" + s.String())
	}
	return b.String()
}

type editor struct {
	Name        string
	Role        string
	Affiliation string
	Focus       string
}

func (e editor) Persona() string {
	return fmt.Sprintf("Name: %s\nRole: %s\nAffiliation: %s\nFocus: %s", e.Name, e.Role, e.Affiliation, e.Focus)
}

type wikiSection struct {
	Title     string
	Content   string
	Citations []string
}

func (s wikiSection) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "## %s\n\n%s", s.Title, s.Content)
	for i, url := range s.Citations {
		fmt.Fprintf(&b, "\n [%d] %s", i+1, url)
	}
	return b.String()
}

// generateOutline is the offline replacement for generate_outline_direct.
func generateOutline(topic string) outline {
	return outline{
		PageTitle: topic,
		Sections: []section{
			{Title: "Background", Description: "Defines the topic and its place in modern language-model research.", Subsections: []subsection{{Title: "Definitions", Description: "Core terminology for " + topic + "."}}},
			{Title: "Key developments", Description: "Summarizes the architectural and retrieval techniques behind " + topic + ".", Subsections: []subsection{{Title: "Architecture", Description: "Attention-based models and long-context training."}}},
			{Title: "Impact", Description: "Describes practical consequences and open questions."},
		},
	}
}

// surveyEditors is the offline replacement for survey_subjects.
func surveyEditors(topic string) []editor {
	return []editor{
		{Name: "Context_Researcher", Role: "Context-window researcher", Affiliation: "Uni A", Focus: "How context-window size changes retrieval strategies for " + topic + "."},
		{Name: "Retrieval_Editor", Role: "RAG editor", Affiliation: "Wiki Project", Focus: "How retrieval-grounded generation should be described for " + topic + "."},
	}
}

// answerFromCorpus returns a deterministic expert answer and its citation.
func answerFromCorpus(turn int) (string, string, string) {
	doc := corpus[turn%len(corpus)]
	return doc.Title, doc.Summary + " (" + doc.Category + ")", doc.URL
}

func mergeReferences(existing any, update any) (any, error) {
	out := map[string]string{}
	if m, ok := existing.(map[string]string); ok {
		for k, v := range m {
			out[k] = v
		}
	}
	if m, ok := update.(map[string]string); ok {
		for k, v := range m {
			out[k] = v
		}
	}
	return out, nil
}

func msgsOf(state map[string]any) []messages.Message {
	msgs, _ := state["messages"].([]messages.Message)
	return msgs
}

func intOf(state map[string]any, key string) int {
	v, _ := state[key].(int)
	return v
}

// buildInterviewGraph mirrors the notebook's interview_graph: ask_question
// and answer_question are connected by a conditional loop that stops after a
// fixed number of turns or when the interviewer says thank you.
func buildInterviewGraph(maxTurns int) *graphpkg.CompiledGraph {
	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddReducer("references", mergeReferences)

	g.AddNode("ask_question", func(_ runtime.Runtime, state map[string]any) (any, error) {
		e, _ := state["editor"].(editor)
		turns := intOf(state, "turns")
		content := fmt.Sprintf("Question %d from %s: what should a Wikipedia article say about this aspect?", turns+1, e.Role)
		if turns >= maxTurns {
			content = "Thank you so much for your help!"
		}
		msg := messages.AI(content)
		msg.Name = e.Name
		// The messages reducer appends, so return only this new message.
		return map[string]any{"messages": []messages.Message{msg}}, nil
	})

	g.AddNode("answer_question", func(_ runtime.Runtime, state map[string]any) (any, error) {
		turns := intOf(state, "turns")
		title, answer, url := answerFromCorpus(turns)
		return map[string]any{
			"messages":   []messages.Message{messages.Human(fmt.Sprintf("[%s] %s", title, answer))},
			"turns":      turns + 1,
			"references": map[string]string{url: answer},
		}, nil
	})

	g.AddEdge(types.START, "ask_question")
	g.AddEdge("ask_question", "answer_question")
	g.AddConditionalEdges("answer_question", func(_ runtime.Runtime, state map[string]any) ([]any, error) {
		msgs := msgsOf(state)
		last := ""
		for i := len(msgs) - 1; i >= 0; i-- {
			if msgs[i].Role == messages.RoleAI {
				last = messages.Text(msgs[i])
				break
			}
		}
		turns := intOf(state, "turns")
		if strings.HasSuffix(last, "Thank you so much for your help!") || turns >= maxTurns {
			return graphpkg.To(types.END), nil
		}
		return graphpkg.To("ask_question"), nil
	})

	cg, err := g.Compile()
	if err != nil {
		log.Fatalf("compile interview graph: %v", err)
	}
	return cg
}

func newStormGraph(interviews *graphpkg.CompiledGraph) *graphpkg.CompiledGraph {
	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)

	g.AddNode("init_research", func(_ runtime.Runtime, state map[string]any) (any, error) {
		fmt.Println("---INIT RESEARCH---")
		topic, _ := state["topic"].(string)
		o := generateOutline(topic)
		fmt.Printf("outline: %s (%d sections)\n", o.PageTitle, len(o.Sections))
		return map[string]any{"outline": o, "editors": surveyEditors(topic)}, nil
	})

	g.AddNode("conduct_interviews", func(rt runtime.Runtime, state map[string]any) (any, error) {
		fmt.Println("---CONDUCT INTERVIEWS---")
		topic, _ := state["topic"].(string)
		editors, _ := state["editors"].([]editor)
		results := make([]map[string]any, 0, len(editors))
		allRefs := map[string]string{}
		for i, e := range editors {
			input := map[string]any{
				"editor":     e,
				"turns":      0,
				"references": map[string]string{},
				"messages": []messages.Message{messages.AI(fmt.Sprintf(
					"So you said you were writing an article on %s?", topic))},
			}
			out, err := interviews.InvokeWithOptions(rt, input, graphpkg.Options{ThreadID: fmt.Sprintf("interview-%d", i)})
			if err != nil {
				return nil, err
			}
			refs, _ := out.Values["references"].(map[string]string)
			for k, v := range refs {
				allRefs[k] = v
			}
			results = append(results, map[string]any{"editor": e, "messages": msgsOf(out.Values)})
			fmt.Printf("  %s: %d messages, %d references\n", e.Name, len(msgsOf(out.Values)), len(refs))
		}
		return map[string]any{"interviewResults": results, "references": allRefs}, nil
	})

	g.AddNode("refine_outline", func(_ runtime.Runtime, state map[string]any) (any, error) {
		fmt.Println("---REFINE OUTLINE---")
		o, _ := state["outline"].(outline)
		o.Sections = append(o.Sections, section{
			Title:       "Expert perspectives",
			Description: "Insights gathered from simulated subject-matter expert interviews.",
			Subsections: []subsection{{Title: "Interview highlights", Description: "Context-window and retrieval viewpoints synthesized from the corpus."}},
		})
		fmt.Printf("refined outline has %d sections\n", len(o.Sections))
		return map[string]any{"outline": o}, nil
	})

	g.AddNode("index_references", func(_ runtime.Runtime, state map[string]any) (any, error) {
		fmt.Println("---INDEX REFERENCES---")
		refs, _ := state["references"].(map[string]string)
		urls := make([]string, 0, len(refs))
		for k := range refs {
			urls = append(urls, k)
		}
		sort.Strings(urls)
		var b strings.Builder
		for _, k := range urls {
			fmt.Fprintf(&b, "<Document href=\"%s\"/>\n%s\n</Document>\n", k, refs[k])
		}
		fmt.Printf("indexed %d documents (offline vector store)\n", len(urls))
		return map[string]any{"retrievedDocs": b.String()}, nil
	})

	g.AddNode("write_sections", func(_ runtime.Runtime, state map[string]any) (any, error) {
		fmt.Println("---WRITE SECTIONS---")
		o, _ := state["outline"].(outline)
		docs, _ := state["retrievedDocs"].(string)
		refs, _ := state["references"].(map[string]string)
		urls := make([]string, 0, len(refs))
		for k := range refs {
			urls = append(urls, k)
		}
		sort.Strings(urls)
		sections := make([]wikiSection, 0, len(o.Sections))
		for _, s := range o.Sections {
			sections = append(sections, wikiSection{
				Title:     s.Title,
				Content:   s.Description + "\n\nGrounded in the offline corpus:\n" + docs,
				Citations: urls,
			})
		}
		fmt.Printf("drafted %d sections\n", len(sections))
		return map[string]any{"sections": sections}, nil
	})

	g.AddNode("write_article", func(_ runtime.Runtime, state map[string]any) (any, error) {
		fmt.Println("---WRITE ARTICLE---")
		sections, _ := state["sections"].([]wikiSection)
		var b strings.Builder
		for _, s := range sections {
			b.WriteString(s.String())
			b.WriteString("\n\n")
		}
		article := strings.TrimSpace(b.String())
		fmt.Printf("article length: %d characters\n", len(article))
		return map[string]any{"article": article}, nil
	})

	names := []string{"init_research", "conduct_interviews", "refine_outline", "index_references", "write_sections", "write_article"}
	for i, name := range names {
		if i > 0 {
			g.AddEdge(names[i-1], name)
		}
	}
	g.AddEdge(types.START, names[0])
	g.AddEdge(names[len(names)-1], types.END)

	cg, err := g.Compile(graphpkg.WithCheckpointer(checkpoint.NewMemorySaver()))
	if err != nil {
		log.Fatalf("compile storm graph: %v", err)
	}
	return cg
}

func main() {
	ctx := context.Background()
	interviews := buildInterviewGraph(3)
	storm := newStormGraph(interviews)

	result, err := storm.InvokeWithOptions(ctx, map[string]any{
		"topic": "Groq, NVIDIA, Llama.cpp and the future of LLM Inference",
	}, graphpkg.Options{ThreadID: "my-thread"})
	if err != nil {
		log.Fatalf("invoke storm: %v", err)
	}
	article, _ := result.Values["article"].(string)
	fmt.Println("\n===== FINAL ARTICLE =====")
	fmt.Println(article)
}

package textsplitters

import "fmt"

// Language identifies a programming or markup language with known recursive
// splitter separators.
type Language string

const (
	LanguageC          Language = "c"
	LanguageCPP        Language = "cpp"
	LanguageGo         Language = "go"
	LanguageJava       Language = "java"
	LanguageJS         Language = "js"
	LanguageTS         Language = "ts"
	LanguagePython     Language = "python"
	LanguageMarkdown   Language = "markdown"
	LanguageLatex      Language = "latex"
	LanguageHTML       Language = "html"
	LanguageRust       Language = "rust"
	LanguageRuby       Language = "ruby"
	LanguageR          Language = "r"
	LanguageElixir     Language = "elixir"
	LanguagePHP        Language = "php"
	LanguageSolidity   Language = "sol"
	LanguageCSharp     Language = "csharp"
	LanguageCOBOL      Language = "cobol"
	LanguageScala      Language = "scala"
	LanguageSwift      Language = "swift"
	LanguageKotlin     Language = "kotlin"
	LanguageLua        Language = "lua"
	LanguageHaskell    Language = "haskell"
	LanguagePowerShell Language = "powershell"
	LanguageProto      Language = "proto"
	LanguageRST        Language = "rst"
)

// SeparatorsForLanguage returns Python LangChain-compatible recursive splitter
// separators for common languages.
func SeparatorsForLanguage(language Language) ([]string, error) {
	switch language {
	case LanguageC, LanguageCPP:
		return []string{"\nclass ", "\nvoid ", "\nint ", "\nfloat ", "\ndouble ", "\nif ", "\nfor ", "\nwhile ", "\nswitch ", "\ncase ", "\n\n", "\n", " ", ""}, nil
	case LanguageGo:
		return []string{"\nfunc ", "\nvar ", "\nconst ", "\ntype ", "\nif ", "\nfor ", "\nswitch ", "\ncase ", "\n\n", "\n", " ", ""}, nil
	case LanguageJava:
		return []string{"\nclass ", "\npublic ", "\nprotected ", "\nprivate ", "\nstatic ", "\nif ", "\nfor ", "\nwhile ", "\nswitch ", "\ncase ", "\n\n", "\n", " ", ""}, nil
	case LanguageJS:
		return []string{"\nfunction ", "\nconst ", "\nlet ", "\nvar ", "\nclass ", "\nif ", "\nfor ", "\nwhile ", "\nswitch ", "\ncase ", "\ndefault ", "\n\n", "\n", " ", ""}, nil
	case LanguageTS:
		return []string{"\nenum ", "\ninterface ", "\nnamespace ", "\ntype ", "\nclass ", "\nfunction ", "\nconst ", "\nlet ", "\nvar ", "\nif ", "\nfor ", "\nwhile ", "\nswitch ", "\ncase ", "\ndefault ", "\n\n", "\n", " ", ""}, nil
	case LanguagePython:
		return []string{"\nclass ", "\ndef ", "\n\tdef ", "\n\n", "\n", " ", ""}, nil
	case LanguageR:
		return []string{"\nfunction ", "\nsetClass\\(", "\nsetMethod\\(", "\nsetGeneric\\(", "\nif ", "\nelse ", "\nfor ", "\nwhile ", "\nrepeat ", "\nlibrary\\(", "\nrequire\\(", "\n\n", "\n", " ", ""}, nil
	case LanguageMarkdown:
		return []string{"\n#{1,6} ", "```\n", "\n\\*\\*\\*+\n", "\n---+\n", "\n___+\n", "\n\n", "\n", " ", ""}, nil
	case LanguageLatex:
		return []string{"\n\\\\chapter{", "\n\\\\section{", "\n\\\\subsection{", "\n\\\\subsubsection{", "\n\\\\begin{enumerate}", "\n\\\\begin{itemize}", "\n\\\\begin{description}", "\n\\\\begin{list}", "\n\\\\begin{quote}", "\n\\\\begin{quotation}", "\n\\\\begin{verse}", "\n\\\\begin{verbatim}", "\n\\\\begin{align}", "$$", "$", " ", ""}, nil
	case LanguageHTML:
		return []string{"<body", "<div", "<p", "<br", "<li", "<h1", "<h2", "<h3", "<h4", "<h5", "<h6", "<span", "<table", "<tr", "<td", "<th", "<ul", "<ol", "<header", "<footer", "<nav", "<head", "<style", "<script", "<meta", "<title", ""}, nil
	case LanguageRust:
		return []string{"\nfn ", "\nconst ", "\nlet ", "\nif ", "\nwhile ", "\nfor ", "\nloop ", "\nmatch ", "\n\n", "\n", " ", ""}, nil
	case LanguageRuby:
		return []string{"\ndef ", "\nclass ", "\nif ", "\nunless ", "\nwhile ", "\nfor ", "\ndo ", "\nbegin ", "\nrescue ", "\n\n", "\n", " ", ""}, nil
	case LanguageElixir:
		return []string{"\ndef ", "\ndefp ", "\ndefmodule ", "\ndefprotocol ", "\ndefmacro ", "\ndefmacrop ", "\nif ", "\nunless ", "\ncase ", "\ncond ", "\nwith ", "\nfor ", "\ndo ", "\n\n", "\n", " ", ""}, nil
	case LanguagePHP:
		return []string{"\nfunction ", "\nclass ", "\nif ", "\nforeach ", "\nwhile ", "\ndo ", "\nswitch ", "\ncase ", "\n\n", "\n", " ", ""}, nil
	case LanguageSolidity:
		return []string{"\npragma ", "\nusing ", "\ncontract ", "\ninterface ", "\nlibrary ", "\nconstructor ", "\ntype ", "\nfunction ", "\nevent ", "\nmodifier ", "\nerror ", "\nstruct ", "\nenum ", "\nif ", "\nfor ", "\nwhile ", "\ndo while ", "\nassembly ", "\n\n", "\n", " ", ""}, nil
	case LanguageCSharp:
		return []string{"\ninterface ", "\nenum ", "\ndelegate ", "\nevent ", "\nclass ", "\nabstract ", "\npublic ", "\nprotected ", "\nprivate ", "\nstatic ", "\nreturn ", "\nif ", "\ncontinue ", "\nfor ", "\nforeach ", "\nwhile ", "\nswitch ", "\nbreak ", "\ncase ", "\nelse ", "\ntry ", "\nthrow ", "\nfinally ", "\ncatch ", "\n\n", "\n", " ", ""}, nil
	case LanguageCOBOL:
		return []string{"\nIDENTIFICATION DIVISION.", "\nENVIRONMENT DIVISION.", "\nDATA DIVISION.", "\nPROCEDURE DIVISION.", "\nWORKING-STORAGE SECTION.", "\nLINKAGE SECTION.", "\nFILE SECTION.", "\nINPUT-OUTPUT SECTION.", "\nOPEN ", "\nCLOSE ", "\nREAD ", "\nWRITE ", "\nIF ", "\nELSE ", "\nMOVE ", "\nPERFORM ", "\nUNTIL ", "\nVARYING ", "\nACCEPT ", "\nDISPLAY ", "\nSTOP RUN.", "\n", " ", ""}, nil
	case LanguageScala:
		return []string{"\nclass ", "\nobject ", "\ndef ", "\nval ", "\nvar ", "\nif ", "\nfor ", "\nwhile ", "\nmatch ", "\ncase ", "\n\n", "\n", " ", ""}, nil
	case LanguageSwift:
		return []string{"\nfunc ", "\nclass ", "\nstruct ", "\nenum ", "\nif ", "\nfor ", "\nwhile ", "\ndo ", "\nswitch ", "\ncase ", "\n\n", "\n", " ", ""}, nil
	case LanguageKotlin:
		return []string{"\nclass ", "\npublic ", "\nprotected ", "\nprivate ", "\ninternal ", "\ncompanion ", "\nfun ", "\nval ", "\nvar ", "\nif ", "\nfor ", "\nwhile ", "\nwhen ", "\nelse ", "\n\n", "\n", " ", ""}, nil
	case LanguageLua:
		return []string{"\nlocal ", "\nfunction ", "\nif ", "\nfor ", "\nwhile ", "\nrepeat ", "\n\n", "\n", " ", ""}, nil
	case LanguageHaskell:
		return []string{"\nmain :: ", "\nmain = ", "\nlet ", "\nin ", "\ndo ", "\nwhere ", "\n:: ", "\n= ", "\ndata ", "\nnewtype ", "\ntype ", "\nmodule ", "\nimport ", "\nqualified ", "\nimport qualified ", "\nclass ", "\ninstance ", "\ncase ", "\n| ", "\n= {", "\n, ", "\n\n", "\n", " ", ""}, nil
	case LanguagePowerShell:
		return []string{"\nfunction ", "\nparam ", "\nif ", "\nforeach ", "\nfor ", "\nwhile ", "\nswitch ", "\nclass ", "\ntry ", "\ncatch ", "\nfinally ", "\n\n", "\n", " ", ""}, nil
	case LanguageProto:
		return []string{"\nmessage ", "\nservice ", "\nenum ", "\noption ", "\nimport ", "\nsyntax ", "\n\n", "\n", " ", ""}, nil
	case LanguageRST:
		return []string{"\n=+\n", "\n-+\n", "\n\\*+\n", "\n\n.. *\n\n", "\n\n", "\n", " ", ""}, nil
	default:
		return nil, fmt.Errorf("unsupported language %q", language)
	}
}

// NewRecursiveCharacterFromLanguage creates a recursive splitter configured
// with language-specific separators. Regex mode is enabled to match Python.
func NewRecursiveCharacterFromLanguage(language Language, cfg Config) (*RecursiveCharacterTextSplitter, error) {
	separators, err := SeparatorsForLanguage(language)
	if err != nil {
		return nil, err
	}
	return NewRecursiveCharacter(separators, true, cfg)
}

// NewMarkdown creates a recursive Markdown splitter.
func NewMarkdown(cfg Config) (*RecursiveCharacterTextSplitter, error) {
	return NewRecursiveCharacterFromLanguage(LanguageMarkdown, cfg)
}

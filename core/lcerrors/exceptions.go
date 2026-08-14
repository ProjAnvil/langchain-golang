// Typed exception hierarchy ported from langchain-core's
// langchain_core.exceptions module. These types let callers distinguish
// categories of failure with errors.As/errors.Is while preserving the
// underlying error chain through Unwrap.
package lcerrors

// ErrorCode mirrors Python's ErrorCode enum. The string values are stable and
// are used verbatim when building the troubleshooting URL in CreateMessage.
type ErrorCode string

const (
	// ErrorCodeInvalidPromptInput indicates a prompt was rejected as invalid.
	ErrorCodeInvalidPromptInput ErrorCode = "INVALID_PROMPT_INPUT"
	// ErrorCodeInvalidToolResults indicates tool results were malformed.
	ErrorCodeInvalidToolResults ErrorCode = "INVALID_TOOL_RESULTS"
	// ErrorCodeMessageCoercionFailure indicates a message could not be coerced.
	ErrorCodeMessageCoercionFailure ErrorCode = "MESSAGE_COERCION_FAILURE"
	// ErrorCodeModelAuthentication indicates provider authentication failed.
	ErrorCodeModelAuthentication ErrorCode = "MODEL_AUTHENTICATION"
	// ErrorCodeModelNotFound indicates the requested model was not found.
	ErrorCodeModelNotFound ErrorCode = "MODEL_NOT_FOUND"
	// ErrorCodeModelRateLimit indicates the provider rate-limited the request.
	ErrorCodeModelRateLimit ErrorCode = "MODEL_RATE_LIMIT"
	// ErrorCodeOutputParsingFailure indicates an output parser failed.
	ErrorCodeOutputParsingFailure ErrorCode = "OUTPUT_PARSING_FAILURE"
)

// CreateMessage appends the LangChain troubleshooting line to message, matching
// Python's create_message exactly (including the trailing space).
func CreateMessage(message string, code ErrorCode) string {
	return message + "\n" +
		"For troubleshooting, visit: https://docs.langchain.com/oss/python/langchain/errors/" +
		string(code) + " "
}

// LangChainException is the general LangChain exception.
type LangChainException struct {
	Message string
}

// Error implements the error interface.
func (e *LangChainException) Error() string { return e.Message }

// NewLangChainException creates a LangChainException with the given message.
func NewLangChainException(msg string) *LangChainException {
	return &LangChainException{Message: msg}
}

// OutputParserException marks a parsing error, differentiating it from other
// code or execution errors that may arise inside an output parser.
//
// It mirrors Python's OutputParserException fields: observation, llm_output,
// and send_to_llm. Err carries the underlying error so errors.Is/errors.As can
// still reach it.
type OutputParserException struct {
	Message     string
	Observation string
	LLMOutput   string
	SendToLLM   bool
	Err         error
}

// Error implements the error interface, returning the message.
func (e *OutputParserException) Error() string { return e.Message }

// Unwrap returns the wrapped underlying error, keeping it reachable for
// errors.Is and errors.As.
func (e *OutputParserException) Unwrap() error { return e.Err }

// NewOutputParserException mirrors Python's OutputParserException str path: it
// augments message with the OUTPUT_PARSING_FAILURE troubleshooting link.
func NewOutputParserException(message string) *OutputParserException {
	return &OutputParserException{
		Message: CreateMessage(message, ErrorCodeOutputParsingFailure),
	}
}

// NewOutputParserExceptionFromError wraps an existing error as an
// OutputParserException, preserving its message text and its error chain.
// It returns nil when err is nil.
func NewOutputParserExceptionFromError(err error) *OutputParserException {
	if err == nil {
		return nil
	}
	return &OutputParserException{
		Message: err.Error(),
		Err:     err,
	}
}

// ContextOverflowError is raised when input exceeds the model's context limit.
type ContextOverflowError struct {
	Message string
}

// Error implements the error interface.
func (e *ContextOverflowError) Error() string { return e.Message }

// NewContextOverflowError creates a ContextOverflowError with the given message.
func NewContextOverflowError(msg string) *ContextOverflowError {
	return &ContextOverflowError{Message: msg}
}

// TracerException is the base class for exceptions in tracers.
type TracerException struct {
	Message string
}

// Error implements the error interface.
func (e *TracerException) Error() string { return e.Message }

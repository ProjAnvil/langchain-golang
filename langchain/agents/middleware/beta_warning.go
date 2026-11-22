// The Go equivalent of Python's @beta(obj_type="middleware") decoration:
// beta middleware surfaces a once-per-process warning through the default
// slog logger. Python's warnings machinery has no Go counterpart; slog is
// the closest idiomatic channel, and the once-per-class semantics of
// LangChainBetaWarning are kept via a per-name sync.Once.

package middleware

import (
	"fmt"
	"log/slog"
	"sync"
)

// betaWarnings tracks which beta middleware classes have already warned.
var betaWarnings sync.Map // name -> *sync.Once

// warnBetaMiddleware emits the beta warning for name once per process,
// mirroring LangChainBetaWarning's once-per-class filter: "The middleware
// `X` is in beta. It is actively being worked on, so the API may change."
func warnBetaMiddleware(name string) {
	once, _ := betaWarnings.LoadOrStore(name, &sync.Once{})
	once.(*sync.Once).Do(func() {
		slog.Warn(fmt.Sprintf(
			"The middleware `%s` is in beta. It is actively being worked on, so the API may change.", name))
	})
}

package savertest_test

import (
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/savertest"
)

func TestMemorySaverContract(t *testing.T) {
	savertest.Run(t, func(t *testing.T) checkpoint.Saver {
		return checkpoint.NewMemorySaver()
	})
}

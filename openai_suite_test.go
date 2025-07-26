//go:build integration

// run this with: go test -tags=integration -run ^TestOpenAI_StandardSuite

package openai

import (
	"os"
	"testing"

	"github.com/nexxia-ai/aigentic/ai"
	"github.com/nexxia-ai/aigentic/tests/integration"
	"github.com/nexxia-ai/aigentic/utils"
)

func init() {
	utils.LoadEnvFile("./.env")
}

// TestOpenAI_StandardSuite runs the standard test suite against the OpenAI implementation
func TestOpenAI_StandardSuite(t *testing.T) {
	suite := integration.ModelTestSuite{
		NewModel: func() *ai.Model {
			return NewModel("gpt-4o-mini", os.Getenv("OPENAI_API_KEY"))
		},
		Name: "OpenAI",
		SkipTests: []string{
			"ProcessImage",
		},
	}
	integration.RunModelTestSuite(t, suite)
}

// TestOpenAI_IndividualTests demonstrates how to run individual tests
func TestOpenAI_IndividualTests(t *testing.T) {
	model := NewModel("gpt-4o-mini", os.Getenv("OPENAI_API_KEY"))

	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	t.Run("GenerateSimple", func(t *testing.T) {
		integration.TestGenerateSimple(t, model)
	})

	t.Run("ProcessImage", func(t *testing.T) {
		integration.TestProcessImage(t, model)
	})

	t.Run("ProcessAttachments", func(t *testing.T) {
		integration.TestProcessAttachments(t, model)
	})

	t.Run("GenerateContentWithTools", func(t *testing.T) {
		integration.TestGenerateContentWithTools(t, model)
	})
}

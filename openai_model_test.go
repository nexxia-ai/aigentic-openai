//go:build integration

// run this with: go test -v -tags=integration -run ^TestOpenAI_ModelSuite

package openai

import (
	"os"
	"testing"

	"github.com/nexxia-ai/aigentic/ai"
	"github.com/nexxia-ai/aigentic/utils"
)

func init() {
	utils.LoadEnvFile("../.env")
}

// TestOpenAI_StandardSuite runs the standard test suite against the OpenAI implementation
func TestOpenAI_ModelSuite(t *testing.T) {
	suite := ai.ModelTestSuite{
		NewModel: func() *ai.Model {
			return NewModel("gpt-4o-mini", os.Getenv("OPENAI_API_KEY"))
		},
		Name: "OpenAI",
		SkipTests: []string{
			"ProcessImage",
		},
	}
	ai.RunModelTestSuite(t, suite)
}

// TestOpenAI_IndividualTests demonstrates how to run individual tests
func TestOpenAI_IndividualTests(t *testing.T) {
	model := NewModel("gpt-4o-mini", os.Getenv("OPENAI_API_KEY"))

	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set")
	}

	t.Run("GenerateSimple", func(t *testing.T) {
		ai.TestGenerateSimple(t, model)
	})

	t.Run("ProcessImage", func(t *testing.T) {
		ai.TestProcessImage(t, model)
	})

	t.Run("ProcessAttachments", func(t *testing.T) {
		ai.TestProcessAttachments(t, model)
	})

	t.Run("GenerateContentWithTools", func(t *testing.T) {
		ai.TestGenerateContentWithTools(t, model)
	})
}

//go:build integration

// run this with: go test -v -tags=integration -run ^TestOpenAI_ModelSuite

package openai

import (
	"os"
	"testing"

	"github.com/nexxia-ai/aigentic/ai"
)

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

func TestOpenAI_ModelSuite_OpenRouter(t *testing.T) {
	suite := ai.ModelTestSuite{
		NewModel: func() *ai.Model {
			return NewModel("qwen/qwen3-30b-a3b-instruct-2507", os.Getenv("OPENROUTER_API_KEY"), OpenRouterBaseURL)
		},
		Name: "OpenRouter",
		SkipTests: []string{
			"ProcessImage",
		},
	}
	ai.RunModelTestSuite(t, suite)
}

func TestOpenAI_ProcessImage(t *testing.T) {
	ai.TestProcessImage(t, NewModel("gpt-4o", os.Getenv("OPENAI_API_KEY")))
}

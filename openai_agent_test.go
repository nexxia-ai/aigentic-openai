//go:build integration

// run this with: go test -v -tags=integration -run ^TestOpenAI_AgentSuite

package openai

import (
	"os"
	"testing"

	"github.com/nexxia-ai/aigentic"
	"github.com/nexxia-ai/aigentic/ai"
	"github.com/nexxia-ai/aigentic/document"
	"github.com/stretchr/testify/assert"
)

func TestOpenAI_AgentSuite(t *testing.T) {
	aigentic.RunIntegrationTestSuite(t, aigentic.IntegrationTestSuite{
		NewModel: func() *ai.Model {
			return NewModel("gpt-4o-mini", os.Getenv("OPENAI_API_KEY"))
		},
		Name: "OpenAI",
		SkipTests: []string{
			"TeamCoordination",
		},
	})
}

func TestOpenAI_AgentSuite_OpenRouter(t *testing.T) {
	aigentic.RunIntegrationTestSuite(t, aigentic.IntegrationTestSuite{
		NewModel: func() *ai.Model {
			return NewModel("qwen/qwen3-30b-a3b-instruct-2507", os.Getenv("OPENROUTER_API_KEY"), OpenRouterBaseURL)
		},
		Name: "OpenRouter",
		SkipTests: []string{
			"TeamCoordination",
		},
	})
}
func TestOpenAI_AgentSuite_Helicone(t *testing.T) {
	aigentic.RunIntegrationTestSuite(t, aigentic.IntegrationTestSuite{
		NewModel: func() *ai.Model {
			return NewModel("gpt-4o-mini", os.Getenv("HELICONE_API_KEY"), HeliconeBaseURL)
		},
		Name: "OpenRouter",
		SkipTests: []string{
			"TeamCoordination",
		},
	})
}

func TestOpenAI_BasicAgent(t *testing.T) {
	model := NewModel("gpt-4o-mini", os.Getenv("OPENAI_API_KEY"))
	aigentic.TestBasicAgent(t, model)
}

func TestOpenAI_TeamCoordination(t *testing.T) {
	model := NewModel("gpt-4o-mini", os.Getenv("OPENAI_API_KEY"))
	aigentic.TestTeamCoordination(t, model)
}

// TestAgent_Run_WithFileID tests the agent with OpenAI Files API integration
func TestOpenAI_Agent_WithFileID(t *testing.T) {
	// Skip if no OpenAI API key is available
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Fatal("Skipping OpenAI integration test: OPENAI_API_KEY not set")
	}

	model := NewModel("o4-mini", "")

	// Create a document reference for the file ID
	fileDoc := document.NewInMemoryDocument("file-WjBr55R67mVmhXCsvKZ6Zs", "document.pdf", nil, nil)

	agent := aigentic.Agent{
		Model:              model,
		Description:        "You are a helpful assistant that analyzes files and provides insights.",
		Instructions:       "When you see a file reference, analyze it and provide a summary. If you cannot access the file, explain why.",
		Trace:              aigentic.NewTrace(),
		DocumentReferences: []*document.Document{fileDoc},
	}

	// Test the agent with file ID
	_, err := agent.Execute("Please analyze the attached file and tell me what it contains. If you can access it, start your response with 'SUCCESS:' followed by the analysis.")
	assert.NoError(t, err)
}

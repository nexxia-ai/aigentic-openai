//go:build integration

// run this with: go test -v -tags=integration -run ^TestOpenAI_AgentSuite

package openai

import (
	"os"
	"testing"

	"github.com/nexxia-ai/aigentic"
	"github.com/nexxia-ai/aigentic/ai"
	"github.com/stretchr/testify/assert"
)

func TestOpenAI_AgentSuite(t *testing.T) {
	aigentic.RunIntegrationTestSuite(t, aigentic.IntegrationTestSuite{
		NewModel: func() *ai.Model {
			return NewModel("gpt-4o-mini", os.Getenv("OPENAI_API_KEY"))
		},
		Name: "OpenAI",
		SkipTests: []string{
			"Streaming",
		},
	})
}

func TestOpenAI_ToolIntegration(t *testing.T) {
	aigentic.TestToolIntegration(t, NewModel("gpt-4o-mini", os.Getenv("OPENAI_API_KEY")))
}
func TestOpenAI_Streaming(t *testing.T) {
	aigentic.TestStreaming(t, NewModel("gpt-4o-mini", os.Getenv("OPENAI_API_KEY")))
}

func TestOpenAI_MultiAgentChain(t *testing.T) {
	aigentic.TestMultiAgentChain(t, NewModel("gpt-4o-mini", os.Getenv("OPENAI_API_KEY")))
}

func TestOpenAI_TeamCoordination(t *testing.T) {
	aigentic.TestTeamCoordination(t, NewModel("gpt-4o-mini", os.Getenv("OPENAI_API_KEY")))
}

// TestAgent_Run_WithFileID tests the agent with OpenAI Files API integration
func TestOpenAI_Agent_WithFileID(t *testing.T) {
	// Skip if no OpenAI API key is available
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Fatal("Skipping OpenAI integration test: OPENAI_API_KEY not set")
	}

	model := NewModel("o4-mini", "")

	agent := aigentic.Agent{
		Model:        model,
		Description:  "You are a helpful assistant that analyzes files and provides insights.",
		Instructions: "When you see a file reference, analyze it and provide a summary. If you cannot access the file, explain why.",
		Trace:        aigentic.NewTrace(),
		Attachments: []aigentic.Attachment{
			{
				Type:     "file",
				MimeType: "application/pdf",
				Name:     "file-WjBr55R67mVmhXCsvKZ6Zs",
			},
		},
	}

	// Test the agent with file ID
	_, err := agent.RunAndWait("Please analyze the attached file and tell me what it contains. If you can access it, start your response with 'SUCCESS:' followed by the analysis.")
	assert.NoError(t, err)
}

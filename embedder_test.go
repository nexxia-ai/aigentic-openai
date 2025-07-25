package openai

import (
	"os"
	"testing"
	"time"

	"github.com/nexxia-ai/aigentic"
)

func TestOpenAIEmbedder(t *testing.T) {
	// Check for required environment variables
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Fatal("OPENAI_API_KEY environment variable is required for test")
	}

	// Create a new OpenAI embedder
	embedder := NewOpenAIEmbedder(apiKey)

	// Test default configuration
	if embedder.Model != "text-embedding-ada-002" {
		t.Errorf("Expected default ai.Model 'text-embedding-ada-002', got '%s'", embedder.Model)
	}

	if embedder.Dimensions != 1536 {
		t.Errorf("Expected default dimensions 1536, got %d", embedder.Dimensions)
	}

	// Test embedding
	text := "This is a sample text to embed"
	embedding, err := embedder.Embed(text)
	if err != nil {
		t.Fatalf("Failed to embed text: %v", err)
	}

	if len(embedding) != 1536 {
		t.Errorf("Expected embedding length 1536, got %d", len(embedding))
	}

	// Test that embedding values are reasonable (not all zeros)
	hasNonZero := false
	for _, val := range embedding {
		if val != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("Embedding contains only zeros")
	}

	// Test empty text
	_, err = embedder.Embed("")
	if err == nil {
		t.Error("Expected error for empty text")
	}

	// Test ai.Model switching
	embedder.SetModel("text-embedding-3-small")
	if embedder.Model != "text-embedding-3-small" {
		t.Errorf("Expected ai.Model 'text-embedding-3-small', got '%s'", embedder.Model)
	}

	// Test embedding with new ai.Model
	embedding2, err := embedder.Embed("Another sample text")
	if err != nil {
		t.Fatalf("Failed to embed text with new ai.Model: %v", err)
	}

	if len(embedding2) != 1536 {
		t.Errorf("Expected embedding length 1536 for text-embedding-3-small, got %d", len(embedding2))
	}

	// Test timeout setting
	embedder.SetTimeout(10 * time.Second)
	if embedder.HTTPClient.Timeout != 10*time.Second {
		t.Errorf("Expected timeout 10s, got %v", embedder.HTTPClient.Timeout)
	}

	// Test base URL setting
	embedder.SetBaseURL("https://custom.openai.com/v1")
	if embedder.BaseURL != "https://custom.openai.com/v1" {
		t.Errorf("Expected base URL 'https://custom.openai.com/v1', got '%s'", embedder.BaseURL)
	}
}

func TestOpenAIEmbedderWithRetriever(t *testing.T) {
	// Check for required environment variables
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Fatal("OPENAI_API_KEY environment variable is required for test")
	}

	// Create embedder
	embedder := NewOpenAIEmbedder(apiKey)
	embedder.SetModel("text-embedding-3-small")

	// Test that embedder implements the Embedder interface
	var _ aigentic.Embedder = embedder

	// Test embedding for retriever usage
	query := "What is the company policy on remote work?"
	embedding, err := embedder.Embed(query)
	if err != nil {
		t.Fatalf("Failed to embed query: %v", err)
	}

	if len(embedding) != 1536 {
		t.Errorf("Expected embedding length 1536, got %d", len(embedding))
	}

	// Verify embedding is suitable for vector search (has reasonable values)
	sum := 0.0
	for _, val := range embedding {
		sum += val * val
	}
	magnitude := sum
	if magnitude < 0.1 || magnitude > 10.0 {
		t.Errorf("Embedding magnitude %f is outside expected range [0.1, 10.0]", magnitude)
	}
}

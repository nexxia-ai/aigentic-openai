package openai

import (
	"os"
	"testing"
	"time"

	"github.com/nexxia-ai/aigentic/utils"
)

func init() {
	utils.LoadEnvFile("../.env")
}

// Integration test that requires real OpenAI API
func TestOpenAIEmbedderIntegration(t *testing.T) {
	// Check for required environment variables
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Fatal("OPENAI_API_KEY environment variable is required for integration test")
	}

	// Create embedder
	embedder := NewOpenAIEmbedder(apiKey)

	// Test different ai.Models
	models := []string{
		"text-embedding-ada-002",
		"text-embedding-3-small",
		"text-embedding-3-large",
	}

	for _, model := range models {
		t.Run("ai.Model_"+model, func(t *testing.T) {
			embedder.SetModel(model)

			// Test embedding generation
			text := "This is a test text for embedding with " + model
			embedding, err := embedder.Embed(text)
			if err != nil {
				t.Fatalf("Failed to embed text with ai.Model %s: %v", model, err)
			}

			// Verify embedding dimensions
			expectedDimensions := 1536
			if model == "text-embedding-3-large" {
				expectedDimensions = 3072
			}

			if len(embedding) != expectedDimensions {
				t.Errorf("Expected embedding length %d for ai.Model %s, got %d", expectedDimensions, model, len(embedding))
			}

			// Verify embedding quality
			sum := 0.0
			for _, val := range embedding {
				sum += val * val
			}
			magnitude := sum
			if magnitude < 0.1 || magnitude > 10.0 {
				t.Errorf("Embedding magnitude %f for ai.Model %s is outside expected range [0.1, 10.0]", magnitude, model)
			}

			t.Logf("ai.Model %s: dimensions=%d, magnitude=%.3f", model, len(embedding), magnitude)
		})
	}
}

// Integration test for different text types
func TestOpenAIEmbedderTextTypes(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Fatal("OPENAI_API_KEY environment variable is required for integration test")
	}

	embedder := NewOpenAIEmbedder(apiKey)
	embedder.SetModel("text-embedding-ada-002")

	testCases := []struct {
		name string
		text string
	}{
		{"Short Text", "Hello world"},
		{"Medium Text", "This is a medium length text that should generate a reasonable embedding"},
		{"Long Text", "This is a much longer text that contains more words and should still generate a valid embedding. It includes various types of content and should test the embedder's ability to handle longer inputs."},
		{"Technical Text", "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed."},
		{"Question", "What is the capital of France?"},
		{"Code Snippet", "function hello() { return 'Hello, World!'; }"},
		{"Special Characters", "Text with special chars: @#$%^&*()_+-=[]{}|;':\",./<>?"},
		{"Unicode", "Text with unicode: 你好世界 🌍"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			embedding, err := embedder.Embed(tc.text)
			if err != nil {
				t.Fatalf("Failed to embed text '%s': %v", tc.text, err)
			}

			if len(embedding) != 1536 {
				t.Errorf("Expected embedding length 1536, got %d", len(embedding))
			}

			// Verify embedding quality
			sum := 0.0
			for _, val := range embedding {
				sum += val * val
			}
			magnitude := sum
			if magnitude < 0.1 || magnitude > 10.0 {
				t.Errorf("Embedding magnitude %f for text '%s' is outside expected range [0.1, 10.0]", magnitude, tc.text)
			}

			t.Logf("Text type '%s': length=%d, magnitude=%.3f", tc.name, len(tc.text), magnitude)
		})
	}
}

// Integration test for configuration changes
func TestOpenAIEmbedderConfiguration(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Fatal("OPENAI_API_KEY environment variable is required for integration test")
	}

	embedder := NewOpenAIEmbedder(apiKey)

	// Test base URL change
	t.Run("Custom Base URL", func(t *testing.T) {
		originalURL := embedder.BaseURL
		embedder.SetBaseURL("https://api.openai.com/v1") // Should still work

		embedding, err := embedder.Embed("Test text")
		if err != nil {
			t.Fatalf("Failed to embed with custom base URL: %v", err)
		}

		if len(embedding) != 1536 {
			t.Errorf("Expected embedding length 1536, got %d", len(embedding))
		}

		// Restore original URL
		embedder.SetBaseURL(originalURL)
	})

	// Test timeout change
	t.Run("Custom Timeout", func(t *testing.T) {
		originalTimeout := embedder.HTTPClient.Timeout
		embedder.SetTimeout(60 * time.Second)

		embedding, err := embedder.Embed("Test text with custom timeout")
		if err != nil {
			t.Fatalf("Failed to embed with custom timeout: %v", err)
		}

		if len(embedding) != 1536 {
			t.Errorf("Expected embedding length 1536, got %d", len(embedding))
		}

		// Restore original timeout
		embedder.SetTimeout(originalTimeout)
	})
}

// Integration test for error conditions
func TestOpenAIEmbedderErrors(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Fatal("OPENAI_API_KEY environment variable is required for integration test")
	}

	embedder := NewOpenAIEmbedder(apiKey)

	// Test empty text
	t.Run("Empty Text", func(t *testing.T) {
		_, err := embedder.Embed("")
		if err == nil {
			t.Error("Expected error for empty text")
		}
	})

	// Test very long text (should still work but might be slow)
	t.Run("Very Long Text", func(t *testing.T) {
		longText := ""
		for i := 0; i < 1000; i++ {
			longText += "This is a very long text that should test the embedder's ability to handle large inputs. "
		}

		embedding, err := embedder.Embed(longText)
		if err != nil {
			t.Fatalf("Failed to embed very long text: %v", err)
		}

		if len(embedding) != 1536 {
			t.Errorf("Expected embedding length 1536, got %d", len(embedding))
		}

		t.Logf("Successfully embedded text of length %d", len(longText))
	})
}

// Integration test for retriever compatibility
func TestOpenAIEmbedderRetrieverCompatibility(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Fatal("OPENAI_API_KEY environment variable is required for integration test")
	}

	embedder := NewOpenAIEmbedder(apiKey)
	embedder.SetModel("text-embedding-ada-002")

	// Test queries that might be used with retrievers
	queries := []string{
		"What is the company policy on remote work?",
		"How do I implement authentication in my application?",
		"What are the best practices for API design?",
		"Explain the difference between REST and GraphQL",
		"What is the process for deploying to production?",
	}

	for i, query := range queries {
		t.Run("Query_"+string(rune('A'+i)), func(t *testing.T) {
			embedding, err := embedder.Embed(query)
			if err != nil {
				t.Fatalf("Failed to embed query '%s': %v", query, err)
			}

			if len(embedding) != 1536 {
				t.Errorf("Expected embedding length 1536, got %d", len(embedding))
			}

			// Verify embedding quality for retriever usage
			sum := 0.0
			for _, val := range embedding {
				sum += val * val
			}
			magnitude := sum
			if magnitude < 0.1 || magnitude > 10.0 {
				t.Errorf("Embedding magnitude %f for query '%s' is outside expected range [0.1, 10.0]", magnitude, query)
			}

			t.Logf("Query %d: length=%d, magnitude=%.3f", i+1, len(query), magnitude)
		})
	}
}

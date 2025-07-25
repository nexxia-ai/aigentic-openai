package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// OpenAIEmbedder implements text embedding using OpenAI's API
type OpenAIEmbedder struct {
	APIKey     string
	BaseURL    string
	Model      string
	Dimensions int
	HTTPClient *http.Client
}

// OpenAIEmbeddingRequest represents a request to OpenAI's embedding API
type OpenAIEmbeddingRequest struct {
	Input string `json:"input"`
	Model string `json:"model"`
}

// OpenAIEmbeddingResponse represents a response from OpenAI's embedding API
type OpenAIEmbeddingResponse struct {
	Data []struct {
		Embedding []float64 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

// NewOpenAIEmbedder creates a new OpenAI embedder with default configuration
func NewOpenAIEmbedder(apiKey string) *OpenAIEmbedder {
	return &OpenAIEmbedder{
		APIKey:     apiKey,
		BaseURL:    "https://api.openai.com/v1",
		Model:      "text-embedding-ada-002",
		Dimensions: 1536, // Default for text-embedding-ada-002
		HTTPClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// Embed converts text to vector embedding using OpenAI's API
func (e *OpenAIEmbedder) Embed(text string) ([]float64, error) {
	if text == "" {
		return nil, fmt.Errorf("text cannot be empty")
	}

	// Prepare request
	request := OpenAIEmbeddingRequest{
		Input: text,
		Model: e.Model,
	}

	requestBody, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Build API URL
	url := fmt.Sprintf("%s/embeddings", strings.TrimSuffix(e.BaseURL, "/"))

	// Create HTTP request
	req, err := http.NewRequestWithContext(context.Background(), "POST", url, bytes.NewReader(requestBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", e.APIKey))

	// Make the request
	resp, err := e.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	// Check for HTTP errors
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("OpenAI API returned status %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var embeddingResponse OpenAIEmbeddingResponse
	if err := json.Unmarshal(body, &embeddingResponse); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	// Validate response
	if len(embeddingResponse.Data) == 0 {
		return nil, fmt.Errorf("no embedding data in response")
	}

	// Return the embedding
	return embeddingResponse.Data[0].Embedding, nil
}

// SetModel updates the embedding Model and dimensions
func (e *OpenAIEmbedder) SetModel(model string) {
	e.Model = model

	// Update dimensions based on model
	switch model {
	case "text-embedding-ada-002":
		e.Dimensions = 1536
	case "text-embedding-3-small":
		e.Dimensions = 1536
	case "text-embedding-3-large":
		e.Dimensions = 3072
	default:
		// Keep current dimensions if model is unknown
	}
}

// SetBaseURL updates the base URL for the API
func (e *OpenAIEmbedder) SetBaseURL(baseURL string) {
	e.BaseURL = baseURL
}

// SetTimeout updates the HTTP client timeout
func (e *OpenAIEmbedder) SetTimeout(timeout time.Duration) {
	e.HTTPClient.Timeout = timeout
}

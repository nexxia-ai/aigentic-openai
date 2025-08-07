package openai

import (
	"errors"
	"strings"
	"testing"

	"github.com/nexxia-ai/aigentic/ai"
)

func TestIsRetryableError(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		expected bool
	}{
		{
			name:     "502 Bad Gateway",
			err:      errors.New("status: 502 Bad Gateway, code: 502"),
			expected: true,
		},
		{
			name:     "503 Service Unavailable",
			err:      errors.New("status: 503 Service Unavailable, code: 503"),
			expected: true,
		},
		{
			name:     "504 Gateway Timeout",
			err:      errors.New("status: 504 Gateway Timeout, code: 504"),
			expected: true,
		},
		{
			name:     "429 Rate Limit",
			err:      errors.New("status: 429 Too Many Requests, code: 429"),
			expected: true,
		},
		{
			name:     "Connection refused",
			err:      errors.New("connection refused"),
			expected: true,
		},
		{
			name:     "Timeout error",
			err:      errors.New("timeout"),
			expected: true,
		},
		{
			name:     "Network error",
			err:      errors.New("network error"),
			expected: true,
		},
		{
			name:     "400 Bad Request (not retryable)",
			err:      errors.New("status: 400 Bad Request, code: 400"),
			expected: false,
		},
		{
			name:     "401 Unauthorized (not retryable)",
			err:      errors.New("status: 401 Unauthorized, code: 401"),
			expected: false,
		},
		{
			name:     "Nil error",
			err:      nil,
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isRetryableError(tt.err)
			isRetryable := errors.Is(result, ai.ErrTemporary)
			if isRetryable != tt.expected {
				t.Errorf("isRetryableError(%v) = %v, want %v", tt.err, isRetryable, tt.expected)
			}
		})
	}
}

func TestRetryErrorMessages(t *testing.T) {
	// Test that retry error messages are properly formatted
	err := errors.New("status: 502 Bad Gateway, code: 502")

	result := isRetryableError(err)
	if !errors.Is(result, ai.ErrTemporary) {
		t.Error("502 error should be retryable")
	}

	// Test that the error message contains the expected content
	errStr := err.Error()
	if !strings.Contains(errStr, "502") {
		t.Error("Error message should contain 502 status code")
	}
}

func TestOpenAIConvertMessages_ResourceMessage_NewFormat(t *testing.T) {
	tests := []struct {
		name     string
		message  ai.ResourceMessage
		expected interface{}
	}{
		{
			name: "File ID with URI",
			message: ai.ResourceMessage{
				Role: ai.UserRole,
				URI:  "file://file-abc123",
				Name: "document.pdf",
			},
			expected: []OpenAIContentPart{
				{
					Type: "file",
					File: &OpenAIFile{
						FileID: "file-abc123",
					},
				},
				{
					Type: "text",
					Text: "File: document.pdf",
				},
			},
		},
		{
			name: "File ID with description",
			message: ai.ResourceMessage{
				Role:        ai.UserRole,
				URI:         "file://file-def456",
				Name:        "image.jpg",
				Description: "A beautiful landscape image",
			},
			expected: []OpenAIContentPart{
				{
					Type: "file",
					File: &OpenAIFile{
						FileID: "file-def456",
					},
				},
				{
					Type: "text",
					Text: "A beautiful landscape image",
				},
			},
		},
		{
			name: "Image with base64 encoding (unchanged)",
			message: ai.ResourceMessage{
				Role:     ai.UserRole,
				MIMEType: "image/png",
				Body:     []byte("fake-image-data"),
				Name:     "test.png",
			},
			expected: "data:image/png;base64,ZmFrZS1pbWFnZS1kYXRh", // base64 of "fake-image-data"
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			messages := []ai.Message{tt.message}
			openaiMessages := openAIConvertMessages(messages)

			if len(openaiMessages) != 1 {
				t.Fatalf("Expected 1 message, got %d", len(openaiMessages))
			}

			// Check content type and structure
			switch expected := tt.expected.(type) {
			case []OpenAIContentPart:
				// Should be content array
				contentParts, ok := openaiMessages[0].Content.([]OpenAIContentPart)
				if !ok {
					t.Fatalf("Expected content to be []OpenAIContentPart, got %T", openaiMessages[0].Content)
				}

				if len(contentParts) != len(expected) {
					t.Errorf("Expected %d content parts, got %d", len(expected), len(contentParts))
				}

				// Check each content part
				for i, expectedPart := range expected {
					if i >= len(contentParts) {
						t.Errorf("Missing content part %d", i)
						continue
					}

					actualPart := contentParts[i]
					if actualPart.Type != expectedPart.Type {
						t.Errorf("Content part %d: expected type %s, got %s", i, expectedPart.Type, actualPart.Type)
					}

					if expectedPart.Text != "" && actualPart.Text != expectedPart.Text {
						t.Errorf("Content part %d: expected text %s, got %s", i, expectedPart.Text, actualPart.Text)
					}

					if expectedPart.File != nil {
						if actualPart.File == nil {
							t.Errorf("Content part %d: expected file, got nil", i)
						} else if actualPart.File.FileID != expectedPart.File.FileID {
							t.Errorf("Content part %d: expected file ID %s, got %s", i, expectedPart.File.FileID, actualPart.File.FileID)
						}
					}
				}

			case string:
				// Should be string content
				content, ok := openaiMessages[0].Content.(string)
				if !ok {
					t.Fatalf("Expected content to be string, got %T", openaiMessages[0].Content)
				}

				if content != expected {
					t.Errorf("Expected content '%s', got '%s'", expected, content)
				}
			}
		})
	}
}

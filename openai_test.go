package openai

import (
	"context"
	"embed"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/nexxia-ai/aigentic/ai"
	"github.com/nexxia-ai/aigentic/utils"
)

//go:embed testdata/*
var testData embed.FS

func init() {
	utils.LoadEnvFile("./.env")
}

// Common echo tool for testing
var echoTool = ai.Tool{
	Name:        "echo",
	Description: "Echoes back the input text",
	InputSchema: map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"text": map[string]interface{}{
				"type":        "string",
				"description": "The text to echo back",
			},
		},
		"required": []string{"text"},
	},
	Execute: func(args map[string]interface{}) (*ai.ToolResult, error) {
		text := args["text"].(string)
		return &ai.ToolResult{
			Content: []ai.ToolContent{{
				Type:    "text",
				Content: text,
			}},
			Error: false,
		}, nil
	},
}

type testArgs struct {
	ctx      context.Context
	messages []ai.Message
	tools    []ai.Tool
}

func TestModel_GenerateSimple(t *testing.T) {
	args := testArgs{
		ctx:      context.Background(),
		messages: []ai.Message{ai.UserMessage{Role: ai.UserRole, Content: "What is the capital of Australia?"}},
		tools:    []ai.Tool{},
	}
	tests := []struct {
		name    string
		m       *ai.Model
		args    testArgs
		want    any
		wantErr bool
	}{
		{name: "openai api",
			m:    NewOpenAIModel("gpt-4o-mini", os.Getenv("OPENAI_API_KEY")),
			args: args,
			want: "Canberra",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.m.Call(tt.args.ctx, tt.args.messages, tt.args.tools)
			if (err != nil) != tt.wantErr {
				t.Errorf("%s: OpenAIModel.GenerateContent() error = %v, wantErr %v", tt.name, err, tt.wantErr)
				return
			}
			_, content := got.Value()
			if !strings.Contains(content, tt.want.(string)) {
				t.Errorf("%s: OpenAIModel.GenerateContent() = %v, want %v", tt.name, got, tt.want)
			}
		})

		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.m.Generate(tt.args.ctx, tt.args.messages, tt.args.tools)
			if (err != nil) != tt.wantErr {
				t.Errorf("%s: OpenAIModel.GenerateContent() error = %v, wantErr %v", tt.name, err, tt.wantErr)
				return
			}
			_, content := got.Value()
			if !strings.Contains(content, tt.want.(string)) {
				t.Errorf("%s: OpenAIModel.GenerateContent() = %v, want %v", tt.name, got, tt.want)
			}
		})
	}
}

func DONRUN_BROKENTestModel_ProcessImage(t *testing.T) {
	testArgs := testArgs{
		ctx: context.Background(),
		messages: []ai.Message{
			ai.UserMessage{Role: ai.UserRole, Content: "Extract the text from this image and return the word SUCCESS if it worked followed by the text"},
			ai.ResourceMessage{Role: ai.UserRole, Name: "test.png", MIMEType: "image/png", Body: func() []byte {
				data, _ := testData.ReadFile("testdata/test.png")
				return data
			}()},
		},
		tools: []ai.Tool{},
	}

	tests := []struct {
		name    string
		m       *ai.Model
		want    any
		wantErr bool
	}{
		{
			name: "openai image processing",
			m:    NewOpenAIModel("gpt-4o-mini", os.Getenv("OPENAI_API_KEY")),
			want: "TEST",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.m.Call(testArgs.ctx, testArgs.messages, testArgs.tools)
			if (err != nil) != tt.wantErr {
				t.Errorf("%s: Model.Generate() error = %v, wantErr %v", tt.name, err, tt.wantErr)
				return
			}
			_, content := got.Value()
			t.Logf("%s: Model returned: %s", tt.name, content)
			if !strings.Contains(strings.ToLower(content), strings.ToLower(tt.want.(string))) {
				t.Errorf("%s: Model.Generate() did not find required word '%v' in response: %s", tt.name, tt.want, content)
			}
		})
	}
}

func TestModel_ProcessAttachments(t *testing.T) {
	testArgs := testArgs{
		ctx: context.Background(),
		messages: []ai.Message{
			ai.UserMessage{Role: ai.UserRole, Content: "Read the content of this text file and return the content of the file verbatim. do not add any other text"},
			ai.ResourceMessage{Role: ai.UserRole, Name: "sample.txt", MIMEType: "text/plain", Type: "text", Body: func() []byte {
				data, _ := testData.ReadFile("testdata/sample.txt")
				return data
			}()},
		},
		tools: []ai.Tool{},
	}

	tests := []struct {
		name    string
		m       *ai.Model
		want    any
		wantErr bool
	}{
		{
			name: "openai attachment processing",
			m:    NewOpenAIModel("gpt-4o-mini", os.Getenv("OPENAI_API_KEY")),
			want: "ATTACHMENT_SUCCESS",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.m.Call(testArgs.ctx, testArgs.messages, testArgs.tools)
			if (err != nil) != tt.wantErr {
				t.Errorf("%s: Model.Generate() error = %v, wantErr %v", tt.name, err, tt.wantErr)
				return
			}
			_, content := got.Value()
			t.Logf("%s: Model returned: %s", tt.name, content)
			if !strings.Contains(strings.ToLower(content), strings.ToLower(tt.want.(string))) {
				t.Errorf("%s: Model.Generate() did not find required word '%v' in response: %s", tt.name, tt.want, content)
			}
		})
	}
}

func TestModel_GenerateContentWithTools(t *testing.T) {

	args := testArgs{
		ctx: context.Background(),
		messages: []ai.Message{
			ai.UserMessage{Role: ai.UserRole, Content: "Please use the echo tool to echo back the text 'Hello, World!'"},
		},
		tools: []ai.Tool{
			echoTool,
		},
	}

	tests := []struct {
		name    string
		m       *ai.Model
		wantErr bool
	}{
		{
			name: "openai api with tools",
			m:    NewOpenAIModel("gpt-4o-mini", os.Getenv("OPENAI_API_KEY")),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.m.Call(args.ctx, args.messages, args.tools)
			if (err != nil) != tt.wantErr {
				t.Errorf("%s: Model.Generate() error = %v, wantErr %v", tt.name, err, tt.wantErr)
				return
			}

			// Check that it's an AIMessage with tool calls
			if len(got.ToolCalls) == 0 {
				t.Errorf("%s: Expected tool calls in response, got none", tt.name)
			} else {
				// Check that the tool call is for the echo tool
				found := false
				for _, toolCall := range got.ToolCalls {
					if toolCall.Name == "echo" {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("%s: Expected echo tool call, got: %v", tt.name, got.ToolCalls)
				}
			}
		})
	}
}

func TestModel_ContextTimeout(t *testing.T) {
	// Skip if no OpenAI API key is available for OpenAI tests
	openAIKey := os.Getenv("OPENAI_API_KEY")

	testCases := []struct {
		name        string
		model       *ai.Model
		timeout     time.Duration
		expectError bool
		errorType   string
		skipReason  string
	}{
		{
			name:        "OpenAI model with short timeout",
			model:       NewOpenAIModel("gpt-4o-mini", openAIKey),
			timeout:     1 * time.Millisecond, // Very short timeout
			expectError: true,
			errorType:   "context deadline exceeded",
			skipReason:  "OPENAI_API_KEY not set",
		},
		{
			name:        "OpenAI model with reasonable timeout",
			model:       NewOpenAIModel("gpt-4o-mini", openAIKey),
			timeout:     30 * time.Second, // Reasonable timeout
			expectError: false,
			skipReason:  "OPENAI_API_KEY not set",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Skip OpenAI tests if API key is not available
			if strings.Contains(tc.name, "OpenAI") && openAIKey == "" {
				t.Skip(tc.skipReason)
			}

			// Create a context with timeout
			ctx, cancel := context.WithTimeout(context.Background(), tc.timeout)
			defer cancel()

			// Create a simple message
			messages := []ai.Message{
				ai.UserMessage{
					Role:    ai.UserRole,
					Content: "What is the capital of France?",
				},
			}

			// Make the call
			start := time.Now()
			response, err := tc.model.Call(ctx, messages, []ai.Tool{})
			duration := time.Since(start)

			// Log the duration for debugging
			t.Logf("Call duration: %v", duration)

			if tc.expectError {
				// Should have an error
				if err == nil {
					t.Error("Expected error due to timeout, but got none")
					return
				}

				// Check error type
				errStr := err.Error()
				if !strings.Contains(strings.ToLower(errStr), strings.ToLower(tc.errorType)) {
					t.Errorf("Expected error containing '%s', got: %s", tc.errorType, errStr)
				}

				// Verify the call was actually interrupted (duration should be close to timeout)
				if duration > tc.timeout*2 {
					t.Errorf("Call took too long (%v) for timeout test with %v timeout", duration, tc.timeout)
				}
			} else {
				// Should not have an error
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
					return
				}

				// Should have a response
				if response.Content == "" {
					t.Error("Expected response content, got empty string")
				}

				// Verify the call completed within a reasonable time
				if duration > tc.timeout {
					t.Errorf("Call took longer than timeout (%v > %v)", duration, tc.timeout)
				}
			}
		})
	}
}

func TestModel_ContextCancellation(t *testing.T) {
	// Skip if no OpenAI API key is available for OpenAI tests
	openAIKey := os.Getenv("OPENAI_API_KEY")

	testCases := []struct {
		name        string
		model       *ai.Model
		cancelDelay time.Duration
		expectError bool
		errorType   string
		skipReason  string
	}{
		{
			name:        "OpenAI model with immediate cancellation",
			model:       NewOpenAIModel("gpt-4o-mini", openAIKey),
			cancelDelay: 1 * time.Millisecond,
			expectError: true,
			errorType:   "context canceled",
			skipReason:  "OPENAI_API_KEY not set",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Skip OpenAI tests if API key is not available
			if strings.Contains(tc.name, "OpenAI") && openAIKey == "" {
				t.Skip(tc.skipReason)
			}

			// Create a context that will be cancelled
			ctx, cancel := context.WithCancel(context.Background())

			// Create a simple message
			messages := []ai.Message{
				ai.UserMessage{
					Role:    ai.UserRole,
					Content: "What is the capital of France?",
				},
			}

			// Start the call in a goroutine
			var response ai.AIMessage
			var err error
			var wg sync.WaitGroup
			wg.Add(1)

			go func() {
				defer wg.Done()
				response, err = tc.model.Call(ctx, messages, []ai.Tool{})
			}()

			// Cancel the context after the specified delay
			time.Sleep(tc.cancelDelay)
			cancel()

			// Wait for the call to complete
			wg.Wait()

			if tc.expectError {
				// Should have an error
				if err == nil {
					t.Error("Expected error due to context cancellation, but got none")
					return
				}

				// Check error type
				errStr := err.Error()
				if !strings.Contains(strings.ToLower(errStr), strings.ToLower(tc.errorType)) {
					t.Errorf("Expected error containing '%s', got: %s", tc.errorType, errStr)
				}

				// Log that we got an error as expected
				t.Logf("Got expected error: %v", err)
			} else {
				// Should not have an error
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
					return
				}

				// Should have a response
				if response.Content == "" {
					t.Error("Expected response content, got empty string")
				}

				// Log the response for debugging
				t.Logf("Got successful response: %s", response.Content)
			}
		})
	}
}

func TestModel_GenerateWithTimeout(t *testing.T) {
	// Skip if no OpenAI API key is available for OpenAI tests
	openAIKey := os.Getenv("OPENAI_API_KEY")

	testCases := []struct {
		name        string
		model       *ai.Model
		timeout     time.Duration
		expectError bool
		errorType   string
		skipReason  string
	}{
		{
			name:        "OpenAI Generate with short timeout",
			model:       NewOpenAIModel("gpt-4o-mini", openAIKey),
			timeout:     1 * time.Millisecond,
			expectError: true,
			errorType:   "context deadline exceeded",
			skipReason:  "OPENAI_API_KEY not set",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Skip OpenAI tests if API key is not available
			if strings.Contains(tc.name, "OpenAI") && openAIKey == "" {
				t.Skip(tc.skipReason)
			}

			// Create a context with timeout
			ctx, cancel := context.WithTimeout(context.Background(), tc.timeout)
			defer cancel()

			// Create a simple message
			messages := []ai.Message{
				ai.UserMessage{
					Role:    ai.UserRole,
					Content: "What is the capital of France?",
				},
			}

			// Make the Generate call
			start := time.Now()
			response, err := tc.model.Generate(ctx, messages, []ai.Tool{})
			duration := time.Since(start)

			// Log the duration for debugging
			t.Logf("Generate call duration: %v", duration)

			if tc.expectError {
				// Should have an error
				if err == nil {
					t.Error("Expected error due to timeout, but got none")
					return
				}

				// Check error type
				errStr := err.Error()
				if !strings.Contains(strings.ToLower(errStr), strings.ToLower(tc.errorType)) {
					t.Errorf("Expected error containing '%s', got: %s", tc.errorType, errStr)
				}

				// Verify the call was actually interrupted
				if duration > tc.timeout*2 {
					t.Errorf("Generate call took too long (%v) for timeout test with %v timeout", duration, tc.timeout)
				}
			} else {
				// Should not have an error
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
					return
				}

				// Should have a response
				if response.Content == "" {
					t.Error("Expected response content, got empty string")
				}
			}
		})
	}
}

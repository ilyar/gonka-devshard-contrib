package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestUniqueTransportConcurrent(t *testing.T) {
	const original = `{"model":"fixture","stream":true,"max_tokens":256,"temperature":0.7,"messages":[{"role":"system","content":"keep"},{"role":"user","content":"repeat"}]}`
	seen := map[string]bool{}
	var mu sync.Mutex
	base := roundTripFunc(func(r *http.Request) (*http.Response, error) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			return nil, err
		}
		if r.ContentLength != int64(len(body)) {
			t.Error("wrong content length")
		}
		retry, err := r.GetBody()
		if err != nil {
			return nil, err
		}
		again, _ := io.ReadAll(retry)
		retry.Close()
		if !bytes.Equal(body, again) {
			t.Error("retry changed payload")
		}
		var data struct {
			Model       string
			Stream      bool
			MaxTokens   int `json:"max_tokens"`
			Temperature float64
			Messages    []struct{ Role, Content string }
		}
		if err := json.Unmarshal(body, &data); err != nil {
			return nil, err
		}
		if data.Model != "fixture" || !data.Stream || data.MaxTokens != 256 || data.Temperature != 0.7 || data.Messages[0].Content != "keep" {
			t.Error("changed request parameters")
		}
		content := data.Messages[1].Content
		if !strings.HasPrefix(content, "repeat\n[L:") {
			t.Error("missing marker")
		}
		mu.Lock()
		if seen[content] {
			t.Error("duplicate outgoing content")
		}
		seen[content] = true
		mu.Unlock()
		return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader("ok")), Header: make(http.Header)}, nil
	})
	for _, runID := range []string{strings.Repeat("a", 32), strings.Repeat("b", 32)} {
		transport := uniqueTransport{base: base, runID: runID}
		var wg sync.WaitGroup
		for i := 1; i <= 64; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				req, _ := http.NewRequest("POST", "http://localhost/v1/chat/completions", strings.NewReader(original))
				req.Header.Set("X-Request-Id", fmt.Sprintf("normal-load-42-%06d", i))
				resp, err := transport.RoundTrip(req)
				if err != nil {
					t.Error(err)
					return
				}
				resp.Body.Close()
			}(i)
		}
		wg.Wait()
	}
	if len(seen) != 128 {
		t.Fatalf("got %d unique requests", len(seen))
	}
}

func TestMalformedInputs(t *testing.T) {
	for _, body := range []string{"", "not-json", `{"model":"x","max_tokens":0}`, `{"unexpected":true}`} {
		path := filepath.Join(t.TempDir(), "input.jsonl")
		if err := os.WriteFile(path, []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
		if _, _, err := prepareDataset(path); err == nil {
			t.Fatalf("accepted invalid input %q", body)
		}
	}
}

func TestRunArgsHTTPAndSSERemoteAdapter(t *testing.T) {
	var mu sync.Mutex
	mode := "ok"
	var requests []map[string]any
	seen := map[string]bool{}
	cacheHits := 0

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		currentMode := mode
		mu.Unlock()

		switch r.URL.Path {
		case "/v1/models":
			w.WriteHeader(http.StatusOK)
			if currentMode == "missing" {
				_, _ = w.Write([]byte(`{"data":[]}`))
				return
			}
			_, _ = w.Write([]byte(`{"data":[{"id":"fixture"}]}`))
		case "/v1/chat/completions":
			raw, err := io.ReadAll(r.Body)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			var row map[string]any
			if err := json.Unmarshal(raw, &row); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			norm, _ := json.Marshal(row)
			mu.Lock()
			requests = append(requests, row)
			if seen[string(norm)] {
				cacheHits++
			}
			seen[string(norm)] = true
			mu.Unlock()

			if currentMode == "error" {
				w.WriteHeader(http.StatusInternalServerError)
			} else {
				w.WriteHeader(http.StatusOK)
			}
			if stream, _ := row["stream"].(bool); stream {
				_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n"))
			} else {
				_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
			}
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	root := t.TempDir()
	dataset := filepath.Join(root, "input.jsonl")
	resultsDir := filepath.Join(root, "results")
	scenarioPath := filepath.Join("..", "..", "loadtest", "scenarios", "normal-load.yaml")

	execute := func(stream, check, closedLoop, unique bool) error {
		row := map[string]any{
			"model":       "fixture",
			"stream":      stream,
			"max_tokens":  16,
			"temperature": 0.7,
			"response":    map[string]any{"role": "assistant", "content": "mock-only"},
			"messages":    []map[string]any{{"role": "user", "content": "fixture"}},
		}
		encoded, err := json.Marshal(row)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(dataset, append(encoded, '\n'), 0600); err != nil {
			t.Fatal(err)
		}
		args := []string{
			"-base", server.URL,
			"-dataset", dataset,
			"-scenario", scenarioPath,
			"-output-parent", resultsDir,
			"-duration", "300ms",
			"-rps", "4",
			"-request-timeout", "2s",
		}
		if check {
			args = append(args, "-check")
		}
		if closedLoop {
			args = append(args, "-closed-loop")
		}
		if !unique {
			args = append(args, "-unique-requests=false")
		}
		var out, errBuf bytes.Buffer
		err = runArgs(args, func(k string) string {
			if k == "GONKA_API_KEY" {
				return "fixture-not-a-secret"
			}
			return ""
		}, &out, &errBuf)
		if strings.Contains(out.String()+errBuf.String(), "fixture-not-a-secret") {
			t.Fatal("leaked secret to output")
		}
		return err
	}

	if err := execute(false, true, false, true); err != nil {
		t.Fatalf("preflight failed: %v", err)
	}
	mu.Lock()
	if len(requests) != 0 {
		t.Fatalf("preflight sent %d requests", len(requests))
	}
	mu.Unlock()

	if err := execute(false, false, false, true); err != nil {
		t.Fatalf("normal run failed: %v", err)
	}
	if err := execute(true, false, false, true); err != nil {
		t.Fatalf("stream run failed: %v", err)
	}
	if err := execute(false, false, true, true); err != nil {
		t.Fatalf("closed-loop run failed: %v", err)
	}

	mu.Lock()
	if len(requests) <= 3 {
		t.Fatalf("expected >3 requests, got %d", len(requests))
	}
	if cacheHits != 0 {
		t.Fatalf("expected 0 cache hits with unique-requests, got %d", cacheHits)
	}
	for _, req := range requests {
		if _, hasResp := req["response"]; hasResp {
			t.Fatal("response field must not be sent")
		}
		msgs, _ := req["messages"].([]any)
		last, _ := msgs[len(msgs)-1].(map[string]any)
		content, _ := last["content"].(string)
		if !strings.HasPrefix(content, "fixture\n[L:") {
			t.Fatalf("missing unique marker in %q", content)
		}
	}
	mu.Unlock()

	if err := execute(false, false, false, false); err != nil {
		t.Fatalf("non-unique run failed: %v", err)
	}
	mu.Lock()
	if cacheHits == 0 {
		t.Fatal("expected cache hits when unique-requests=false")
	}
	mode = "error"
	mu.Unlock()

	if err := execute(false, false, false, true); err == nil {
		t.Fatal("expected error when server returns 500")
	}

	mu.Lock()
	mode = "missing"
	before := len(requests)
	mu.Unlock()

	if err := execute(false, false, false, true); err == nil {
		t.Fatal("expected error when model is missing")
	}
	mu.Lock()
	if len(requests) != before {
		t.Fatal("sent requests when model preflight failed")
	}
	mu.Unlock()

	err := filepath.Walk(resultsDir, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return err
		}
		body, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if strings.Contains(string(body), "fixture-not-a-secret") {
			t.Fatalf("secret written to artifact %s", path)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

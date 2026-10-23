// Remote adapter uses RunGenerator, never RunScenario or Docker operations.
package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"devshard/testenv/loadtest"
	"devshard/testenv/replay"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// Keep replay selection unchanged, but make each HTTP body unique across runs.
// The upstream request ID carries the sequence stored in requests.jsonl, so the
// exact suffix can be reconstructed from run metadata and existing artifacts.
type uniqueTransport struct {
	base  http.RoundTripper
	runID string
}

func (t uniqueTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.Method != http.MethodPost || !strings.HasSuffix(req.URL.Path, "/v1/chat/completions") {
		return t.base.RoundTrip(req)
	}
	id := req.Header.Get("X-Request-Id")
	sequence, err := strconv.ParseUint(id[strings.LastIndex(id, "-")+1:], 10, 64)
	if err != nil || sequence == 0 {
		return nil, fmt.Errorf("unique requests: missing upstream sequence")
	}
	raw, err := io.ReadAll(req.Body)
	req.Body.Close()
	if err != nil {
		return nil, err
	}
	var body map[string]json.RawMessage
	if err := json.Unmarshal(raw, &body); err != nil {
		return nil, err
	}
	var messages []replay.Message
	if err := json.Unmarshal(body["messages"], &messages); err != nil {
		return nil, err
	}
	user := -1
	for i := range messages {
		if messages[i].Role == "user" {
			user = i
		}
	}
	if user < 0 {
		return nil, fmt.Errorf("unique requests: user message required")
	}
	messages[user].Content += fmt.Sprintf("\n[L:%s:%020d]", t.runID, sequence)
	body["messages"], err = json.Marshal(messages)
	if err != nil {
		return nil, err
	}
	encoded, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	clone := req.Clone(req.Context())
	clone.ContentLength = int64(len(encoded))
	clone.Body = io.NopCloser(bytes.NewReader(encoded))
	clone.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(encoded)), nil }
	return t.base.RoundTrip(clone)
}

func prepareDataset(path string) (*replay.Dataset, string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, "", err
	}
	data, err := replay.LoadFile(path)
	return data, fmt.Sprintf("%x", sha256.Sum256(raw)), err
}

func defaultPath(relFromTestenv string) string {
	for _, candidate := range []string{
		relFromTestenv,
		filepath.Join("testenv", relFromTestenv),
		filepath.Join("devshard", "testenv", relFromTestenv),
	} {
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
	}
	return relFromTestenv
}

func runArgs(args []string, getenv func(string) string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("wrap-load", flag.ContinueOnError)
	fs.SetOutput(stderr)
	base := fs.String("base", getenv("GONKA_BASE"), "gateway base URL without /v1 (defaults to GONKA_BASE)")
	dataset := fs.String("dataset", "", "upstream-compatible replay JSONL; model per row")
	scenarioPath := fs.String("scenario", defaultPath("loadtest/scenarios/normal-load.yaml"), "upstream normal-load scenario template")
	outputParent := fs.String("output-parent", defaultPath("loadtest/results"), "parent for unique private run directory")
	revision := fs.String("source-revision", "unknown", "upstream source commit recorded by wrapper")
	duration := fs.Duration("duration", 30*time.Second, "duration; seeded selection WITH replacement")
	timeout := fs.Duration("request-timeout", 120*time.Second, "per-request timeout")
	rps := fs.Float64("rps", 1, "offered requests per second")
	closedLoop := fs.Bool("closed-loop", false, "send immediately when a slot is free; no RPS cap")
	unique := fs.Bool("unique-requests", true, "append per-run/per-request marker to messages to avoid response-cache hits")
	concurrency := fs.Int("max-in-flight", 2, "maximum concurrent requests")
	seed := fs.Int64("seed", 42, "replay selection seed")
	check := fs.Bool("check", false, "validate input and model availability without inference")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("unexpected positional arguments")
	}
	u, err := url.Parse(*base)
	if err != nil || u.Host == "" || (u.Scheme != "https" && u.Scheme != "http") || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return fmt.Errorf("invalid gateway base URL")
	}
	if u.Scheme == "http" && u.Hostname() != "127.0.0.1" && u.Hostname() != "localhost" {
		return fmt.Errorf("HTTP allowed only for loopback tests")
	}
	if *duration <= 0 || *timeout <= 0 || *concurrency <= 0 || *rps <= 0 || math.IsNaN(*rps) || math.IsInf(*rps, 0) {
		return fmt.Errorf("duration, timeout, RPS and concurrency must be positive and finite")
	}
	key := strings.TrimSpace(getenv("GONKA_API_KEY"))
	if key == "" {
		return fmt.Errorf("set GONKA_API_KEY; no command-line credentials")
	}
	data, hash, err := prepareDataset(*dataset)
	if err != nil {
		return err
	}
	scenario, err := loadtest.LoadScenario(*scenarioPath)
	if err != nil {
		return err
	}
	scenario.Workload.Events = nil
	scenario.Workload.HostFaults = nil
	scenario.Cases = nil
	scenario.Diagnostics = loadtest.Diagnostics{}
	scenario.Seed = *seed
	scenario.Workload.Duration = duration.String()
	scenario.Workload.MaxInFlight = *concurrency
	scenario.Workload.Request.Timeout = timeout.String()
	scenario.Workload.Traffic = loadtest.TrafficProfile{Type: "constant", RPS: *rps}
	if *closedLoop {
		scenario.Workload.Traffic = loadtest.TrafficProfile{Type: "closed_loop"}
	}
	if err := scenario.Validate(); err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	client := &http.Client{Timeout: *timeout, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	var runID string
	if *unique {
		var nonce [16]byte
		if _, err := rand.Read(nonce[:]); err != nil {
			return err
		}
		runID = fmt.Sprintf("%x", nonce[:])
		client.Transport = uniqueTransport{base: http.DefaultTransport, runID: runID}
	}
	req, err := http.NewRequestWithContext(ctx, "GET", strings.TrimRight(*base, "/")+"/v1/models", nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+key)
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("model preflight HTTP %d", resp.StatusCode)
	}
	var listing struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&listing); err != nil {
		return err
	}
	available := map[string]bool{}
	for _, m := range listing.Data {
		available[m.ID] = true
	}
	for _, m := range data.Models() {
		if !available[m] {
			return fmt.Errorf("dataset model unavailable: %s", m)
		}
	}
	fmt.Fprintf(stdout, "Validated %d samples; seeded selection WITH replacement, not one pass.\n", data.Len())
	if *check {
		return nil
	}
	if *outputParent == "" {
		return fmt.Errorf("output-parent required")
	}
	if err := os.MkdirAll(*outputParent, 0700); err != nil {
		return err
	}
	out, err := os.MkdirTemp(*outputParent, "run-")
	if err != nil {
		return err
	}
	metadata := map[string]any{"source_revision": *revision, "dataset_sha256": hash, "models": data.Models(), "duration": duration.String(), "request_timeout": timeout.String(), "rps": *rps, "max_in_flight": *concurrency, "seed": *seed, "scope": "client HTTP/SSE only; no escrow, host or settlement assertions"}
	metadata["request_transform"] = "none; messages are loaded directly from dataset"
	metadata["unique_requests"] = *unique
	if *unique {
		metadata["run_id"] = runID
		metadata["request_transform"] = "append \\n[L:<run_id>:<20-digit upstream request sequence>] to last user message; source dataset unchanged"
	}
	metadata["traffic_type"] = scenario.Workload.Traffic.Type
	if *closedLoop {
		metadata["rps"] = nil
	}
	body, err := json.MarshalIndent(metadata, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(out, "remote-inputs.json"), body, 0600); err != nil {
		return err
	}
	fmt.Fprintln(stdout, "Artifacts:", out)
	summary, err := loadtest.RunGenerator(ctx, loadtest.GeneratorConfig{GatewayURL: strings.TrimRight(*base, "/"), APIKey: key, Scenario: scenario, OutputDir: out, Client: client, Replay: replay.NewSelector(data, *seed)})
	if err != nil {
		return err
	}
	fmt.Fprintf(stdout, "Completed=%d failed=%d dropped=%d\n", summary.Completed, summary.Failed, summary.Dropped)
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if summary.Completed == 0 || summary.Failed > 0 || summary.Dropped > 0 {
		return fmt.Errorf("client run unsuccessful; inspect artifacts")
	}
	return nil
}

func run() error {
	return runArgs(os.Args[1:], os.Getenv, os.Stdout, os.Stderr)
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "wrap-load:", err)
		os.Exit(1)
	}
}

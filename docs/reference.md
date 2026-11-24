# Reference

## Method

1. **Same requests.** The sample is drawn with a seed and frozen to `runs/<run>/tasks.json` with a sha256
   before anything is sent. Both providers get the same bytes, at the same time and concurrency. The same
   seed gives the same sample; `--replay <run>` re-sends a past run's exact tasks.
2. **No stale cache.** A per-run tag opens the system prompt and the first tool's description. Turns
   inside one session share their prefix on purpose.
3. **Provider numbers.** Token counts and cache hits come from each provider's own `usage`.
4. **Errors are counted, never hidden.** 429 / 5xx / connection errors are retried twice and counted.
   The budget is a hard stop; requests past it are recorded as skipped.
5. **No LLM judge.** Grading is deterministic code. A quality gap within ~2 standard errors is "level";
   speed or cost gaps under 10% are "about even".

## Scenarios

**Trace replay (`agentic`).** Real agent sessions: `source=swe-smith` (default,
[SWE-smith](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories)) or `source=agentx`
([raw Claude Code sessions](https://huggingface.co/datasets/semianalysisai/swebench-cc-traces-raw), 10
sessions, ~50k tokens of system prompt and tools). A window of `turns` consecutive steps is picked between
`min_context` and `max_context`. Open loop: each step sends the session so far, then continues with the
*recorded* step, so both providers get identical requests and tools never run. Not scored for accuracy;
a step fails on broken tool calls, empty, truncated or degenerate replies.

**Long agent session (`session`).** SWE-smith sessions concatenated to `start_context` (default 64k),
framed like a production coding agent (~20k tokens of real system prompts and MCP tool schemas) or
`framing=trace`. Each turn asks for heavy written work (root cause, fix, tests, PR text), up to 64k
output within `max_context`. `loop=open` (default) keeps requests identical; `loop=closed` carries the
model's own replies forward like a live session.

**Your data (`custom`).** Your JSONL via `--data` or the UI's Data page. Lines with assistant turns are
replayed like trace replay; lines without are sent as single requests. Format below.

**Synthetic workload (`synthetic`).** Seeded random words shaped by `isl`, `osl` (= max_tokens),
`cache_hit` (shared-prefix share), `turns` and `turn_isl`. The first session runs alone to warm the
prefix. Not graded; fails only on errors or stopping under half the requested output.

**Quality benchmarks.**

| Suite | Data | Checked by |
|---|---|---|
| `tools` | BFCL v4 simple, multiple, parallel, parallel-multiple, irrelevance | Function names, call count, arguments |
| `math` | MATH-500 + AIME 2025 | Last `\boxed{}`, normalised and compared numerically |
| `gpqa` | GPQA-Diamond, choices shuffled per seed | Answer letter (questions stay local; never publish them) |
| `robustness` | Serving-bug probes | `<think>` round-trip, `json_schema`, parallel tool calls, clean greeting |

Degeneration detectors (loop, word salad, gibberish, language switch, leaked special tokens,
low-entropy text, budget exhausted on reasoning) run on every suite except `synthetic` and add the
`degenerate` code. They point a human at text to read; the web UI's Review page records labels.

## Metrics

| Metric | Definition |
|---|---|
| TTFT | Request sent → first streamed token of any kind. Connection setup excluded. |
| Output tok/s | (output tokens − 1) ÷ (last token − first token); only with ≥ 32 tokens. |
| Throughput | All output tokens ÷ wall time of the run, per provider. |
| Cost | Uncached input × input price + cached × cached price + output × output price. Per request, per correct task, total. |
| Accuracy | Passed ÷ graded tasks (quality suites only). |
| Cache hit | Cached ÷ prompt tokens, from `usage`. |
| Cacheable | Prompt tokens this provider had already prefilled earlier in the run (exact prefix match). |
| **Cache reuse** | Cached ÷ cacheable, capped at 100%. Compare providers on this one. |

Failure codes: transport (`http_429`, `http_4xx`, `http_5xx`, `timeout`, `connection`, `empty_200`,
`stream_error`, `budget`), output (`truncated`, `short_output`, `tool_call_invalid_json`,
`tool_call_unknown_tool`, `tool_args_schema`, `missing_tool_call`, `unexpected_tool_call`,
`reasoning_leak`, `bad_json`, `degenerate`), correctness (`no_answer`, `wrong_answer`, `wrong_tool`,
`wrong_call_count`, `wrong_args`, `mismatch`). Labels: `providerbench/failures.py`.

## CLI

`./pb <command> --help` shows every option.

| Command | |
|---|---|
| `./pb run A B --suites S [--preview] [--seed N] [--budget USD] [--json]` | Run; `S` is `tools=30,math=20` or `"agentic:n=5,turns=3;synthetic:isl=2000"` |
| `./pb run A B --data f.jsonl --suites "custom:n=20"` | Your data |
| `./pb run --replay <run>` | Re-send a past run's exact tasks |
| `./pb run ... --mock` | Use the two fake local providers |
| `./pb show <run> [--json]` | Print a past run |
| `./pb export <run> [--format zip\|md\|html] [--out F]` | Raw zip, Markdown, or a self-contained HTML report |
| `./pb serve [--port 8765] [--mock]` | The web UI |
| `./pb suites` | Suites and their parameters |
| `./pb providers` · `./pb models` | Providers and key status · each provider's `/models` vs `models.toml` |
| `./pb provider add <id> <base_url> [--private] [--no-key]` · `check <id>` · `rm <id>` | Add a provider (`--private`: your own endpoint), re-check it, remove one of yours |
| `./pb prices [refresh]` · `./pb prices set <prov> <model> <in> <out> [--cached X] [--currency C]` | Prices per 1M tokens |
| `./pb data check f.jsonl` · `./pb data brief` | Validate your JSONL · print the conversion brief |

Other options for `run`: `--concurrency` (default 4), `--thinking default|off`, `--temperature`, `-n`.

## Providers and models

One file per provider in `providers/`; any OpenAI-compatible chat-completions endpoint works.

```toml
name = "Acme"
base_url = "https://api.acme.ai/v1"        # /chat/completions is appended; {ENV_VAR} / {model} are filled in
key_env = "ACME_API_KEY"                   # read from the environment or .env ("" = no key)

[headers]                                  # optional; {session} = one id per session (cache affinity)
x-session-affinity = "{session}"

[models]                                   # canonical name (models.toml) -> this provider's id
"glm-5.3-flash" = "acme/glm-5.3-flash"

[thinking_off]                             # merged when a run asks for thinking off; "*" = all models
"glm-5.3-flash" = { reasoning_effort = "none" }

[extra_body]                               # merged on every request for that model
"gemma-4-31b" = { chat_template_kwargs = { enable_thinking = true } }

[[pricing]]                                # tried in order; first price found wins
source = "models_endpoint"                 # price in GET /models (public = true if no key needed)
[[pricing]]
source = "openrouter"                      # this provider's OpenRouter listing (also gives quantization)
provider = "Acme"
[[pricing]]
source = "scrape"                          # regex with named groups model, input, output[, cached_input]
url = "https://acme.ai/pricing"
pattern = '''...'''
[[pricing]]
source = "manual"                          # set with pb prices set; never overwritten
```

### Your own endpoint (self-hosted, Bedrock, a gateway)

Any OpenAI-compatible endpoint can be added as a private provider, saved to
`~/.providerbench/providers/<id>.toml` (never committed). Every model its `/models` lists is offered.

```bash
./pb provider add my-vllm http://localhost:8000/v1 --private --no-key
./pb provider add bedrock https://bedrock-mantle.us-east-1.api.aws/v1 --private \
    --key-env AWS_BEARER_TOKEN_BEDROCK --model <bedrock model id>        # Bedrock API key in .env
./pb provider rm my-vllm
```

`--model` adds ids when the endpoint has no `/models`. In the web UI: **Providers → Add provider**, with a
Bedrock preset. Prices for your own endpoints are manual (`pb prices set`).

`models.toml` lists canonical models (label, context, OpenRouter slug, aliases). A model not listed there
can still be run as `provider:<raw model id>`. The key is only ever sent to the host in `base_url`.

## Your data format

One JSON object per line, OpenAI chat format (full brief: [data-brief.md](data-brief.md)):

```json
{"id": "fix-typo", "tools": [], "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"README.md\"}"}}]}, {"role": "tool", "tool_call_id": "c1", "content": "..."}]}
{"id": "one-shot", "messages": [{"role": "user", "content": "Summarise TCP vs UDP."}]}
```

`arguments` is a JSON string; every tool call is followed by its `tool` result; without `tools`, schemas
are inferred from the calls. Example: [examples/traces.example.jsonl](../examples/traces.example.jsonl).

## Where things live

| | |
|---|---|
| `.env` | API keys (git-ignored). Environment variables win. |
| `~/.providerbench/providers/` | Your own endpoints (self-hosted, Bedrock) |
| `~/.providerbench/runs/` | Every run: spec, frozen sample, every request and reply, review labels |
| `~/.providerbench/uploads/` | Your datasets |
| `~/.providerbench/prices.json`, `cache/` | Price snapshots, downloaded datasets |

Set `PB_HOME` to move `~/.providerbench`. The web server binds to 127.0.0.1 and refuses cross-origin
POSTs; it never returns a key.

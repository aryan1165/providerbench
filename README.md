# providerbench

Send the same frozen requests to two LLM inference providers at the same time, and compare speed, cost,
cache reuse and failures. Every request and reply is kept so you can check the numbers yourself.

Python 3.11+, standard library only. Nothing to install.

## Keys

```bash
cp .env.example .env      # fill in the providers you use; .env is git-ignored
```

Exported environment variables work too and take precedence over `.env`.

## Two ways to use it

**CLI (developers and coding agents).**

```bash
./pb run fireworks:glm-5.3-flash deepinfra:glm-5.3-flash --suites "agentic:n=10,turns=3" --preview
./pb run fireworks:glm-5.3-flash deepinfra:glm-5.3-flash --suites "agentic:n=10,turns=3" --seed <seed> --budget 2
./pb run mock-swift:glm-5.3-flash mock-steady:glm-5.3-flash --mock --suites tools=5   # free, no keys
./pb provider add my-vllm http://localhost:8000/v1 --private --no-key                  # your own endpoint
```

**Web UI (anyone, for quick tests and shareable reports).**

```bash
./pb serve            # http://127.0.0.1:8765
./pb serve --mock     # with two fake local providers, no keys, no cost
```

Pick two providers and a model, pick a scenario, preview, run. The **Providers** page saves keys to `.env`
and adds your own endpoints (self-hosted vLLM / SGLang / Ollama, Amazon Bedrock).
Each run can be exported as a self-contained HTML report or Markdown to share.

## Scenarios

| Scenario | Suite | What it tells you |
|---|---|---|
| Trace replay | `agentic` | Speed, cost and cache reuse on real agent sessions (SWE-smith, AgentX) |
| Long agent session | `session` | Decode speed and cost for long outputs at 64k+ context |
| Your data | `custom` | The same, on your own JSONL traffic |
| Synthetic workload | `synthetic` | Raw serving numbers for an ISL / OSL / cache-hit shape |
| Quality benchmarks | `tools`, `math`, `gpqa`, `robustness` | Known-answer accuracy: catches quantization and template bugs |

Details, metrics, CLI and provider files: [docs/reference.md](docs/reference.md).
Using a coding agent? Point it at [AGENTS.md](AGENTS.md).

## License

Apache-2.0, see [LICENSE](LICENSE). Third-party notices: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

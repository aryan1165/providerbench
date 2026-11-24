# For coding agents

providerbench sends the same frozen requests to two LLM inference providers and compares speed, cost,
cache behaviour and failures. Reference: [docs/reference.md](docs/reference.md).

## Ground rules

- **Never print, log, commit or paste an API key.** Keys come from the environment or `.env` in the repo
  root (git-ignored; template in `.env.example`). The user fills it in; don't read it back.
- **Real runs cost money.** Always `--preview` first, state the request count and estimate, keep `--budget`
  small, and get the user's go-ahead. Use `--mock` to try things for free.
- Python 3.11+ standard library only. No dependencies.

## Compare two providers

1. Ask which providers and model; check both serve it: `./pb providers`, `./pb models`.
2. Pick the suite for the question: agent traffic `agentic`; long reasoning at long context `session`;
   their own logs `custom` with `--data`; raw serving shape `synthetic`; correctness
   `tools,math,gpqa,robustness`.
3. `./pb run A B --suites "<suite:params>" --preview`; show the sample size and estimate.
4. After the go-ahead: same command with `--seed <seed from preview> --budget <usd> --json > run.json`.
5. Report from `targets[].summary`: cost per request (or per correct task), median TTFT and tok/s,
   `cache_reuse_pct`, failure codes with counts, and the sample size.

## Other recipes

- **Convert the user's logs:** follow [docs/data-brief.md](docs/data-brief.md), write `traces.jsonl`, strip
  secrets and personal data, then `./pb data check traces.jsonl` until it reports no errors.
- **Add a built-in provider (for a PR):** the user adds `<ID>_API_KEY` to `.env`; run `./pb provider add <id> <base_url>`,
  review the `[models]` matches, then `./pb provider check <id>` and `./pb prices refresh`. Only map
  model ids the provider actually lists.
- **Add the user's own endpoint** (self-hosted, Bedrock): `./pb provider add <id> <base_url> --private`
  (`--no-key` if it needs no auth, `--model <id>` if it has no `/models`). It stays out of the repo.
- **Add a model:** add a block to `models.toml`; `./pb models` prints the line to add to each provider.
- **Add a suite:** one module in `providerbench/suites/`, registered in `suites/__init__.py`, with offline
  grader tests.

## Checks before you finish

```bash
python3 -m unittest discover -s tests
./pb run mock-swift:glm-5.3-flash mock-steady:glm-5.3-flash --mock --suites tools=5,robustness=5
git status --short    # no .env, run data or exports
```

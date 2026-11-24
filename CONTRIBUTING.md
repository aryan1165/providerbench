# Contributing

- **Same requests, honest numbers.** Anything that changes what is sent or how it is measured needs a
  line in [docs/reference.md](docs/reference.md).
- **Local and private.** Keys stay in the environment or `.env`; run data in `~/.providerbench`. No telemetry.
- **No dependencies.** Python 3.11+ standard library; the UI is plain HTML/CSS/JS with no build step.
- **Deterministic grading.** No LLM judges.

```bash
python3 -m unittest discover -s tests             # offline, a few seconds
PB_NET=1 python3 -m unittest discover -s tests    # also downloads and samples every dataset
./pb serve --mock                                 # click through the UI with fake providers
```

A provider PR: one file in `providers/`, no keys, `base_url` on the provider's own domain, model ids that
point at the same weights as the canonical model, and the output of `./pb provider check <id>`.

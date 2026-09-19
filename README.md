# AutoJobSearch

AutoJobSearch is a local-first, open-source job discovery and application pipeline.
It finds jobs, applies deterministic filters, uses a local Ollama model for evidence-grounded
fit assessment, and prepares applications for explicit human review.

The project is intentionally review-gated. It does not bypass CAPTCHAs, evade anti-bot
systems, or mark an application submitted without positive confirmation evidence.

## Status

Early development. The safe discovery, scoring, persistence, and local-LLM foundation is
being built before live form submission is enabled.

## Principles

- Personal data and generated artifacts live under `~/.autojobsearch`, never in Git.
- Rules handle objective filters before an LLM is called.
- LLM responses use JSON schemas and are validated.
- Generated claims must cite facts from the applicant profile.
- Application submission is disabled until a user approves the prepared application.
- CAPTCHAs and ambiguous required questions pause for manual action.

## Development

```bash
uv sync --extra dev --extra browser
uv run autojobsearch init
uv run autojobsearch doctor
uv run pytest
```

Ollama defaults to `qwen3.5:9b` at `http://127.0.0.1:11434`.

```bash
ollama pull qwen3.5:9b
uv run autojobsearch doctor
```

Discover and score a public Greenhouse board:

```bash
uv run autojobsearch discover-greenhouse <board-token>
```

Use `--no-use-llm` to exercise discovery and deterministic filters without Ollama.

## Repository safety

Never commit real profiles, resumes, browser state, databases, screenshots, or application
artifacts. The `.gitignore` rejects these paths and only sanitized examples belong here.

## License

MIT

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

Discovery runs deterministic filters without Ollama by default. Add `--use-llm` to assess only
the jobs that pass those filters. The command prints a compact summary; add `--json` when full
machine-readable results are needed.

Routine fit scoring disables Qwen's extended reasoning mode and caps structured output so local
batch triage remains practical. Deterministic rules still run first, avoiding model calls for
jobs that clearly miss the configured title, location, seniority, or experience limits.

Prepare and optionally dry-fill an application without submitting it:

```bash
uv run autojobsearch prepare-application <job-id> <application-url> --fill
```

The command saves the exact plan and verification report under
`~/.autojobsearch/applications/<job-id>/`. Required questions that cannot be answered from an
explicit profile value remain unresolved and block approval. This command never clicks submit.
When the plan is ready, its full digest is saved as `approval-digest.txt`; blocked reruns remove
any stale digest. Resume and cover-letter paths belong in the private profile's `documents`
object. Exact custom question labels can be added to `approved_answers` using normalized
underscore keys such as `how_did_you_hear_about_this_job`.

After all required fields resolve, review the saved plan and use its digest to authorize exactly
that plan. Submission requires both the matching digest and an explicit confirmation flag:

```bash
uv run autojobsearch submit-application <job-id> <application-url> \
  --approval-digest <reviewed-digest> --confirm-submit
```

The plan is rebuilt from the live form before filling. Changed fields invalidate the digest,
unresolved required questions block execution, and a click is not reported as success unless a
positive confirmation page is observed.

## Repository safety

Never commit real profiles, resumes, browser state, databases, screenshots, or application
artifacts. The `.gitignore` rejects these paths and only sanitized examples belong here.

## License

MIT

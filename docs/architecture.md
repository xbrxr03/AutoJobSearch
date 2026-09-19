# Architecture

AutoJobSearch separates deterministic automation from model-assisted judgment.

1. Discovery adapters normalize public postings into `JobPosting` records.
2. Hard filters reject objective mismatches before an LLM is called.
3. Ollama returns schema-constrained assessments grounded in applicant fact IDs.
4. SQLite persists jobs, state transitions, and audit events outside the repository.
5. The browser layer scans, fills, and verifies ATS forms using deterministic Playwright actions.
6. Unknown questions, CAPTCHAs, and uncertain submission outcomes stop for human review.

The LLM does not receive unrestricted browser, filesystem, or shell control. It produces
validated decisions for bounded tasks. Browser execution remains explicit and auditable.

## Submission invariant

Clicking a submit button is not proof of success. A job can enter
`submission_confirmed` only when positive confirmation evidence has been recorded.

## Data boundary

Source code and sanitized fixtures live in Git. Profiles, resumes, cookies, databases,
screenshots, generated documents, and browser traces live under `~/.autojobsearch`.

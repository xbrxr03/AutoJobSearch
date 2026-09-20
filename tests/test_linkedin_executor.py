from pathlib import Path

import pytest

from autojobsearch.application.linkedin import (
    LinkedInOutcome,
    parse_linkedin_outcome,
    run_linkedin_easy_apply,
    validate_linkedin_job_url,
)
from autojobsearch.models import ApplicantProfile, JobStatus


def profile(resume: Path) -> ApplicantProfile:
    return ApplicantProfile.model_validate(
        {
            "person": {
                "first_name": "Jane",
                "last_name": "Doe",
                "email": "jane@example.com",
                "phone": "647-555-0100",
                "city": "Toronto",
                "region": "ON",
                "country": "Canada",
            },
            "preferences": {
                "target_titles": ["software engineer"],
                "locations": ["Toronto, ON"],
            },
            "documents": {"resume": str(resume)},
        }
    )


@pytest.mark.parametrize(
    ("text", "outcome", "status"),
    [
        (
            "APPLICATION_SUBMITTED: LinkedIn displayed confirmation.",
            LinkedInOutcome.APPLICATION_SUBMITTED,
            JobStatus.SUBMISSION_CONFIRMED,
        ),
        (
            "MANUAL_ACTION_REQUIRED: CAPTCHA shown.",
            LinkedInOutcome.MANUAL_ACTION_REQUIRED,
            JobStatus.MANUAL_ACTION,
        ),
        ("SKIP_ROLE: senior role", LinkedInOutcome.SKIP_ROLE, JobStatus.REJECTED),
        ("STOP_NO_MATCH", LinkedInOutcome.STOP_NO_MATCH, JobStatus.FAILED),
    ],
)
def test_parses_strict_terminal_outcomes(
    text: str, outcome: LinkedInOutcome, status: JobStatus
) -> None:
    assert parse_linkedin_outcome(text) == (outcome, status)


def test_ambiguous_or_unlabelled_result_is_never_success() -> None:
    assert parse_linkedin_outcome("I think the application went through") == (
        LinkedInOutcome.APPLICATION_UNCONFIRMED,
        JobStatus.UNCERTAIN,
    )
    assert parse_linkedin_outcome("Note: APPLICATION_SUBMITTED") == (
        LinkedInOutcome.APPLICATION_UNCONFIRMED,
        JobStatus.UNCERTAIN,
    )


def test_rejects_non_linkedin_and_non_jobs_urls() -> None:
    with pytest.raises(ValueError, match="LinkedIn"):
        validate_linkedin_job_url("https://example.com/jobs/1")
    with pytest.raises(ValueError, match="Jobs"):
        validate_linkedin_job_url("https://www.linkedin.com/feed/")


@pytest.mark.asyncio
async def test_runner_uses_persistent_profile_and_local_ollama_without_real_browser(
    tmp_path: Path,
) -> None:
    resume = tmp_path / "resume.pdf"
    resume.write_bytes(b"%PDF fixture")
    chrome = tmp_path / "Chrome"
    chrome.mkdir()
    captured: dict = {}

    class FakeHistory:
        def final_result(self) -> str:
            return "MANUAL_ACTION_REQUIRED: sign-in checkpoint"

        def save_to_file(self, path: Path) -> None:
            Path(path).write_text("{}", encoding="utf-8")

    class FakeAgent:
        def __init__(self, **kwargs) -> None:
            captured["agent"] = kwargs

        async def run(self, max_steps: int):
            captured["max_steps"] = max_steps
            return FakeHistory()

    def fake_browser(**kwargs):
        captured["browser"] = kwargs
        return object()

    def fake_llm(**kwargs):
        captured["llm"] = kwargs
        return object()

    result = await run_linkedin_easy_apply(
        url="https://www.linkedin.com/jobs/view/123/",
        profile=profile(resume),
        resume_path=resume,
        model="qwen-local",
        ollama_base_url="http://127.0.0.1:11434/",
        profile_dir=chrome,
        profile_name="Profile 2",
        artifact_dir=tmp_path / "artifacts",
        agent_factory=FakeAgent,
        browser_factory=fake_browser,
        llm_factory=fake_llm,
    )

    assert result.status == JobStatus.MANUAL_ACTION
    assert captured["browser"]["user_data_dir"] == chrome
    assert captured["browser"]["profile_directory"] == "Profile 2"
    assert captured["llm"]["model"] == "qwen-local"
    assert captured["llm"]["host"] == "http://127.0.0.1:11434"
    assert captured["agent"]["available_file_paths"] == [str(resume)]
    assert "Do not open or apply to any other listing" in captured["agent"]["task"]
    assert captured["max_steps"] == 40
    assert result.history_path.is_file()

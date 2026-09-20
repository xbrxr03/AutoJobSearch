from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autojobsearch.application.indeed_executor import (
    IndeedOutcome,
    classify_indeed_result,
    run_indeed_apply,
)
from autojobsearch.job_boards import JobBoard, JobBoardRunConfig
from autojobsearch.models import ApplicantProfile, JobStatus


@pytest.mark.parametrize(
    ("message", "outcome", "status"),
    [
        (
            "Receipt visible. APPLICATION_SUBMITTED",
            IndeedOutcome.APPLICATION_SUBMITTED,
            JobStatus.SUBMISSION_CONFIRMED,
        ),
        (
            "Challenge shown. MANUAL_ACTION_REQUIRED",
            IndeedOutcome.MANUAL_ACTION_REQUIRED,
            JobStatus.MANUAL_ACTION,
        ),
        (
            "Only external apply. SKIP_EXTERNAL_APPLY",
            IndeedOutcome.SKIP_EXTERNAL_APPLY,
            JobStatus.REJECTED,
        ),
        (
            "Submit clicked, receipt missing. APPLICATION_UNCONFIRMED",
            IndeedOutcome.APPLICATION_UNCONFIRMED,
            JobStatus.UNCERTAIN,
        ),
    ],
)
def test_classifies_bounded_outcomes(message, outcome, status) -> None:
    assert classify_indeed_result(message) == (outcome, status)


def test_ambiguous_or_unlabeled_output_is_uncertain() -> None:
    expected = (IndeedOutcome.APPLICATION_UNCONFIRMED, JobStatus.UNCERTAIN)
    assert classify_indeed_result("I think it worked") == expected
    assert classify_indeed_result("APPLICATION_SUBMITTED then MANUAL_ACTION_REQUIRED") == expected


def _profile(resume: Path) -> ApplicantProfile:
    return ApplicantProfile.model_validate(
        {
            "person": {
                "first_name": "Test",
                "last_name": "User",
                "email": "test@example.com",
                "phone": "555-0100",
                "city": "Toronto",
                "region": "ON",
                "country": "Canada",
            },
            "preferences": {"target_titles": ["Software Engineer"], "locations": ["Remote"]},
            "documents": {"resume": str(resume)},
        }
    )


@pytest.mark.asyncio
async def test_runner_uses_local_ollama_persistent_chrome_and_never_really_submits(
    tmp_path: Path,
) -> None:
    resume = tmp_path / "resume.pdf"
    resume.write_bytes(b"fixture")
    chrome = tmp_path / "Chrome"
    chrome.mkdir()
    history = MagicMock()
    history.final_result.return_value = "No match. STOP_NO_MATCH"
    history.save_to_file = lambda path: path.write_text("{}", encoding="utf-8")
    agent = AsyncMock()
    agent.run.return_value = history
    browser = AsyncMock()

    config = JobBoardRunConfig(
        board=JobBoard.INDEED,
        search_url="https://ca.indeed.com/jobs?q=software",
        resume_path=resume,
        max_applications=1,
        max_listings_to_check=2,
    )
    with (
        patch("browser_use.Browser", return_value=browser) as browser_cls,
        patch("browser_use.ChatOllama") as ollama_cls,
        patch("browser_use.Agent", return_value=agent),
    ):
        result = await run_indeed_apply(
            config=config,
            profile=_profile(resume),
            model="qwen-test",
            ollama_base_url="http://127.0.0.1:11434",
            profile_dir=chrome,
            profile_directory="Profile 2",
            artifact_dir=tmp_path / "artifacts",
        )

    assert result.outcome is IndeedOutcome.STOP_NO_MATCH
    assert result.status is JobStatus.REJECTED
    ollama_cls.assert_called_once_with(
        model="qwen-test",
        host="http://127.0.0.1:11434",
        timeout=180.0,
        ollama_options={"temperature": 0, "num_ctx": 32768},
    )
    assert browser_cls.call_args.kwargs["user_data_dir"] == chrome
    assert browser_cls.call_args.kwargs["profile_directory"] == "Profile 2"
    assert browser_cls.call_args.kwargs["allowed_domains"] == ["indeed.com", "*.indeed.com"]
    browser.kill.assert_awaited_once()

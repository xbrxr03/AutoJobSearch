import json
import subprocess
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
    (chrome / "Profile 2").mkdir()
    captured: dict = {}

    def fake_subprocess(command, *, input, cwd, env, timeout):
        captured["command"] = command
        captured["script"] = input
        captured["cwd"] = cwd
        captured["env"] = env
        captured["timeout"] = timeout
        config_path = tmp_path / "artifacts" / "browser-harness-config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        Path(config["history_path"]).write_text("{}", encoding="utf-8")
        Path(config["result_path"]).write_text(
            json.dumps(
                {
                    "ok": True,
                    "final_result": "MANUAL_ACTION_REQUIRED: sign-in checkpoint",
                    "history_path": config["history_path"],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    result = await run_linkedin_easy_apply(
        url="https://www.linkedin.com/jobs/view/123/",
        profile=profile(resume),
        resume_path=resume,
        model="qwen-local",
        ollama_base_url="http://127.0.0.1:11434/",
        profile_dir=chrome,
        profile_name="Profile 2",
        artifact_dir=tmp_path / "artifacts",
        subprocess_runner=fake_subprocess,
    )

    assert result.status == JobStatus.MANUAL_ACTION
    assert captured["command"] == ("uv", "run", "browser-use")
    assert captured["env"]["BH_TAB_MARKER"] == "0"
    assert captured["env"]["BH_HOME"] == str((tmp_path / "artifacts/browser-harness").resolve())
    assert "from browser_harness.daemon import get_ws_url" in captured["script"]
    assert "cdp_url=get_ws_url()" in captured["script"]
    config = json.loads((tmp_path / "artifacts" / "browser-harness-config.json").read_text())
    assert config["model"] == "qwen-local"
    assert config["ollama_base_url"] == "http://127.0.0.1:11434"
    assert config["allowed_domains"] == ["linkedin.com", "*.linkedin.com"]
    assert config["available_file_paths"] == [str(resume)]
    assert "Do not open or apply to any other listing" in config["task"]
    assert config["max_steps"] == 40
    assert result.history_path.is_file()

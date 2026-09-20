import json
import subprocess
from pathlib import Path

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
    (chrome / "Profile 2").mkdir()
    captured = {}

    config = JobBoardRunConfig(
        board=JobBoard.INDEED,
        search_url="https://ca.indeed.com/jobs?q=software",
        resume_path=resume,
        max_applications=1,
        max_listings_to_check=2,
    )

    def fake_subprocess(command, *, input, cwd, env, timeout):
        captured["command"] = command
        captured["script"] = input
        captured["env"] = env
        config_path = tmp_path / "artifacts" / "browser-harness-config.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        Path(payload["history_path"]).write_text("{}", encoding="utf-8")
        Path(payload["result_path"]).write_text(
            json.dumps(
                {
                    "ok": True,
                    "final_result": "No match. STOP_NO_MATCH",
                    "history_path": payload["history_path"],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    result = await run_indeed_apply(
        config=config,
        profile=_profile(resume),
        model="qwen-test",
        ollama_base_url="http://127.0.0.1:11434",
        profile_dir=chrome,
        profile_directory="Profile 2",
        artifact_dir=tmp_path / "artifacts",
        subprocess_runner=fake_subprocess,
    )

    assert result.outcome is IndeedOutcome.STOP_NO_MATCH
    assert result.status is JobStatus.REJECTED
    assert captured["command"] == ("uv", "run", "browser-use")
    assert captured["env"]["BH_TAB_MARKER"] == "0"
    assert "keep_alive=True" in captured["script"]
    harness_config = json.loads(
        (tmp_path / "artifacts" / "browser-harness-config.json").read_text()
    )
    assert harness_config["model"] == "qwen-test"
    assert harness_config["ollama_base_url"] == "http://127.0.0.1:11434"
    assert harness_config["allowed_domains"] == ["indeed.com", "*.indeed.com"]
    assert harness_config["available_file_paths"] == [str(resume)]
    assert harness_config["max_steps"] == 12

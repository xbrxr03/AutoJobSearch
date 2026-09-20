from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from autojobsearch.application.linkedin import LinkedInOutcome, LinkedInRunResult
from autojobsearch.cli import _canonical_job_url, app
from autojobsearch.models import JobPosting, JobStatus
from autojobsearch.storage import Store

runner = CliRunner()


def discovered_job() -> JobPosting:
    return JobPosting.model_validate(
        {
            "source": "fixture",
            "external_id": "1",
            "url": "https://example.com/jobs/1",
            "title": "Software Engineer",
            "company": "Example",
            "location": "Toronto, Ontario",
            "description": "Requires 2 years of Python experience.",
        }
    )


def initialize_home(home: Path) -> None:
    result = runner.invoke(app, ["init"], env={"AUTOJOBSEARCH_HOME": str(home)})
    assert result.exit_code == 0


def test_discovery_prints_summary_by_default(tmp_path: Path) -> None:
    initialize_home(tmp_path)
    with patch("autojobsearch.cli.GreenhouseDiscovery.discover", return_value=[discovered_job()]):
        result = runner.invoke(
            app,
            ["discover-greenhouse", "example", "--no-use-llm"],
            env={"AUTOJOBSEARCH_HOME": str(tmp_path)},
        )

    assert result.exit_code == 0
    assert "Discovered" in result.stdout
    assert "Shortlisted" in result.stdout
    assert '"external_id"' not in result.stdout


def test_discovery_json_flag_prints_full_results(tmp_path: Path) -> None:
    initialize_home(tmp_path)
    with patch("autojobsearch.cli.GreenhouseDiscovery.discover", return_value=[discovered_job()]):
        result = runner.invoke(
            app,
            ["discover-greenhouse", "example", "--no-use-llm", "--json"],
            env={"AUTOJOBSEARCH_HOME": str(tmp_path)},
        )

    assert result.exit_code == 0
    assert '"title": "Software Engineer"' in result.stdout


def test_submit_command_requires_explicit_confirmation() -> None:
    result = runner.invoke(
        app,
        [
            "submit-application",
            "1",
            "https://example.com/apply",
            "--approval-digest",
            "abc",
        ],
    )
    assert result.exit_code == 1
    assert "pass --confirm-submit" in result.stdout


def test_canonical_job_url_matches_lever_apply_route() -> None:
    posting = "https://jobs.lever.co/example/abc123"
    application = "https://jobs.lever.co/example/abc123/apply?source=site"

    assert _canonical_job_url(posting) == _canonical_job_url(application)


def test_linkedin_command_requires_explicit_confirmation() -> None:
    result = runner.invoke(app, ["linkedin-easy-apply", "1"])

    assert result.exit_code == 1
    assert "pass --confirm-submit" in result.stdout


def test_linkedin_command_audits_terminal_status_without_opening_browser(tmp_path: Path) -> None:
    initialize_home(tmp_path)
    resume = tmp_path / "resume.pdf"
    resume.write_bytes(b"%PDF fixture")
    profile_path = tmp_path / "profile.json"
    profile_text = profile_path.read_text(encoding="utf-8")
    profile_path.write_text(
        profile_text.replace('"resume": ""', f'"resume": "{resume}"'),
        encoding="utf-8",
    )
    chrome = tmp_path / "Chrome"
    chrome.mkdir()
    store = Store(tmp_path / "autojobsearch.sqlite3")
    job = JobPosting.model_validate(
        {
            **discovered_job().model_dump(),
            "url": "https://www.linkedin.com/jobs/view/123/",
            "source": "linkedin",
        }
    )
    job_id = store.upsert_job(job)
    history = tmp_path / "applications" / str(job_id) / "linkedin-browser-use-history.json"

    async def fake_run(**kwargs) -> LinkedInRunResult:
        history.parent.mkdir(parents=True, exist_ok=True)
        history.write_text("{}", encoding="utf-8")
        return LinkedInRunResult(
            outcome=LinkedInOutcome.MANUAL_ACTION_REQUIRED,
            status=JobStatus.MANUAL_ACTION,
            detail="MANUAL_ACTION_REQUIRED: sign-in checkpoint",
            history_path=history,
        )

    with patch("autojobsearch.cli.run_linkedin_easy_apply", side_effect=fake_run):
        result = runner.invoke(
            app,
            [
                "linkedin-easy-apply",
                str(job_id),
                "--confirm-submit",
                "--profile-dir",
                str(chrome),
            ],
            env={"AUTOJOBSEARCH_HOME": str(tmp_path)},
        )

    assert result.exit_code == 0
    assert "MANUAL_ACTION_REQUIRED (manual_action)" in result.stdout
    assert store.job_status(job_id) == JobStatus.MANUAL_ACTION
    with store.connect() as connection:
        events = connection.execute(
            "SELECT event_type FROM events WHERE job_id = ? ORDER BY id", (job_id,)
        ).fetchall()
    assert [event["event_type"] for event in events] == ["filling", "manual_action"]

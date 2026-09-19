from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from autojobsearch.cli import app
from autojobsearch.models import JobPosting

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

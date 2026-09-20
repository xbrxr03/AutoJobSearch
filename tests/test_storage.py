import sqlite3

from autojobsearch.models import JobPosting, JobStatus
from autojobsearch.storage import Store


def posting() -> JobPosting:
    return JobPosting(
        source="greenhouse",
        external_id="42",
        url="https://example.com/jobs/42?source=test",
        title="Software Engineer",
        company="Example",
        location="Toronto",
    )


def test_upsert_deduplicates_and_records_transition(tmp_path) -> None:
    store = Store(tmp_path / "test.sqlite3")
    store.initialize()
    first = store.upsert_job(posting())
    second = store.upsert_job(posting())
    assert first == second
    assert store.job_url(first) == "https://example.com/jobs/42?source=test"

    store.transition(first, JobStatus.SHORTLISTED, {"score": 88})
    assert store.job_status(first) is JobStatus.SHORTLISTED

    connection = sqlite3.connect(store.path)
    assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    assert connection.execute("SELECT status FROM jobs").fetchone()[0] == "shortlisted"
    assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1

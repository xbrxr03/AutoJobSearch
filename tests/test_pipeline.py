from autojobsearch.models import ApplicantProfile, FitAssessment, JobPosting, JobStatus
from autojobsearch.pipeline import Pipeline
from autojobsearch.storage import Store


class FakeLLM:
    def assess_fit(self, job: JobPosting, profile: ApplicantProfile) -> FitAssessment:
        return FitAssessment(
            score=82,
            recommendation="strong_match",
            matched_requirements=["Software engineering"],
            missing_requirements=[],
            evidence_fact_ids=["fact-1"],
            explanation="Strong title and location match.",
        )


def profile() -> ApplicantProfile:
    return ApplicantProfile.model_validate(
        {
            "person": {
                "first_name": "Jane",
                "last_name": "Doe",
                "email": "jane@example.com",
                "phone": "+1 555 010 0100",
                "city": "Toronto",
                "region": "Ontario",
                "country": "Canada",
            },
            "preferences": {
                "target_titles": ["software engineer"],
                "locations": ["Toronto"],
                "minimum_score": 60,
                "blocked_keywords": ["staff"],
            },
        }
    )


def test_pipeline_shortlists_with_llm_assessment(tmp_path) -> None:
    store = Store(tmp_path / "pipeline.sqlite3")
    store.initialize()
    pipeline = Pipeline(store, profile(), FakeLLM())
    job = JobPosting(
        source="fixture",
        external_id="1",
        url="https://example.com/jobs/1",
        title="Software Engineer",
        company="Example",
        location="Toronto",
    )
    _, rules, assessment = pipeline.ingest_and_score(job)
    assert rules.accepted
    assert assessment is not None
    assert assessment.score == 82


def test_pipeline_stops_before_llm_when_hard_filter_rejects(tmp_path) -> None:
    store = Store(tmp_path / "pipeline.sqlite3")
    store.initialize()
    pipeline = Pipeline(store, profile(), FakeLLM())
    job = JobPosting(
        source="fixture",
        external_id="2",
        url="https://example.com/jobs/2",
        title="Staff Software Engineer",
        company="Example",
        location="Toronto",
    )
    _, rules, assessment = pipeline.ingest_and_score(job)
    assert not rules.accepted
    assert assessment is None


def test_rediscovery_does_not_regress_confirmed_submission(tmp_path) -> None:
    store = Store(tmp_path / "pipeline.sqlite3")
    store.initialize()
    pipeline = Pipeline(store, profile(), None)
    job = JobPosting(
        source="fixture",
        external_id="3",
        url="https://example.com/jobs/3",
        title="Software Engineer",
        company="Example",
        location="Toronto",
    )
    job_id, _, _ = pipeline.ingest_and_score(job)
    store.transition(job_id, JobStatus.SUBMISSION_CONFIRMED)

    pipeline.ingest_and_score(job)

    assert store.job_status(job_id) is JobStatus.SUBMISSION_CONFIRMED

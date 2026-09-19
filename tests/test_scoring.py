from autojobsearch.models import ApplicantProfile, JobPosting
from autojobsearch.scoring import extract_required_years, hard_filter


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
                "locations": ["Toronto", "Remote Canada"],
                "remote_allowed": True,
                "max_required_years": 3,
                "minimum_score": 60,
                "blocked_keywords": ["senior", "staff"],
            },
        }
    )


def job(**overrides) -> JobPosting:
    data = {
        "source": "fixture",
        "external_id": "1",
        "url": "https://example.com/jobs/1",
        "title": "Software Engineer",
        "company": "Example",
        "location": "Toronto, Ontario",
        "description": "Requires 2 years of Python experience.",
    }
    data.update(overrides)
    return JobPosting.model_validate(data)


def test_extract_required_years_uses_smallest_requirement() -> None:
    assert extract_required_years("2 years Python and 5 years industry experience") == 2


def test_accepts_matching_entry_level_job() -> None:
    result = hard_filter(job(), profile())
    assert result.accepted
    assert result.score == 90


def test_rejects_blocked_seniority() -> None:
    result = hard_filter(job(title="Senior Software Engineer"), profile())
    assert not result.accepted
    assert result.score == 0


def test_rejects_experience_above_limit() -> None:
    result = hard_filter(job(description="Minimum 7+ years of experience."), profile())
    assert not result.accepted
    assert "Requires 7 years" in result.reasons[-1]

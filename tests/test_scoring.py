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


def test_extract_required_years_uses_highest_explicit_requirement() -> None:
    assert extract_required_years("2 years Python and 5 years industry experience") == 5


def test_extract_required_years_ignores_unrelated_company_age() -> None:
    description = "Operating for 20 years. This role requires 3 years of Python experience."
    assert extract_required_years(description) == 3


def test_extract_required_years_handles_have_and_word_numbers() -> None:
    assert extract_required_years("You have 5+ years building production systems.") == 5
    assert extract_required_years("You have at least two years' experience.") == 2


def test_accepts_matching_entry_level_job() -> None:
    result = hard_filter(job(), profile())
    assert result.accepted
    assert result.score == 90


def test_rejects_blocked_seniority() -> None:
    result = hard_filter(job(title="Senior Software Engineer"), profile())
    assert not result.accepted
    assert result.score == 0


def test_rejects_abbreviated_seniority() -> None:
    result = hard_filter(job(title="Sr. Software Engineer"), profile())
    assert not result.accepted
    assert result.score == 0


def test_seniority_word_in_description_does_not_block_entry_level_title() -> None:
    result = hard_filter(
        job(description="Collaborate with a product manager and senior stakeholders."), profile()
    )
    assert result.accepted


def test_rejects_senior_role_hidden_behind_generic_title() -> None:
    result = hard_filter(
        job(description="The Role: As a Senior Software Engineer, you will lead delivery."),
        profile(),
    )
    assert not result.accepted


def test_remote_only_requires_an_explicit_remote_signal() -> None:
    applicant = profile()
    applicant.preferences.remote_only = True

    rejected = hard_filter(job(), applicant)
    accepted = hard_filter(
        job(location="Remote Canada", description="This is a fully remote position in Canada."),
        applicant,
    )
    wrong_geography = hard_filter(
        job(location="New York", description="This is a fully remote position in the US."),
        applicant,
    )

    assert not rejected.accepted
    assert accepted.accepted
    assert not wrong_geography.accepted


def test_remote_only_accepts_country_named_in_title() -> None:
    applicant = profile()
    applicant.preferences.remote_only = True
    result = hard_filter(
        job(
            title="Canada - Software Engineer",
            location="Remote or Mississauga",
            description="This role may be performed remotely.",
        ),
        applicant,
    )
    assert result.accepted


def test_remote_only_rejects_hybrid_workplace_metadata() -> None:
    applicant = profile()
    applicant.preferences.remote_only = True
    result = hard_filter(
        job(
            location="Remote or Mississauga",
            description="Work from home is available.",
            metadata={"workplace_type": "hybrid"},
        ),
        applicant,
    )
    assert not result.accepted
    assert "hybrid" in result.reasons[-1]


def test_rejects_experience_above_limit() -> None:
    result = hard_filter(job(description="Minimum 7+ years of experience."), profile())
    assert not result.accepted
    assert "Requires 7 years" in result.reasons[-1]

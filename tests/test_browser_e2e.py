from pathlib import Path

import pytest

from autojobsearch.application import approve_plan, build_fill_plan
from autojobsearch.application.browser import ApplicationBrowser
from autojobsearch.models import ApplicantProfile, JobStatus


@pytest.mark.asyncio
async def test_local_fixture_scan_fill_verify_and_submit(tmp_path) -> None:
    resume = tmp_path / "resume.txt"
    resume.write_text("Sanitized test resume", encoding="utf-8")
    profile = ApplicantProfile.model_validate(
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
            "preferences": {"target_titles": ["developer"], "locations": ["Toronto"]},
            "approved_answers": {"requires_sponsorship": "No"},
            "documents": {"resume": str(resume)},
        }
    )
    fixture = Path(__file__).parent / "fixtures" / "simple_application.html"
    async with ApplicationBrowser(tmp_path / "chrome", headless=True) as browser:
        await browser.open(fixture.as_uri())
        fields = await browser.scan()
        assert (
            next(
                field for field in fields if field.selector == 'textarea[name="custom-question"]'
            ).label
            == "Why this role?"
        )
        plan = build_fill_plan(1, fields, profile)
        assert plan.ready_for_review
        verification = await browser.fill(plan)
        assert all(item.matched for item in verification)
        result = await browser.submit(plan, approve_plan(plan), verification)

    assert result.status == JobStatus.SUBMISSION_CONFIRMED
    assert result.submitted

from pathlib import Path

from autojobsearch.job_boards import (
    JobBoard,
    JobBoardQuery,
    JobBoardRunConfig,
    build_indeed_search_url,
    build_job_board_task,
    build_linkedin_search_url,
    default_board_queries,
)
from autojobsearch.models import ApplicantProfile


def profile() -> ApplicantProfile:
    return ApplicantProfile.model_validate(
        {
            "person": {
                "first_name": "Abrar",
                "last_name": "Habib",
                "email": "abrar@example.com",
                "phone": "647-555-0100",
                "city": "Scarborough",
                "region": "ON",
                "country": "Canada",
                "linkedin_url": "https://www.linkedin.com/in/example",
                "github_url": "https://github.com/example",
            },
            "preferences": {
                "target_titles": ["software engineering intern"],
                "locations": ["Toronto, ON"],
                "max_required_years": 2,
                "blocked_keywords": ["sales"],
            },
            "approved_answers": {
                "work_authorization": "Authorized to work in Canada",
                "sponsorship": "No",
            },
        }
    )


def test_builds_indeed_search_url() -> None:
    url = build_indeed_search_url(
        JobBoardQuery(
            keywords="software engineering intern",
            location="Toronto, ON",
            days_old=7,
        )
    )

    assert url.startswith("https://ca.indeed.com/jobs?")
    assert "q=software+engineering+intern" in url
    assert "l=Toronto%2C+ON" in url
    assert "fromage=7" in url


def test_builds_linkedin_easy_apply_search_url() -> None:
    url = build_linkedin_search_url(
        JobBoardQuery(keywords="junior python developer", location="Ontario, Canada")
    )

    assert url.startswith("https://www.linkedin.com/jobs/search/?")
    assert "keywords=junior+python+developer" in url
    assert "f_AL=true" in url
    assert "sortBy=DD" in url


def test_default_board_queries_come_from_profile_preferences() -> None:
    queries = default_board_queries(profile())

    assert len(queries) == 1
    assert queries[0].keywords == "software engineering intern"
    assert queries[0].location == "Toronto, ON"


def test_indeed_task_blocks_external_apply_and_captcha_bypass() -> None:
    task = build_job_board_task(
        JobBoardRunConfig(
            board=JobBoard.INDEED,
            search_url="https://ca.indeed.com/jobs?q=software",
            resume_path=Path("/tmp/resume.pdf"),
            max_applications=1,
            max_listings_to_check=3,
        ),
        profile(),
    )

    assert "Apply with Indeed" in task
    assert "Never click \"Apply on company site\"" in task
    assert "MANUAL_ACTION_REQUIRED" in task
    assert "Do not try to solve or bypass" in task
    assert "Check at most 3 listings" in task
    assert "work_authorization: Authorized to work in Canada" in task


def test_linkedin_task_requires_easy_apply_modal() -> None:
    task = build_job_board_task(
        JobBoardRunConfig(
            board=JobBoard.LINKEDIN,
            search_url="https://www.linkedin.com/jobs/search/?keywords=software&f_AL=true",
            resume_path=Path("/tmp/resume.pdf"),
        ),
        profile(),
    )

    assert "Easy Apply" in task
    assert "Only click \"Easy Apply\" buttons" in task
    assert "Do not click external \"Apply\" buttons" in task
    assert "already applied" in task

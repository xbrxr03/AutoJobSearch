from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from urllib.parse import urlencode

from pydantic import BaseModel, Field, HttpUrl

from .models import ApplicantProfile


class JobBoard(StrEnum):
    INDEED = "indeed"
    LINKEDIN = "linkedin"


class JobBoardQuery(BaseModel):
    keywords: str
    location: str
    remote: bool = False
    days_old: int = Field(default=7, ge=1, le=30)
    easy_apply_only: bool = True


class JobBoardRunConfig(BaseModel):
    board: JobBoard
    search_url: HttpUrl
    resume_path: Path
    max_applications: int = Field(default=1, ge=1, le=5)
    max_listings_to_check: int = Field(default=5, ge=1, le=25)


def build_indeed_search_url(query: JobBoardQuery, *, country_domain: str = "ca") -> str:
    params = {
        "q": query.keywords,
        "l": "" if query.remote else query.location,
        "radius": "50",
        "sort": "date",
        "fromage": str(query.days_old),
    }
    return f"https://{country_domain}.indeed.com/jobs?{urlencode(params)}"


def build_linkedin_search_url(query: JobBoardQuery) -> str:
    params = {
        "keywords": query.keywords,
        "location": "Canada" if query.remote else query.location,
        "sortBy": "DD",
    }
    if query.easy_apply_only:
        params["f_AL"] = "true"
    return f"https://www.linkedin.com/jobs/search/?{urlencode(params)}"


def default_board_queries(profile: ApplicantProfile) -> list[JobBoardQuery]:
    location = profile.preferences.locations[0] if profile.preferences.locations else (
        ", ".join(
            item
            for item in (
                profile.person.city,
                profile.person.region,
                profile.person.country,
            )
            if item
        )
        or "Canada"
    )
    return [
        JobBoardQuery(keywords=title, location=location, remote=False)
        for title in profile.preferences.target_titles
    ]


def build_job_board_task(config: JobBoardRunConfig, profile: ApplicantProfile) -> str:
    if config.board == JobBoard.INDEED:
        return _build_indeed_task(config, profile)
    if config.board == JobBoard.LINKEDIN:
        return _build_linkedin_task(config, profile)
    raise ValueError(f"Unsupported job board: {config.board}")


def _common_answers(profile: ApplicantProfile) -> str:
    person = profile.person
    answers = [
        f"- First name: {person.first_name}",
        f"- Last name: {person.last_name}",
        f"- Email: {person.email}",
        f"- Phone: {person.phone}",
        f"- Location: {person.city}, {person.region}, {person.country}",
    ]
    if person.linkedin_url:
        answers.append(f"- LinkedIn: {person.linkedin_url}")
    if person.github_url:
        answers.append(f"- GitHub: {person.github_url}")
    if person.portfolio_url:
        answers.append(f"- Portfolio: {person.portfolio_url}")
    for key, value in sorted(profile.approved_answers.items()):
        answers.append(f"- {key}: {value}")
    return "\n".join(answers)


def _target_context(profile: ApplicantProfile) -> str:
    target_titles = ", ".join(profile.preferences.target_titles)
    locations = ", ".join(profile.preferences.locations)
    blocked = ", ".join(profile.preferences.blocked_keywords)
    lines = [
        f"- Target titles: {target_titles or 'software roles matching profile preferences'}",
        f"- Accepted locations: {locations or 'profile location, remote Canada'}",
        f"- Remote allowed: {profile.preferences.remote_allowed}",
        f"- Maximum required years: {profile.preferences.max_required_years}",
    ]
    if blocked:
        lines.append(f"- Blocked keywords: {blocked}")
    return "\n".join(lines)


def _safety_rules(config: JobBoardRunConfig) -> str:
    return "\n".join(
        [
            "Critical rules:",
            f"1. Apply to at most {config.max_applications} job(s), then stop.",
            f"2. Check at most {config.max_listings_to_check} listings, then report STOP_NO_MATCH.",
            (
                "3. If a CAPTCHA, reCAPTCHA, SMS/email identity check, or suspicious-login "
                "prompt appears, report MANUAL_ACTION_REQUIRED and stop. Do not try to solve "
                "or bypass it."
            ),
            "4. Do not apply to jobs outside the target role and location rules.",
            (
                "5. Do not invent answers. If a required field cannot be answered from the "
                "approved answers, report MANUAL_ACTION_REQUIRED."
            ),
            "6. After one successful submission, verify a visible Indeed receipt, report "
            "APPLICATION_SUBMITTED, and stop.",
        ]
    )


def _build_indeed_task(config: JobBoardRunConfig, profile: ApplicantProfile) -> str:
    return f"""You are using a real logged-in Chrome profile to search Indeed and apply only
through Apply with Indeed.

{_safety_rules(config)}
7. Never click "Apply on company site" or any external redirect. Report
SKIP_EXTERNAL_APPLY for that listing and continue.

Navigate to:
{config.search_url}

Find a valid listing:
{_target_context(profile)}
- Must have "Apply with Indeed", "Easily apply", or equivalent inline Indeed apply flow.
- Must not be expired or closed.
- Skip listings marked already applied. Never resubmit an application.

Use these approved answers:
{_common_answers(profile)}

Resume:
{config.resume_path}

Completion labels you may report:
APPLICATION_SUBMITTED, STOP_NO_MATCH, SKIP_LOCATION, SKIP_ROLE, SKIP_EXTERNAL_APPLY,
MANUAL_ACTION_REQUIRED, APPLICATION_UNCONFIRMED.
End your final response with exactly one of these labels. Use APPLICATION_UNCONFIRMED if
Submit was clicked but no visible confirmation appeared; never click Submit again."""


def _build_linkedin_task(config: JobBoardRunConfig, profile: ApplicantProfile) -> str:
    return f"""You are using a real logged-in Chrome profile to search LinkedIn and apply only
through Easy Apply.

{_safety_rules(config)}
7. Only click "Easy Apply" buttons that open a LinkedIn modal.
Do not click external "Apply" buttons.
8. Move through the Easy Apply modal with Next, Review, and Submit application when all
required answers are known.

Navigate to:
{config.search_url}

Find a valid listing:
{_target_context(profile)}
- Must show Easy Apply.
- Must not be promoted-only noise, expired, closed, or already applied.

Use these approved answers:
{_common_answers(profile)}

Resume:
{config.resume_path}

Completion labels you may report:
APPLICATION_SUBMITTED, STOP_NO_MATCH, SKIP_LOCATION, SKIP_ROLE, SKIP_EXTERNAL_APPLY,
MANUAL_ACTION_REQUIRED."""

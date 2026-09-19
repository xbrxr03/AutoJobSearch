from __future__ import annotations

import re

from .models import ApplicantProfile, HardFilterResult, JobPosting

YEARS_PATTERN = re.compile(r"(\d{1,2})\+?\s+years?", re.IGNORECASE)
EXPERIENCE_SIGNAL = re.compile(
    r"\b(?:experience|experienced|minimum|required|requires?|qualifications?|at least|must have)\b",
    re.IGNORECASE,
)


def _contains_any(text: str, values: list[str]) -> list[str]:
    folded = text.casefold()
    return [value for value in values if value.casefold() in folded]


def extract_required_years(description: str) -> int | None:
    requirements: list[int] = []
    for match in YEARS_PATTERN.finditer(description):
        context_start = max(
            description.rfind(".", 0, match.start()),
            description.rfind("!", 0, match.start()),
            description.rfind("?", 0, match.start()),
            description.rfind("\n", 0, match.start()),
        )
        endings = [
            position
            for delimiter in ".!?\n"
            if (position := description.find(delimiter, match.end())) != -1
        ]
        context_end = min(endings) if endings else len(description)
        context = description[context_start + 1 : context_end]
        if EXPERIENCE_SIGNAL.search(context):
            requirements.append(int(match.group(1)))
    return max(requirements) if requirements else None


def hard_filter(job: JobPosting, profile: ApplicantProfile) -> HardFilterResult:
    prefs = profile.preferences
    title_and_description = f"{job.title}\n{job.description}"
    reasons: list[str] = []
    score = 50

    company_blocks = _contains_any(job.company, prefs.blocked_companies)
    if company_blocks:
        return HardFilterResult(
            accepted=False, score=0, reasons=[f"Blocked company: {company_blocks[0]}"]
        )

    blocked = _contains_any(title_and_description, prefs.blocked_keywords)
    if blocked:
        return HardFilterResult(accepted=False, score=0, reasons=[f"Blocked keyword: {blocked[0]}"])

    title_matches = _contains_any(job.title, prefs.target_titles)
    if title_matches:
        score += 25
        reasons.append(f"Target title match: {title_matches[0]}")
    else:
        score -= 20
        reasons.append("No target title match")

    location_matches = _contains_any(job.location, prefs.locations)
    if location_matches:
        score += 15
        reasons.append(f"Preferred location: {location_matches[0]}")
    elif job.is_remote and prefs.remote_allowed:
        score += 15
        reasons.append("Remote role accepted")
    else:
        score -= 20
        reasons.append("Location does not match preferences")

    years = extract_required_years(job.description)
    if years is not None and years > prefs.max_required_years:
        return HardFilterResult(
            accepted=False,
            score=max(0, score - 30),
            reasons=[*reasons, f"Requires {years} years; maximum is {prefs.max_required_years}"],
        )
    if years is not None:
        reasons.append(f"Experience requirement within limit: {years} years")

    score = min(100, max(0, score))
    return HardFilterResult(accepted=score >= prefs.minimum_score, score=score, reasons=reasons)

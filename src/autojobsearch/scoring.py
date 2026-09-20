from __future__ import annotations

import re

from .models import ApplicantProfile, HardFilterResult, JobPosting

YEAR_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
YEARS_PATTERN = re.compile(
    rf"(\d{{1,2}}|{'|'.join(YEAR_WORDS)})\+?\s+years?", re.IGNORECASE
)
EXPERIENCE_SIGNAL = re.compile(
    r"\b(?:experience|experienced|minimum|required|requires?|qualifications?|at least|"
    r"must have|have)\b",
    re.IGNORECASE,
)
REMOTE_SIGNAL = re.compile(
    r"\b(?:fully remote|remote[- ]first|remote position|remote role|work from anywhere|"
    r"canada\s*\(remote\)|remote(?:\s*-|\s+in)?\s*canada|remote or)\b",
    re.IGNORECASE,
)


def _contains_any(text: str, values: list[str]) -> list[str]:
    folded = text.casefold()
    return [value for value in values if value.casefold() in folded]


def _blocked_title_keyword(title: str, blocked_keywords: list[str]) -> str | None:
    matches = _contains_any(title, blocked_keywords)
    if matches:
        return matches[0]
    if any(keyword.casefold() == "senior" for keyword in blocked_keywords) and re.search(
        r"\bsr\.?\b", title, re.IGNORECASE
    ):
        return "senior"
    return None


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
            value = match.group(1).casefold()
            requirements.append(int(value) if value.isdigit() else YEAR_WORDS[value])
    return max(requirements) if requirements else None


def hard_filter(job: JobPosting, profile: ApplicantProfile) -> HardFilterResult:
    prefs = profile.preferences
    reasons: list[str] = []
    score = 50

    company_blocks = _contains_any(job.company, prefs.blocked_companies)
    if company_blocks:
        return HardFilterResult(
            accepted=False, score=0, reasons=[f"Blocked company: {company_blocks[0]}"]
        )

    blocked_keyword = _blocked_title_keyword(job.title, prefs.blocked_keywords)
    # Some boards publish a neutral title while the opening paragraph reveals the true level.
    if not blocked_keyword:
        lede = job.description[:500]
        for keyword in prefs.blocked_keywords:
            escaped = re.escape(keyword)
            if re.search(
                rf"\b(?:as an?|looking for an?|role is for an?)\b.{{0,50}}\b{escaped}\b",
                lede,
                re.IGNORECASE,
            ):
                blocked_keyword = keyword
                break
    if blocked_keyword:
        return HardFilterResult(
            accepted=False, score=0, reasons=[f"Blocked keyword: {blocked_keyword}"]
        )

    title_matches = _contains_any(job.title, prefs.target_titles)
    if title_matches:
        score += 25
        reasons.append(f"Target title match: {title_matches[0]}")
    else:
        score -= 20
        reasons.append("No target title match")

    inferred_remote = bool(
        job.is_remote
        or REMOTE_SIGNAL.search(job.location)
        or REMOTE_SIGNAL.search(job.description)
    )
    workplace_type = str(job.metadata.get("workplace_type") or "").casefold()
    if prefs.remote_only and workplace_type in {"hybrid", "on-site", "onsite"}:
        return HardFilterResult(
            accepted=False,
            score=max(0, score - 30),
            reasons=[*reasons, f"Remote-only search; workplace type is {workplace_type}"],
        )
    if prefs.remote_only and not inferred_remote:
        return HardFilterResult(
            accepted=False,
            score=max(0, score - 30),
            reasons=[*reasons, "Remote-only search; role is not explicitly remote"],
        )

    location_matches = _contains_any(job.location, prefs.locations)
    location_folded = job.location.strip().casefold()
    country_in_location = profile.person.country.casefold() in location_folded
    generic_remote_location = not location_folded or location_folded in {
        "remote",
        "worldwide",
    } or location_folded.startswith("remote or")
    remote_geography_matches = bool(
        location_matches
        or country_in_location
        or _contains_any(job.title, [*prefs.locations, profile.person.country])
    )
    if generic_remote_location:
        remote_geography_matches = remote_geography_matches or bool(
            _contains_any(job.description, [*prefs.locations, profile.person.country])
        )
    if prefs.remote_only and inferred_remote and not remote_geography_matches:
        return HardFilterResult(
            accepted=False,
            score=max(0, score - 30),
            reasons=[*reasons, "Remote role does not explicitly accept a preferred geography"],
        )
    if location_matches:
        score += 15
        reasons.append(f"Preferred location: {location_matches[0]}")
    elif inferred_remote and prefs.remote_allowed:
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

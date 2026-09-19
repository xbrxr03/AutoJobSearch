from __future__ import annotations

import re

from ..models import ApplicantProfile, ApplicationPlan, FillAction, FormField


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _profile_values(profile: ApplicantProfile) -> list[tuple[tuple[str, ...], str, str]]:
    person = profile.person
    return [
        (("first name", "given name"), person.first_name, "profile.person.first_name"),
        (("last name", "surname", "family name"), person.last_name, "profile.person.last_name"),
        (
            ("full name", "your name", "name"),
            f"{person.first_name} {person.last_name}",
            "profile.person",
        ),
        (("email", "email address"), person.email, "profile.person.email"),
        (("phone", "telephone", "mobile"), person.phone, "profile.person.phone"),
        (("city",), person.city, "profile.person.city"),
        (("state", "province", "region"), person.region, "profile.person.region"),
        (("country",), person.country, "profile.person.country"),
        (("linkedin",), person.linkedin_url, "profile.person.linkedin_url"),
        (("github",), person.github_url, "profile.person.github_url"),
        (("portfolio", "website"), person.portfolio_url, "profile.person.portfolio_url"),
    ]


def _approved_answer(label: str, profile: ApplicantProfile) -> tuple[str, str] | None:
    normalized_label = _normalized(label)
    mappings = {
        "authorized_to_work": ("authorized to work", "legally authorized", "eligible to work"),
        "requires_sponsorship": ("sponsorship", "sponsor", "visa support"),
    }
    for key, phrases in mappings.items():
        if key in profile.approved_answers and any(
            phrase in normalized_label for phrase in phrases
        ):
            return profile.approved_answers[key], f"profile.approved_answers.{key}"
    return None


def build_fill_plan(
    job_id: int, fields: list[FormField], profile: ApplicantProfile
) -> ApplicationPlan:
    actions: list[FillAction] = []
    unresolved: list[FormField] = []
    values = _profile_values(profile)

    for field in fields:
        label = _normalized(field.label)
        match: tuple[str, str] | None = None
        for phrases, value, source in values:
            if value and any(phrase == label or phrase in label for phrase in phrases):
                match = value, source
                break
        if match is None:
            match = _approved_answer(field.label, profile)

        if match is None:
            unresolved.append(field)
            continue

        value, source = match
        actions.append(
            FillAction(
                selector=field.selector,
                label=field.label,
                value=value,
                source=source,
                requires_review=field.field_type in {"radio", "select", "checkbox"},
            )
        )

    return ApplicationPlan(job_id=job_id, actions=actions, unresolved=unresolved)

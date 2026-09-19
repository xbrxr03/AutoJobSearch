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
        (
            ("location city", "current location"),
            f"{person.city}, {person.region}, {person.country}",
            "profile.person.location",
        ),
        (
            ("current city state and country", "city state and country"),
            f"{person.city}, {person.region}, {person.country}",
            "profile.person.location",
        ),
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
    for key, value in profile.approved_answers.items():
        if _normalized(key) == normalized_label:
            return value, f"profile.approved_answers.{key}"
    return None


def build_fill_plan(
    job_id: int, fields: list[FormField], profile: ApplicantProfile
) -> ApplicationPlan:
    actions: list[FillAction] = []
    unresolved: list[FormField] = []
    values = _profile_values(profile)

    for field in fields:
        label = _normalized(field.label)
        match = _approved_answer(field.label, profile)
        if field.field_type == "file":
            selector = _normalized(field.selector)
            if "resume" in selector and profile.documents.resume:
                match = profile.documents.resume, "profile.documents.resume"
            elif "cover letter" in selector and profile.documents.cover_letter:
                match = profile.documents.cover_letter, "profile.documents.cover_letter"
        for phrases, value, source in values:
            if match is not None:
                break
            if value and any(
                phrase == label or (phrase not in {"name", "country"} and phrase in label)
                for phrase in phrases
            ):
                match = value, source
                break

        if match is None:
            unresolved.append(field)
            continue

        value, source = match
        if field.field_type in {"select", "combobox"}:
            if not field.options and source in {
                "profile.person.country",
                "profile.person.location",
            }:
                exact_option = value
            else:
                exact_option = next(
                    (option for option in field.options if option.casefold() == value.casefold()),
                    None,
                )
            if exact_option is None:
                prefix_matches = [
                    option
                    for option in field.options
                    if option.casefold().startswith(f"{value.casefold()} ")
                ]
                exact_option = prefix_matches[0] if len(prefix_matches) == 1 else None
            if exact_option is None:
                unresolved.append(field)
                continue
            value = exact_option
        actions.append(
            FillAction(
                selector=field.selector,
                label=field.label,
                value=value,
                source=source,
                requires_review=field.field_type
                in {"radio", "select", "combobox", "checkbox", "file"},
            )
        )

    return ApplicationPlan(job_id=job_id, actions=actions, unresolved=unresolved)

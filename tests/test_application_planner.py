from autojobsearch.application import build_fill_plan
from autojobsearch.models import ApplicantProfile, FormField


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
            "preferences": {"target_titles": ["developer"], "locations": ["Toronto"]},
            "approved_answers": {
                "authorized_to_work": "Yes",
                "requires_sponsorship": "No",
            },
        }
    )


def test_planner_uses_profile_and_approved_answers() -> None:
    fields = [
        FormField(selector="#first", label="First name", field_type="text", required=True),
        FormField(selector="#email", label="Email address", field_type="email", required=True),
        FormField(
            selector="#sponsor",
            label="Will you require visa sponsorship?",
            field_type="select",
            required=True,
            options=["Yes", "No"],
        ),
    ]
    plan = build_fill_plan(1, fields, profile())
    assert plan.ready_for_review
    assert [action.value for action in plan.actions] == ["Jane", "jane@example.com", "No"]
    assert plan.actions[-1].requires_review


def test_planner_blocks_readiness_on_unknown_required_field() -> None:
    fields = [
        FormField(
            selector="#clearance",
            label="Government clearance level",
            field_type="select",
            required=True,
        )
    ]
    plan = build_fill_plan(1, fields, profile())
    assert not plan.ready_for_review
    assert plan.unresolved == fields


def test_planner_does_not_guess_legal_name_question() -> None:
    field = FormField(
        selector="#legal-name",
        label="Legal Name (if different than above)",
        field_type="text",
        required=True,
    )
    plan = build_fill_plan(1, [field], profile())
    assert plan.actions == []
    assert plan.unresolved == [field]


def test_planner_formats_compound_location_question() -> None:
    field = FormField(
        selector="#current-location",
        label="Please indicate your current city, state, and country.",
        field_type="text",
        required=True,
    )
    plan = build_fill_plan(1, [field], profile())
    assert plan.actions[0].value == "Toronto, Ontario, Canada"


def test_planner_requires_exact_combobox_option() -> None:
    field = FormField(
        selector="#location",
        label="Location (City)",
        field_type="combobox",
        required=True,
        options=["Toronto, Ontario, Canada", "Toronto, Ohio, United States"],
    )
    plan = build_fill_plan(1, [field], profile())
    assert plan.actions == []
    assert plan.unresolved == [field]


def test_planner_accepts_unique_country_option_with_dial_code() -> None:
    field = FormField(
        selector="#country",
        label="Country",
        field_type="combobox",
        required=True,
        options=["United States +1", "Canada +1", "United Kingdom +44"],
    )
    plan = build_fill_plan(1, [field], profile())
    assert plan.actions[0].value == "Canada +1"


def test_planner_uses_exact_custom_approved_answer() -> None:
    applicant = profile().model_copy(deep=True)
    applicant.approved_answers["how_did_you_hear_about_this_job"] = "GitHub"
    field = FormField(
        selector="#source",
        label="How did you hear about this job?",
        field_type="text",
        required=True,
    )
    plan = build_fill_plan(1, [field], applicant)
    assert plan.actions[0].value == "GitHub"


def test_planner_maps_resume_by_selector() -> None:
    applicant = profile().model_copy(deep=True)
    applicant.documents.resume = "/private/resume.pdf"
    field = FormField(selector="#resume", label="Attach", field_type="file")
    plan = build_fill_plan(1, [field], applicant)
    assert plan.actions[0].value == "/private/resume.pdf"
    assert plan.actions[0].requires_review

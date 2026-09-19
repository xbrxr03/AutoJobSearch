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

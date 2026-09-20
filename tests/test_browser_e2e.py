import json

from autojobsearch.application import build_fill_plan
from autojobsearch.application.browser_use_scanner import parse_scanned_fields
from autojobsearch.models import ApplicantProfile


def test_local_fixture_scan_and_plan_contract_never_launches_or_submits(tmp_path) -> None:
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
    fields = parse_scanned_fields(
        json.dumps(
            [
                {
                    "tag": "input",
                    "type": "text",
                    "id": "first-name",
                    "name": "first_name",
                    "required": True,
                    "label": "First name",
                    "options": [],
                },
                {
                    "tag": "input",
                    "type": "email",
                    "id": "email",
                    "name": "email",
                    "required": True,
                    "label": "Email",
                    "options": [],
                },
                {
                    "tag": "textarea",
                    "type": "textarea",
                    "id": "custom-question",
                    "name": "custom-question",
                    "required": False,
                    "label": "Why this role?",
                    "options": [],
                },
            ]
        )
    )

    custom = next(field for field in fields if field.selector == 'textarea[name="custom-question"]')
    assert custom.label == "Why this role?"

    plan = build_fill_plan(1, fields, profile)
    assert [(action.label, action.value) for action in plan.actions] == [
        ("First name", "Jane"),
        ("Email", "jane@example.com"),
    ]
    assert plan.unresolved == [custom]
    assert plan.ready_for_review

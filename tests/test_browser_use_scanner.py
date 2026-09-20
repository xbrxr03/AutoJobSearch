import json

from autojobsearch.application.browser_use_executor import allowed_domains, application_url
from autojobsearch.application.browser_use_scanner import parse_scanned_fields


def test_application_url_targets_direct_ats_forms() -> None:
    assert application_url("https://jobs.lever.co/acme/123") == (
        "https://jobs.lever.co/acme/123/apply"
    )
    assert application_url("https://jobs.ashbyhq.com/acme/123") == (
        "https://jobs.ashbyhq.com/acme/123/application"
    )


def test_allowed_domains_follow_target_ats() -> None:
    assert "*.ashbyhq.com" in allowed_domains("https://jobs.ashbyhq.com/acme/123")
    assert "*.lever.co" in allowed_domains("https://jobs.lever.co/acme/123")


def test_parse_scanned_fields_ignores_hidden_controls() -> None:
    raw = json.dumps(
        [
            {
                "tag": "input",
                "type": "hidden",
                "id": "token",
                "name": "token",
                "required": False,
                "label": "",
                "options": [],
            },
            {
                "tag": "select",
                "type": "select-one",
                "id": "",
                "name": "eligible",
                "required": True,
                "label": "Authorized to work?",
                "options": ["Select...", "Yes", "No"],
            },
        ]
    )
    fields = parse_scanned_fields(raw)
    assert len(fields) == 1
    assert fields[0].selector == 'select[name="eligible"]'
    assert fields[0].required

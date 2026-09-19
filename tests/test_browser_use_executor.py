from autojobsearch.application.browser_use_executor import build_browser_use_task
from autojobsearch.models import ApplicationPlan, FillAction


def test_browser_use_task_is_locked_to_reviewed_values() -> None:
    plan = ApplicationPlan(
        job_id=7,
        actions=[
            FillAction(
                selector="#email",
                label="Email",
                value="jane@example.com",
                source="profile.person.email",
            )
        ],
    )
    task = build_browser_use_task("https://jobs.example.test/apply", plan)
    assert "jane@example.com" in task
    assert "do not invent" in task
    assert "APPLICATION_SUBMITTED" in task
    assert "Never apply to another job" in task
    assert "wait for autocomplete suggestions" in task


def test_browser_use_task_documents_reviewed_lever_location_alias() -> None:
    plan = ApplicationPlan(
        job_id=8,
        actions=[
            FillAction(
                selector="#location-input",
                label="Current location",
                value="Scarborough, Ontario, Canada",
                source="profile.person.location",
            )
        ],
    )

    task = build_browser_use_task("https://jobs.lever.co/example/apply", plan)

    assert "Toronto, ON, CAN" in task
    assert "handled by the pipeline" in task

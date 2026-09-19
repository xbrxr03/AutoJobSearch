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

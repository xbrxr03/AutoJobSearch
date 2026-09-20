import sys
from types import ModuleType, SimpleNamespace

import pytest

from autojobsearch.application import approve_plan
from autojobsearch.application.browser_use_executor import (
    browser_use_result_status,
    build_browser_use_task,
    submit_with_browser_use,
)
from autojobsearch.application.review import ReviewRequiredError
from autojobsearch.models import ApplicationPlan, FillAction, JobStatus


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

    assert "Scarborough, Ontario, Canada" in task
    assert "handled by the pipeline" in task


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ("APPLICATION_SUBMITTED", JobStatus.SUBMISSION_CONFIRMED),
        ("APPLICATION_SUBMITTED: visible receipt", JobStatus.SUBMISSION_CONFIRMED),
        ("APPLICATION_UNCONFIRMED: no receipt", JobStatus.UNCERTAIN),
        ("I did not report APPLICATION_SUBMITTED", JobStatus.UNCERTAIN),
        ("NOT APPLICATION_SUBMITTED", JobStatus.UNCERTAIN),
    ],
)
def test_browser_use_result_requires_an_explicit_success_token(result, expected) -> None:
    assert browser_use_result_status(result) == expected


@pytest.mark.asyncio
async def test_executor_rejects_unapproved_plan_before_importing_browser_use(
    tmp_path, monkeypatch
) -> None:
    plan = ApplicationPlan(job_id=42)
    monkeypatch.setitem(sys.modules, "browser_use", None)

    with pytest.raises(ReviewRequiredError, match="explicit plan approval"):
        await submit_with_browser_use(
            url="https://jobs.example.test/apply",
            plan=plan,
            approval=None,
            model="qwen-local",
            ollama_base_url="http://127.0.0.1:11434",
            profile_dir=tmp_path / "profile",
            artifact_dir=tmp_path / "artifacts",
        )


@pytest.mark.asyncio
async def test_executor_contract_uses_local_ollama_and_real_profile_without_false_confirmation(
    tmp_path, monkeypatch
) -> None:
    captured = {}

    class CompletedEvent:
        def __await__(self):
            async def done():
                return self

            return done().__await__()

        async def event_result(self, **_kwargs):
            return None

    class Event:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Browser:
        def __init__(self, **kwargs):
            captured["browser"] = kwargs
            self.event_bus = SimpleNamespace(dispatch=lambda _event: CompletedEvent())

        async def start(self):
            captured["started"] = True

        async def kill(self):
            captured["killed"] = True

    class ChatOllama:
        def __init__(self, **kwargs):
            captured["ollama"] = kwargs

    class History:
        def final_result(self):
            return "NOT APPLICATION_SUBMITTED"

        def save_to_file(self, path):
            captured["history_path"] = path

    class Agent:
        def __init__(self, **kwargs):
            captured["agent"] = kwargs

        async def run(self, *, max_steps):
            captured["max_steps"] = max_steps
            return History()

    browser_use = ModuleType("browser_use")
    browser_use.Agent = Agent
    browser_use.Browser = Browser
    browser_use.ChatOllama = ChatOllama
    events = ModuleType("browser_use.browser.events")
    for name in (
        "ClickElementEvent",
        "NavigateToUrlEvent",
        "ScrollToTextEvent",
        "SelectDropdownOptionEvent",
        "SendKeysEvent",
        "TypeTextEvent",
        "UploadFileEvent",
    ):
        setattr(events, name, Event)
    browser_package = ModuleType("browser_use.browser")
    browser_package.events = events
    monkeypatch.setitem(sys.modules, "browser_use", browser_use)
    monkeypatch.setitem(sys.modules, "browser_use.browser", browser_package)
    monkeypatch.setitem(sys.modules, "browser_use.browser.events", events)

    plan = ApplicationPlan(job_id=42)
    status, result = await submit_with_browser_use(
        url="https://jobs.example.test/apply",
        plan=plan,
        approval=approve_plan(plan),
        model="qwen-local",
        ollama_base_url="http://127.0.0.1:11434/",
        profile_dir=tmp_path / "real-profile",
        artifact_dir=tmp_path / "artifacts",
    )

    assert status == JobStatus.UNCERTAIN
    assert result == "NOT APPLICATION_SUBMITTED"
    assert captured["browser"]["user_data_dir"] == tmp_path / "real-profile"
    assert captured["browser"]["profile_directory"] == "Default"
    assert captured["browser"]["headless"] is False
    assert captured["ollama"]["host"] == "http://127.0.0.1:11434"
    assert captured["ollama"]["model"] == "qwen-local"
    assert captured["agent"]["browser"].__class__ is Browser
    assert captured["max_steps"] == 4

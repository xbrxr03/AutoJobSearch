from __future__ import annotations

import asyncio
import json
from pathlib import Path
from urllib.parse import urlparse

from ..models import ApplicationPlan, JobStatus, PlanApproval
from .review import validate_approval


def _lever_location_alias(url: str, plan: ApplicationPlan) -> tuple[str, str] | None:
    if urlparse(url).hostname != "jobs.lever.co":
        return None
    reviewed_location = next(
        (
            action.value.strip().casefold()
            for action in plan.actions
            if "current location" in action.label.casefold()
        ),
        "",
    )
    if reviewed_location == "scarborough, ontario, canada":
        return "Toronto", "Toronto, ON, CAN"
    return None


def build_browser_use_task(url: str, plan: ApplicationPlan) -> str:
    fields = [{"label": action.label.strip(), "value": action.value} for action in plan.actions]
    location_alias = _lever_location_alias(url, plan)
    location_rule = (
        "The Current location field is handled by the pipeline before you begin. Verify it is "
        "populated, but do not type into it or change it. Lever represents Scarborough, Ontario "
        "as Toronto, ON, CAN; this is the approved autocomplete normalization for the "
        "applicant's Toronto municipality."
        if location_alias
        else "For Current location, typing text is not enough: wait for autocomplete suggestions "
        "and click the suggestion representing the approved city, region, and country."
    )
    return f"""Submit exactly one reviewed job application at {url}.

Use only the field values in this JSON array; do not invent, rewrite, or omit required answers:
{json.dumps(fields, indent=2)}

Rules:
1. Navigate only within the application site and its CAPTCHA provider.
2. Upload the file path supplied for the Resume/CV field.
3. Leave optional fields absent from the JSON blank, including pronouns and demographics.
4. {location_rule}
5. For native select fields, use select_dropdown with the exact approved option; never use click.
6. Verify every filled value, the selected location suggestion, and successful resume upload.
7. If an hCaptcha or other visual CAPTCHA appears, complete it through the visible browser UI.
8. Click the final submit button exactly once, then wait 10 seconds and inspect the result.
   Never click submit a second time in the same run.
9. Report APPLICATION_SUBMITTED only after the page visibly confirms receipt with text such as
   'application submitted', 'thank you for applying', or 'application received'.
10. If no positive receipt is visible after the wait, report APPLICATION_UNCONFIRMED and explain
    the blocker without another submit attempt.
11. Stop immediately after the first submission attempt. Never apply to another job.
12. Batch multiple independent field-entry actions into each step when the controls are visible.
"""


async def submit_with_browser_use(
    *,
    url: str,
    plan: ApplicationPlan,
    approval: PlanApproval,
    model: str,
    ollama_base_url: str,
    profile_dir: Path,
    artifact_dir: Path,
) -> tuple[JobStatus, str]:
    validate_approval(plan, approval)

    from browser_use import Agent, Browser, ChatOllama
    from browser_use.browser.events import ClickElementEvent, NavigateToUrlEvent, SendKeysEvent

    resume_paths = [
        action.value
        for action in plan.actions
        if "resume" in action.label.casefold() and Path(action.value).is_file()
    ]
    browser = Browser(
        executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        user_data_dir=profile_dir,
        profile_directory="Default",
        headless=False,
        keep_alive=False,
        allowed_domains=["jobs.lever.co", "*.lever.co", "*.hcaptcha.com", "hcaptcha.com"],
    )
    llm = ChatOllama(
        model=model,
        host=ollama_base_url.rstrip("/"),
        timeout=180.0,
        ollama_options={"temperature": 0, "num_ctx": 32768},
    )
    location_alias = _lever_location_alias(url, plan)

    async def set_reviewed_lever_location(query: str, expected: str) -> None:
        state = await browser.get_browser_state_summary(include_screenshot=False)
        location = next(
            (
                node
                for node in state.dom_state.selector_map.values()
                if node.attributes.get("id") == "location-input"
            ),
            None,
        )
        if location is None:
            raise RuntimeError("Lever location input was not found")

        click = browser.event_bus.dispatch(ClickElementEvent(node=location))
        await click
        await click.event_result(raise_if_any=True, raise_if_none=False)

        # Lever's React autocomplete ignores bulk text insertion. Real key events reliably
        # populate its suggestion list. Scarborough is part of the City of Toronto, and
        # Lever's location database exposes the municipality only as Toronto, ON, CAN.
        for character in query:
            keypress = browser.event_bus.dispatch(SendKeysEvent(keys=character))
            await keypress
            await keypress.event_result(raise_if_any=True, raise_if_none=False)
            await asyncio.sleep(0.12)

        await asyncio.sleep(1.5)
        state = await browser.get_browser_state_summary(include_screenshot=False)
        suggestion = next(
            (
                node
                for node in state.dom_state.selector_map.values()
                if node.attributes.get("id", "").startswith("location-")
                and expected in node.get_all_children_text()
            ),
            None,
        )
        if suggestion is None:
            raise RuntimeError(f"Reviewed Lever location suggestion was not found: {expected}")

        choose = browser.event_bus.dispatch(ClickElementEvent(node=suggestion))
        await choose
        await choose.event_result(raise_if_any=True, raise_if_none=False)
        await asyncio.sleep(0.5)

        state = await browser.get_browser_state_summary(include_screenshot=False)
        selected = next(
            (
                node
                for node in state.dom_state.selector_map.values()
                if node.attributes.get("id") == "location-input"
            ),
            None,
        )
        selected_value = (
            selected.snapshot_node.input_value
            if selected is not None and selected.snapshot_node is not None
            else None
        )
        if selected_value != expected:
            raise RuntimeError(
                f"Lever location selection was not retained: {selected_value!r}"
            )

    await browser.start()
    try:
        navigation = browser.event_bus.dispatch(NavigateToUrlEvent(url=url, new_tab=False))
        await navigation
        await navigation.event_result(raise_if_any=True, raise_if_none=False)
        await asyncio.sleep(1.0)
        if location_alias:
            await set_reviewed_lever_location(*location_alias)
    except Exception:
        await browser.kill()
        raise

    agent = Agent(
        task=build_browser_use_task(url, plan),
        llm=llm,
        browser=browser,
        available_file_paths=resume_paths,
        use_vision=True,
        use_judge=False,
        use_thinking=False,
        enable_planning=False,
        llm_timeout=180,
        step_timeout=240,
        llm_screenshot_size=(1024, 768),
        vision_detail_level="low",
        max_history_items=8,
        directly_open_url=False,
        extend_system_message=(
            "The supplied job-application task is your only objective. Never navigate to any "
            "unrelated URL or act on unrelated remembered/example tasks. The reviewed field "
            "values are immutable."
        ),
        max_actions_per_step=3,
        max_failures=5,
    )
    history = await agent.run(max_steps=40)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    history.save_to_file(artifact_dir / "browser-use-history.json")
    final_result = history.final_result() or "APPLICATION_UNCONFIRMED: no final result"
    confirmed = "APPLICATION_SUBMITTED" in final_result.upper()
    status = JobStatus.SUBMISSION_CONFIRMED if confirmed else JobStatus.UNCERTAIN
    return status, final_result

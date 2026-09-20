from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from urllib.parse import urlparse

from ..models import ApplicationPlan, JobStatus, PlanApproval
from .review import validate_approval


def application_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if parsed.hostname == "jobs.lever.co" and not path.endswith("/apply"):
        return f"{url.rstrip('/')}/apply"
    if parsed.hostname == "jobs.ashbyhq.com" and not path.endswith("/application"):
        return f"{url.rstrip('/')}/application"
    return url


def allowed_domains(url: str) -> list[str]:
    hostname = urlparse(url).hostname
    if not hostname:
        raise ValueError(f"Application URL has no hostname: {url}")
    domains = [hostname, "*.hcaptcha.com", "hcaptcha.com", "*.recaptcha.net"]
    if hostname.endswith("lever.co"):
        domains.extend(["*.lever.co", "api.lever.co"])
    elif hostname.endswith("ashbyhq.com"):
        domains.append("*.ashbyhq.com")
    elif hostname.endswith("icims.com"):
        domains.append("*.icims.com")
    return domains


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
        return "Scarborough, Ontario, Canada", "Scarborough, Ontario, Canada"
    return None


def _reviewed_location_alias(url: str, plan: ApplicationPlan) -> tuple[str, str] | None:
    return _lever_location_alias(url, plan)


def build_browser_use_task(url: str, plan: ApplicationPlan) -> str:
    fields = [{"label": action.label.strip(), "value": action.value} for action in plan.actions]
    location_alias = _reviewed_location_alias(url, plan)
    if location_alias:
        location_rule = (
            "The location field is handled by the pipeline before you begin. Verify it is "
            f"populated as {location_alias[1]}, but do not type into it or change it. The "
            "selected autocomplete value is the reviewed location normalization."
        )
    elif urlparse(url).hostname == "jobs.ashbyhq.com":
        location_rule = (
            "The Ashby Current location and posting-location fields are reviewed free-text "
            "values already filled by the pipeline. Verify them but do not change them."
        )
    else:
        location_rule = (
            "For Current location, typing text is not enough: wait for autocomplete suggestions "
            "and click the suggestion representing the approved city, region, and country."
        )
    return f"""Submit exactly one reviewed job application at {url}.

Use only the field values in this JSON array; do not invent, rewrite, or omit required answers:
{json.dumps(fields, indent=2)}

Rules:
1. Navigate only within the application site and its CAPTCHA provider.
   The pipeline has already filled and deterministically verified every reviewed field; do not
   re-enter them or read the resume file. Dismiss any privacy banner, visually spot-check the
   visible form once, and proceed directly to the final submit button.
2. The pipeline has already uploaded the reviewed Resume/CV. Verify that its filename is visible,
   but do not type a path into the file control and do not replace the file.
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
    from browser_use.browser.events import (
        ClickElementEvent,
        NavigateToUrlEvent,
        ScrollToTextEvent,
        SelectDropdownOptionEvent,
        SendKeysEvent,
        TypeTextEvent,
        UploadFileEvent,
    )

    resume_paths = [
        action.value
        for action in plan.actions
        if action.source == "profile.documents.resume" and Path(action.value).is_file()
    ]
    target_url = application_url(url)
    browser_options = {
        "executable_path": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "user_data_dir": profile_dir,
        "profile_directory": "Default",
        "headless": False,
        "keep_alive": False,
    }
    if urlparse(target_url).hostname == "jobs.lever.co":
        browser_options["allowed_domains"] = allowed_domains(target_url)
    browser = Browser(
        **browser_options,
    )
    llm = ChatOllama(
        model=model,
        host=ollama_base_url.rstrip("/"),
        timeout=180.0,
        ollama_options={"temperature": 0, "num_ctx": 32768},
    )
    location_alias = _reviewed_location_alias(url, plan)

    async def set_reviewed_location(query: str, expected: str) -> None:
        state = await browser.get_browser_state_summary(include_screenshot=False)
        location = next(
            (
                node
                for node in state.dom_state.selector_map.values()
                if node.attributes.get("id") == "location-input"
                or node.attributes.get("name") == "location"
                or node.attributes.get("name") == "currentLocation"
                or node.attributes.get("role") == "combobox"
                or node.attributes.get("placeholder") == "Start typing..."
            ),
            None,
        )
        if location is None:
            raise RuntimeError("Application location input was not found")

        click = browser.event_bus.dispatch(ClickElementEvent(node=location))
        await click
        await click.event_result(raise_if_any=True, raise_if_none=False)

        # Use browser-use's native text event so typing targets this exact DOM node.
        # Sending character key events to the page can silently hit the wrong focus
        # target when Lever restores focus during its React re-render.
        type_location = browser.event_bus.dispatch(
            TypeTextEvent(node=location, text=query, clear=True)
        )
        await type_location
        await type_location.event_result(raise_if_any=True, raise_if_none=False)

        await asyncio.sleep(0.5)
        session = await browser.get_or_create_cdp_session()
        if urlparse(target_url).hostname == "jobs.lever.co":
            expected_json = json.dumps(expected)
            selected_json = json.dumps(json.dumps({"name": expected}))
            lever_result = await session.cdp_client.send_raw(
                "Runtime.evaluate",
                {
                    "expression": f"""
(() => {{
  const visible = document.querySelector('input[name="location"], #location-input');
  const selected = document.querySelector(
    '#selected-location, input[name="selectedLocation"], input[name="selected-location"]'
  );
  if (!visible || !selected) return null;
  visible.value = {expected_json};
  selected.value = {selected_json};
  document.querySelectorAll('.dropdown-container').forEach(element => {{
    element.style.display = 'none';
  }});
  document.querySelectorAll('.dropdown-results').forEach(element => {{
    element.replaceChildren();
  }});
  return {{visible: visible.value, selected: selected.value}};
}})()
""",
                    "returnByValue": True,
                },
                session_id=session.session_id,
            )
            lever_values = lever_result.get("result", {}).get("value") or {}
            if lever_values.get("visible") == expected and lever_values.get(
                "selected"
            ) == json.dumps({"name": expected}):
                return

        current_result = await session.cdp_client.send_raw(
            "Runtime.evaluate",
            {
                "expression": (
                    "document.querySelector('input[name=\"location\"], "
                    "#location-input, input[name=\"currentLocation\"]')?.value || ''"
                ),
                "returnByValue": True,
            },
            session_id=session.session_id,
        )
        current_value = current_result.get("result", {}).get("value") or None
        if current_value == expected:
            return

        expected_json = json.dumps(expected)
        set_result = await session.cdp_client.send_raw(
            "Runtime.evaluate",
            {
                "expression": f"""
(() => {{
  const element = document.querySelector(
    'input[name="location"], #location-input, input[name="currentLocation"]'
  );
  if (!element) return null;
  const setter = Object.getOwnPropertyDescriptor(
    HTMLInputElement.prototype, 'value'
  ).set;
  setter.call(element, {expected_json});
  element.dispatchEvent(new InputEvent('input', {{bubbles: true, inputType: 'insertText'}}));
  element.dispatchEvent(new Event('change', {{bubbles: true}}));
  element.blur();
  return element.value;
}})()
""",
                "returnByValue": True,
            },
            session_id=session.session_id,
        )
        set_value = set_result.get("result", {}).get("value") or None
        await asyncio.sleep(0.5)
        if set_value == expected:
            verify_result = await session.cdp_client.send_raw(
                "Runtime.evaluate",
                {
                    "expression": (
                        "document.querySelector('input[name=\"location\"], "
                        "#location-input, input[name=\"currentLocation\"]')?.value || ''"
                    ),
                    "returnByValue": True,
                },
                session_id=session.session_id,
            )
            if verify_result.get("result", {}).get("value") == expected:
                return

        await asyncio.sleep(1.0)
        query_json = json.dumps(query.casefold())
        choose_result = await session.cdp_client.send_raw(
            "Runtime.evaluate",
            {
                "expression": f"""
(() => {{
  const query = {query_json};
  const candidates = [...document.querySelectorAll(
    '[id^="location-"], [role="option"], [class*="option" i]'
  )]
    .filter(element => {{
      const text = (element.innerText || '').trim().toLowerCase();
      const rect = element.getBoundingClientRect();
      return text.includes(query) && text.includes('canada') && rect.width && rect.height;
    }})
    .sort((a, b) => a.innerText.trim().length - b.innerText.trim().length);
  if (!candidates.length) return false;
  candidates[0].click();
  return candidates[0].innerText.trim();
}})()
""",
                "returnByValue": True,
            },
            session_id=session.session_id,
        )
        chosen_by_dom = choose_result.get("result", {}).get("value")
        if chosen_by_dom:
            await asyncio.sleep(0.5)

        state = await browser.get_browser_state_summary(include_screenshot=False)
        suggestion = next(
            (
                node
                for node in state.dom_state.selector_map.values()
                if (
                    node.attributes.get("id", "").startswith("location-")
                    or node.attributes.get("role") == "option"
                )
                and expected in node.get_all_children_text()
            ),
            None,
        )
        if suggestion is None and not chosen_by_dom:
            # Some Lever boards render the menu outside browser-use's DOM map.
            # With an exact reviewed city query, select the first autocomplete
            # result by keyboard, then validate the retained city/country below.
            for key in ("ARROWDOWN", "ENTER"):
                choose_key = browser.event_bus.dispatch(SendKeysEvent(keys=key))
                await choose_key
                await choose_key.event_result(raise_if_any=True, raise_if_none=False)
            chosen_by_dom = True
            await asyncio.sleep(0.5)
        if suggestion is not None and not chosen_by_dom:
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
                or node.attributes.get("name") == "location"
                or node.attributes.get("name") == "currentLocation"
                or node.attributes.get("role") == "combobox"
                or node.attributes.get("placeholder") == "Start typing..."
            ),
            None,
        )
        selected_value = (
            selected.snapshot_node.input_value
            if selected is not None and selected.snapshot_node is not None
            else None
        )
        if not selected_value:
            session = await browser.get_or_create_cdp_session()
            value_result = await session.cdp_client.send_raw(
                "Runtime.evaluate",
                {
                    "expression": (
                        "document.querySelector('input[name=\"location\"], "
                        "#location-input, input[name=\"currentLocation\"]')?.value || ''"
                    ),
                    "returnByValue": True,
                },
                session_id=session.session_id,
            )
            selected_value = value_result.get("result", {}).get("value") or None
        selected_folded = (selected_value or "").casefold()
        location_matches = query.casefold() in selected_folded and (
            "canada" in selected_folded or "can" in selected_folded
        )
        if selected_value != expected and not location_matches:
            session = await browser.get_or_create_cdp_session()
            debug_result = await session.cdp_client.send_raw(
                "Runtime.evaluate",
                {
                    "expression": """
JSON.stringify({
  url: location.href,
  active: document.activeElement && {
    tag: document.activeElement.tagName,
    id: document.activeElement.id,
    name: document.activeElement.getAttribute('name'),
    value: document.activeElement.value
  },
  locations: [...document.querySelectorAll(
    '#location-input, input[name="location"], input[name="currentLocation"]'
  )].map(element => ({
    tag: element.tagName,
    id: element.id,
    name: element.getAttribute('name'),
    value: element.value,
    outerHTML: element.outerHTML.slice(0, 500)
  })),
  suggestions: [...document.querySelectorAll('[role="option"], [id^="location-"]')]
    .filter(element => element.getBoundingClientRect().width)
    .slice(0, 10)
    .map(element => ({id: element.id, role: element.getAttribute('role'), text: element.innerText}))
})
""",
                    "returnByValue": True,
                },
                session_id=session.session_id,
            )
            debug_value = debug_result.get("result", {}).get("value")
            raise RuntimeError(
                f"Location selection was not retained: {selected_value!r}; DOM={debug_value}"
            )

    async def upload_reviewed_resume(file_path: str) -> None:
        state = await browser.get_browser_state_summary(include_screenshot=False)
        file_input = next(
            (
                node
                for node in state.dom_state.selector_map.values()
                if node.tag_name.casefold() == "input"
                and node.attributes.get("type", "").casefold() == "file"
                and "resume" in node.attributes.get("id", "").casefold()
            ),
            None,
        )
        if file_input is None:
            file_input = next(
                (
                    node
                    for node in state.dom_state.selector_map.values()
                    if node.tag_name.casefold() == "input"
                    and node.attributes.get("type", "").casefold() == "file"
                ),
                None,
            )
        if file_input is None:
            raise RuntimeError("Resume file input was not found")
        upload = browser.event_bus.dispatch(UploadFileEvent(node=file_input, file_path=file_path))
        await upload
        await upload.event_result(raise_if_any=True, raise_if_none=False)
        await asyncio.sleep(1.0)

    def node_for_selector(state, selector: str, expected_value: str):
        if selector.startswith("#"):
            element_id = selector[1:].replace('\\"', '"').replace("\\\\", "\\")
            return next(
                (
                    node
                    for node in state.dom_state.selector_map.values()
                    if node.attributes.get("id") == element_id
                ),
                None,
            )
        match = re.search(r'\[(?:name(?:\$)?|placeholder)="(.*)"\]$', selector)
        if not match:
            return None
        attribute = "placeholder" if "[placeholder=" in selector else "name"
        expected = match.group(1).replace('\\"', '"').replace("\\\\", "\\")
        candidates = [
            node
            for node in state.dom_state.selector_map.values()
            if (
                node.attributes.get(attribute, "").endswith(expected)
                if "[name$=" in selector
                else node.attributes.get(attribute) == expected
            )
        ]
        if len(candidates) <= 1:
            return candidates[0] if candidates else None
        expected_folded = expected_value.casefold()
        return next(
            (
                node
                for node in candidates
                if node.attributes.get("value", "").casefold() == expected_folded
                or node.get_all_children_text().strip().casefold() == expected_folded
                or any(
                    ancestor.get_all_children_text().strip().casefold() == expected_folded
                    for ancestor in [node.parent, node.parent.parent if node.parent else None]
                    if ancestor is not None
                )
            ),
            candidates[0],
        )

    async def fill_reviewed_actions() -> None:
        for action in plan.actions:
            if action.source == "profile.documents.resume" or (
                location_alias
                and action.label.strip().casefold() in {"location", "current location"}
            ):
                continue
            if action.selector.startswith("input[name"):
                session = await browser.get_or_create_cdp_session()
                selector_json = json.dumps(action.selector)
                value_json = json.dumps(action.value.casefold())
                group_result = await session.cdp_client.send_raw(
                    "Runtime.evaluate",
                    {
                        "expression": f"""
(() => {{
  const selector = {selector_json};
  const expected = {value_json};
  const normalize = value => value.toLowerCase().replace(/[^a-z0-9$+]+/g, ' ').trim();
  const candidates = [...document.querySelectorAll(selector)].filter(
    element => element.type === 'checkbox' || element.type === 'radio'
  );
  const choice = candidates.find(element => {{
    const labels = [
      element.closest('label'),
      element.parentElement,
      element.parentElement?.parentElement
    ].filter(Boolean);
    return labels.some(label => normalize(label.innerText || '') === normalize(expected));
  }});
  if (!choice) return false;
  if (!choice.checked) choice.click();
  return choice.checked;
}})()
""",
                        "returnByValue": True,
                    },
                    session_id=session.session_id,
                )
                if group_result.get("result", {}).get("value") is True:
                    await asyncio.sleep(0.25)
                    continue
            if action.requires_review and action.value.casefold() in {"yes", "no"}:
                session = await browser.get_or_create_cdp_session()
                selector_json = json.dumps(action.selector)
                value_json = json.dumps(action.value.casefold())
                expression = f"""
(() => {{
  const selector = {selector_json};
  const value = {value_json};
  const element = selector.startsWith('#')
    ? document.getElementById(selector.slice(1))
    : document.querySelector(selector);
  if (!element) return false;
  const option = element.closest('.ashby-application-form-field-entry')?.querySelector(
    `button[data-option="${{value}}"]`
  );
  if (!option) return false;
  option.click();
  return option.getAttribute('aria-pressed') === 'true';
}})()
"""
                result = await session.cdp_client.send_raw(
                    "Runtime.evaluate",
                    {"expression": expression, "returnByValue": True},
                    session_id=session.session_id,
                )
                if result.get("result", {}).get("value") is True:
                    await asyncio.sleep(0.25)
                    continue
            if action.requires_review and action.selector.startswith("input[name"):
                session = await browser.get_or_create_cdp_session()
                selector_json = json.dumps(action.selector)
                value_json = json.dumps(action.value.casefold())
                expression = f"""
(() => {{
  const selector = {selector_json};
  const expected = {value_json};
  const normalize = value => value.toLowerCase().replace(/[^a-z0-9$+]+/g, ' ').trim();
  const candidate = [...document.querySelectorAll(selector)].find(element => {{
    const option = element.closest('.ashby-application-form-input-radio-group-option');
    return option && normalize(option.innerText || '') === normalize(expected);
  }});
  if (!candidate) return false;
  candidate.closest('.ashby-application-form-input-radio-group-option').click();
  return candidate.checked === true;
}})()
"""
                result = await session.cdp_client.send_raw(
                    "Runtime.evaluate",
                    {"expression": expression, "returnByValue": True},
                    session_id=session.session_id,
                )
                if result.get("result", {}).get("value") is True:
                    await asyncio.sleep(0.25)
                    continue
            state = await browser.get_browser_state_summary(include_screenshot=False)
            node = node_for_selector(state, action.selector, action.value)
            if node is None:
                scroll = browser.event_bus.dispatch(
                    ScrollToTextEvent(text=action.label, direction="down")
                )
                await scroll
                await scroll.event_result(raise_if_any=False, raise_if_none=False)
                await asyncio.sleep(0.5)
                state = await browser.get_browser_state_summary(include_screenshot=False)
                node = node_for_selector(state, action.selector, action.value)
            if node is None:
                session = await browser.get_or_create_cdp_session()
                selector_json = json.dumps(action.selector)
                value_json = json.dumps(action.value)
                fallback_result = await session.cdp_client.send_raw(
                    "Runtime.evaluate",
                    {
                        "expression": f"""
(() => {{
  const element = document.querySelector({selector_json});
  const reviewed = {value_json};
  if (!element) return false;
  if (element.tagName === 'SELECT') {{
    const option = [...element.options].find(item => item.text.trim() === reviewed);
    if (!option) return false;
    element.value = option.value;
  }} else if (element.type === 'checkbox' || element.type === 'radio') {{
    const candidates = [...document.querySelectorAll({selector_json})];
    const normalize = value => value.toLowerCase().replace(/[^a-z0-9$+]+/g, ' ').trim();
    const choice = candidates.find(candidate => {{
      const container = candidate.closest('label, .application-answer, .application-question');
      return container && normalize(container.innerText || '') === normalize(reviewed);
    }});
    if (!choice) return false;
    choice.click();
    return choice.checked;
  }} else {{
    const prototype = element.tagName === 'TEXTAREA'
      ? HTMLTextAreaElement.prototype
      : HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(prototype, 'value').set;
    setter.call(element, reviewed);
  }}
  element.dispatchEvent(new Event('input', {{bubbles: true}}));
  element.dispatchEvent(new Event('change', {{bubbles: true}}));
  return element.value === reviewed;
}})()
""",
                        "returnByValue": True,
                    },
                    session_id=session.session_id,
                )
                if fallback_result.get("result", {}).get("value") is True:
                    await asyncio.sleep(0.25)
                    continue
                raise RuntimeError(f"Reviewed field was not found: {action.label}")
            input_type = node.attributes.get("type", "").casefold()
            if input_type in {"checkbox", "radio"}:
                event = browser.event_bus.dispatch(ClickElementEvent(node=node))
            elif node.tag_name.casefold() == "select":
                event = browser.event_bus.dispatch(
                    SelectDropdownOptionEvent(node=node, text=action.value)
                )
            else:
                event = browser.event_bus.dispatch(
                    TypeTextEvent(node=node, text=action.value, clear=True)
                )
            await event
            await event.event_result(raise_if_any=True, raise_if_none=False)

    await browser.start()
    try:
        navigation = browser.event_bus.dispatch(NavigateToUrlEvent(url=target_url, new_tab=False))
        await navigation
        await navigation.event_result(raise_if_any=True, raise_if_none=False)
        await asyncio.sleep(1.0)
        if location_alias:
            await set_reviewed_location(*location_alias)
        if resume_paths:
            await upload_reviewed_resume(resume_paths[0])
        await fill_reviewed_actions()
    except Exception:
        await browser.kill()
        raise

    agent = Agent(
        task=build_browser_use_task(target_url, plan),
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
    history = await agent.run(max_steps=4)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    history.save_to_file(artifact_dir / "browser-use-history.json")
    final_result = history.final_result() or "APPLICATION_UNCONFIRMED: no final result"
    confirmed = "APPLICATION_SUBMITTED" in final_result.upper()
    status = JobStatus.SUBMISSION_CONFIRMED if confirmed else JobStatus.UNCERTAIN
    return status, final_result

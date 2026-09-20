from __future__ import annotations

import asyncio
import json
from pathlib import Path

from ..models import FormField
from .browser_use_executor import application_url

SCAN_EXPRESSION = r"""
JSON.stringify([...document.querySelectorAll('input,textarea,select')].map((el) => {
  const explicit = el.id ? document.querySelector(`label[for="${CSS.escape(el.id)}"]`) : null;
  const wrapping = el.closest('label');
  let ancestorLabel = null;
  for (let parent = el.parentElement, i = 0; parent && i < 5; parent = parent.parentElement, i++) {
    ancestorLabel = [...parent.children].find(
      child => child.tagName === 'LABEL' || child.classList?.contains('application-label')
    ) || null;
    if (ancestorLabel) break;
  }
  let sibling = el.previousElementSibling;
  while (sibling && !sibling.innerText?.trim()) sibling = sibling.previousElementSibling;
  const parentSibling = el.parentElement?.previousElementSibling;
  let container = el.parentElement;
  for (let i = 0; container && i < 4; i++, container = container.parentElement) {
    const text = container.innerText?.trim() || '';
    if (text && text.length < 1000 &&
        !['application-dropdown'].includes(container.className)) break;
  }
  return {
    tag: el.tagName.toLowerCase(), type: el.type || '', id: el.id || '', name: el.name || '',
    placeholder: el.placeholder || '', role: el.getAttribute('role') || '',
    required: el.required || el.getAttribute('aria-required') === 'true',
    label: (explicit?.innerText || wrapping?.innerText || ancestorLabel?.innerText ||
            sibling?.innerText ||
            parentSibling?.innerText || container?.innerText || '').trim(),
    options: el.tagName === 'SELECT' ? [...el.options].map(o => o.text.trim()).filter(Boolean) : []
  };
}))
"""


def _css_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _field_selector(item: dict[str, object]) -> str:
    element_id = str(item.get("id") or "")
    if element_id:
        return f'#{_css_string(element_id)}'
    tag = str(item["tag"])
    name = str(item.get("name") or "")
    if name:
        return f'{tag}[name="{_css_string(name)}"]'
    placeholder = str(item.get("placeholder") or "")
    if placeholder:
        return f'{tag}[placeholder="{_css_string(placeholder)}"]'
    role = str(item.get("role") or "")
    return f'{tag}[role="{_css_string(role)}"]'


def parse_scanned_fields(raw: str) -> list[FormField]:
    fields: list[FormField] = []
    for item in json.loads(raw):
        input_type = str(item.get("type") or "text").casefold()
        if input_type == "hidden" or not any(
            item.get(key) for key in ("id", "name", "placeholder", "role")
        ):
            continue
        tag = str(item["tag"]).casefold()
        field_type = "select" if tag == "select" else "file" if input_type == "file" else input_type
        fields.append(
            FormField(
                selector=_field_selector(item),
                label=str(item.get("label") or item.get("name") or "").strip(),
                field_type=field_type,
                required=bool(item.get("required")),
                options=list(item.get("options") or []),
            )
        )
    return fields


async def scan_with_browser_use(
    *, url: str, profile_dir: Path, headless: bool = False
) -> list[FormField]:
    from browser_use import Browser
    from browser_use.browser.events import NavigateToUrlEvent

    target_url = application_url(url)
    browser = Browser(
        executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        user_data_dir=profile_dir,
        profile_directory="Default",
        headless=headless,
        keep_alive=False,
    )
    await browser.start()
    try:
        navigation = browser.event_bus.dispatch(NavigateToUrlEvent(url=target_url, new_tab=False))
        await navigation
        await navigation.event_result(raise_if_any=True, raise_if_none=False)
        await asyncio.sleep(2)
        session = await browser.get_or_create_cdp_session()
        result = await session.cdp_client.send_raw(
            "Runtime.evaluate",
            {"expression": SCAN_EXPRESSION, "returnByValue": True},
            session_id=session.session_id,
        )
        fields = parse_scanned_fields(result["result"]["value"])
        if not fields:
            current_url = await browser.get_current_page_url()
            raise RuntimeError(f"No application fields found at {current_url}")
        return fields
    finally:
        await browser.kill()

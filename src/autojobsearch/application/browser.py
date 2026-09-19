from __future__ import annotations

from pathlib import Path

from playwright.async_api import BrowserContext, Page, Playwright, async_playwright

from ..models import (
    ApplicationEvidence,
    ApplicationPlan,
    BrowserExecutionResult,
    FillVerification,
    FormField,
    JobStatus,
    PlanApproval,
)
from .review import validate_approval

SCAN_SCRIPT = """
() => {
  const visible = (el) => {
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.display !== 'none'
      && style.visibility !== 'hidden'
      && rect.width > 0
      && rect.height > 0;
  };
  const labelFor = (el) => {
    if (el.id) {
      const explicit = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
      if (explicit) return explicit.innerText.trim();
    }
    const parent = el.closest('label');
    if (parent) return parent.innerText.trim();
    const applicationField = el.closest('.application-field');
    if (applicationField?.parentElement) {
      const applicationLabel = applicationField.parentElement.querySelector(
        '.application-label .text, .application-label'
      );
      if (applicationLabel) return applicationLabel.innerText.trim();
    }
    const labelledBy = el.getAttribute('aria-labelledby');
    if (labelledBy) {
      const ref = document.getElementById(labelledBy);
      if (ref) return ref.innerText.trim();
    }
    return el.getAttribute('aria-label') || el.placeholder || el.name || el.id || '';
  };
  const selectorFor = (el, index) => {
    if (el.id) return `#${CSS.escape(el.id)}`;
    if (el.name) return `${el.tagName.toLowerCase()}[name="${CSS.escape(el.name)}"]`;
    return `${el.tagName.toLowerCase()}:nth-of-type(${index + 1})`;
  };
  return [...document.querySelectorAll('input, select, textarea')]
    .filter((el) => visible(el)
      && el.getAttribute('aria-hidden') !== 'true'
      && !['hidden', 'submit', 'button', 'reset'].includes(el.type))
    .map((el, index) => ({
      selector: selectorFor(el, index),
      label: labelFor(el),
      field_type: el.getAttribute('role') === 'combobox'
        ? 'combobox'
        : (el.tagName === 'SELECT' ? 'select' : (el.type || el.tagName.toLowerCase())),
      required: Boolean(el.required || el.getAttribute('aria-required') === 'true'),
      options: el.tagName === 'SELECT' ? [...el.options].map((option) => option.text.trim()) : [],
    }));
}
"""


class ApplicationBrowser:
    def __init__(
        self, profile_dir: Path, *, headless: bool = False, channel: str = "chrome"
    ) -> None:
        self.profile_dir = profile_dir
        self.headless = headless
        self.channel = channel
        self._playwright: Playwright | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None

    async def __aenter__(self) -> ApplicationBrowser:
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self._playwright = await async_playwright().start()
        self.context = await self._playwright.chromium.launch_persistent_context(
            user_data_dir=str(self.profile_dir),
            channel=self.channel,
            headless=self.headless,
            viewport={"width": 1365, "height": 900},
        )
        self.page = self.context.pages[0] if self.context.pages else await self.context.new_page()
        return self

    async def __aexit__(self, *_args: object) -> None:
        if self.context:
            await self.context.close()
        if self._playwright:
            await self._playwright.stop()

    def _require_page(self) -> Page:
        if self.page is None:
            raise RuntimeError("ApplicationBrowser must be used as an async context manager")
        return self.page

    async def open(self, url: str) -> None:
        page = self._require_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=45_000)

    async def scan(self) -> list[FormField]:
        page = self._require_page()
        raw_fields = await page.evaluate(SCAN_SCRIPT)
        fields = [FormField.model_validate(field) for field in raw_fields]
        for field in fields:
            if field.field_type != "combobox" or not field.required:
                continue
            locator = page.locator(field.selector).first
            await locator.click()
            await page.wait_for_timeout(150)
            field.options = [
                text.strip()
                for text in await page.get_by_role("option").all_inner_texts()
                if text.strip()
            ]
            await locator.press("Escape")
        return fields

    async def fill(self, plan: ApplicationPlan) -> list[FillVerification]:
        page = self._require_page()
        results: list[FillVerification] = []
        for action in plan.actions:
            locator = page.locator(action.selector).first
            tag_name = await locator.evaluate("element => element.tagName.toLowerCase()")
            field_type = (await locator.get_attribute("type") or "").casefold()
            role = (await locator.get_attribute("role") or "").casefold()
            control_type = "combobox" if role == "combobox" else field_type
            observed_value: str | None = None
            if role == "combobox":
                await locator.click()
                option = page.get_by_role("option", name=action.value, exact=True)
                search_values = [action.value]
                if "," in action.value:
                    search_values.append(action.value.split(",", 1)[0])
                for search_value in search_values:
                    await locator.fill(search_value)
                    await page.wait_for_timeout(750)
                    if await option.count() == 1:
                        break
                if await option.count() != 1:
                    raise RuntimeError(
                        f"Combobox option is not uniquely selectable: {action.label}={action.value}"
                    )
                await option.click()
            elif tag_name == "select":
                await locator.select_option(label=action.value)
            elif field_type == "checkbox":
                desired = action.value.casefold() in {"yes", "true", "1", "checked"}
                if await locator.is_checked() != desired:
                    await locator.click()
            elif field_type == "radio":
                await locator.check()
            elif field_type == "file":
                file_path = Path(action.value).expanduser().resolve()
                if not file_path.is_file():
                    raise RuntimeError(f"Upload file does not exist: {file_path}")
                await locator.set_input_files(str(file_path))
                observed_value = file_path.name
            else:
                await locator.fill(action.value)

            actual = observed_value or await self._value(locator, tag_name, control_type)
            results.append(
                FillVerification(
                    selector=action.selector,
                    expected=action.value,
                    actual=actual,
                    matched=self._matches(action.value, actual, control_type),
                )
            )
        return results

    async def submit(
        self,
        plan: ApplicationPlan,
        approval: PlanApproval | None,
        verified: list[FillVerification],
    ) -> BrowserExecutionResult:
        validate_approval(plan, approval)
        if not verified or not all(item.matched for item in verified):
            return BrowserExecutionResult(verified=verified, status=JobStatus.MANUAL_ACTION)

        page = self._require_page()
        submit = page.locator(
            'button[type="submit"], input[type="submit"], button:has-text("Submit application")'
        ).first
        if await submit.count() == 0:
            return BrowserExecutionResult(verified=verified, status=JobStatus.MANUAL_ACTION)
        await submit.click()
        await page.wait_for_timeout(1_000)
        text = (await page.locator("body").inner_text())[:5_000]
        evidence = ApplicationEvidence(
            confirmation_url=page.url,
            confirmation_text=text,
        )
        status = JobStatus.SUBMISSION_CONFIRMED if evidence.is_positive else JobStatus.UNCERTAIN
        return BrowserExecutionResult(
            verified=verified,
            submitted=evidence.is_positive,
            evidence=evidence,
            status=status,
        )

    @staticmethod
    async def _value(locator, tag_name: str, field_type: str) -> str:
        if field_type in {"checkbox", "radio"}:
            return "checked" if await locator.is_checked() else "unchecked"
        if field_type == "combobox":
            await locator.click()
            selected_option = locator.page.locator(
                '[role="option"][class*="--is-selected"]:visible'
            )
            if await selected_option.count() == 1:
                value = (await selected_option.inner_text()).strip()
                await locator.press("Escape")
                return value
            await locator.press("Escape")
            shell = locator.locator("xpath=ancestor::*[contains(@class, 'select-shell')][1]")
            selected = shell.locator("[class*='single-value']")
            if await selected.count():
                return (await selected.first.inner_text()).strip()
            return await locator.input_value()
        if tag_name == "select":
            return await locator.locator("option:checked").inner_text()
        return await locator.input_value()

    @staticmethod
    def _matches(expected: str, actual: str, field_type: str) -> bool:
        if field_type == "checkbox":
            desired = expected.casefold() in {"yes", "true", "1", "checked"}
            return (actual == "checked") is desired
        if field_type == "tel":
            expected_digits = "".join(character for character in expected if character.isdigit())
            actual_digits = "".join(character for character in actual if character.isdigit())
            return expected_digits == actual_digits
        if field_type == "file":
            actual_name = actual.replace("\\", "/").rsplit("/", 1)[-1]
            return Path(expected).expanduser().name.casefold() == actual_name.casefold()
        return expected.strip().casefold() == actual.strip().casefold()

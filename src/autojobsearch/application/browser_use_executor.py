from __future__ import annotations

import json
from pathlib import Path

from ..models import ApplicationPlan, JobStatus, PlanApproval
from .review import validate_approval


def build_browser_use_task(url: str, plan: ApplicationPlan) -> str:
    fields = [{"label": action.label.strip(), "value": action.value} for action in plan.actions]
    return f"""Submit exactly one reviewed job application at {url}.

Use only the field values in this JSON array; do not invent, rewrite, or omit required answers:
{json.dumps(fields, indent=2)}

Rules:
1. Navigate only within the application site and its CAPTCHA provider.
2. Upload the file path supplied for the Resume/CV field.
3. Leave optional fields absent from the JSON blank, including pronouns and demographics.
4. Verify every filled value before submitting.
5. If an hCaptcha or other visual CAPTCHA appears, complete it through the visible browser UI.
6. Click the final submit button once.
7. Report APPLICATION_SUBMITTED only after the page visibly confirms receipt with text such as
   'application submitted', 'thank you for applying', or 'application received'.
8. If no positive receipt is visible, report APPLICATION_UNCONFIRMED and explain the blocker.
9. Stop immediately after the first submission attempt. Never apply to another job.
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

    from browser_use import Agent, Browser, ChatOpenAI

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
    llm = ChatOpenAI(
        model=model,
        base_url=f"{ollama_base_url.rstrip('/')}/v1",
        api_key="ollama",
        temperature=0.0,
        frequency_penalty=None,
        reasoning_effort=None,
        max_completion_tokens=4096,
    )
    agent = Agent(
        task=build_browser_use_task(url, plan),
        llm=llm,
        browser=browser,
        available_file_paths=resume_paths,
        use_vision=True,
        use_judge=False,
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

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit, urlunsplit

import typer
from rich.console import Console
from rich.table import Table

from .application import ReviewRequiredError, build_fill_plan, plan_digest, validate_approval
from .application.browser import ApplicationBrowser
from .application.browser_use_executor import submit_with_browser_use
from .application.browser_use_scanner import scan_with_browser_use
from .application.indeed_executor import run_indeed_apply
from .application.linkedin import run_linkedin_easy_apply
from .decision_bench import load_decision_cases, run_decision_benchmark
from .decisions import DecisionEngine
from .discovery import AshbyDiscovery, GreenhouseDiscovery, LeverDiscovery
from .job_boards import JobBoard, JobBoardRunConfig
from .llm import OllamaClient, OllamaError
from .models import ApplicationPlan, BrowserExecutionResult, JobStatus, PlanApproval
from .pipeline import Pipeline
from .profile import load_profile
from .settings import Settings
from .storage import Store

app = typer.Typer(no_args_is_help=True, help="Local-first job search and application pipeline")
console = Console()


def _runtime() -> tuple[Settings, Store]:
    settings = Settings()
    store = Store(settings.database_path)
    store.initialize()
    return settings, store


def _canonical_job_url(url: str) -> str:
    parsed = urlsplit(url)
    path = parsed.path.rstrip("/")
    if path.endswith("/apply"):
        path = path.removesuffix("/apply")
    if path.endswith("/application"):
        path = path.removesuffix("/application")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, "", ""))


def _require_matching_job_url(store: Store, job_id: int, url: str) -> None:
    try:
        stored_url = store.job_url(job_id)
    except KeyError as exc:
        console.print(f"Application blocked: {exc}")
        raise typer.Exit(1) from exc
    if _canonical_job_url(stored_url) != _canonical_job_url(url):
        console.print(
            f"Application blocked: job {job_id} is stored as {stored_url}, not {url}"
        )
        raise typer.Exit(1)


@app.command()
def init(
    force: bool = typer.Option(False, help="Replace an existing local example profile"),
) -> None:
    """Initialize private runtime state outside the Git checkout."""
    settings, _ = _runtime()
    settings.expanded_home.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parents[2] / "config" / "profile.example.json"
    if settings.profile_path.exists() and not force:
        console.print(f"Profile already exists: {settings.profile_path}")
        raise typer.Exit(1)
    shutil.copyfile(source, settings.profile_path)
    console.print(f"Created local profile: {settings.profile_path}")


@app.command()
def doctor() -> None:
    """Check local configuration and Ollama readiness."""
    settings, _ = _runtime()
    table = Table("Check", "Result")
    table.add_row("Runtime home", str(settings.expanded_home))
    table.add_row("Profile", "ready" if settings.profile_path.exists() else "missing; run init")
    table.add_row("Review before submit", str(settings.review_before_submit))
    try:
        models = OllamaClient(settings.ollama_base_url, settings.ollama_model).list_models()
        table.add_row("Ollama", "reachable")
        table.add_row(
            "Model",
            "ready"
            if any(m.split(":latest")[0] == settings.ollama_model for m in models)
            else "missing",
        )
    except OllamaError as exc:
        table.add_row("Ollama", str(exc))
    console.print(table)


@app.command("benchmark-decisions")
def benchmark_decisions(
    trace_path: Path,
    model: str | None = typer.Option(None, help="Local Ollama model to benchmark"),
) -> None:
    """Run saved closed-decision traces against a local Ollama model."""
    settings, _ = _runtime()
    cases = load_decision_cases(trace_path)
    model_name = model or settings.ollama_model
    engine = DecisionEngine(
        OllamaClient(settings.ollama_base_url, model_name),
        provider_name="ollama",
        model=model_name,
    )
    report = run_decision_benchmark(cases, engine)

    summary = Table("Metric", "Value")
    summary.add_row("Cases", str(report.total))
    summary.add_row("Passed", str(report.passed))
    summary.add_row("Failed", str(report.failed))
    summary.add_row("Average latency", f"{report.average_elapsed_ms} ms")
    console.print(summary)

    failures = [result for result in report.results if not result.passed]
    if failures:
        failed = Table("Case", "Expected", "Actual", "Route", "Confidence")
        for result in failures:
            failed.add_row(
                result.name,
                result.expected.model_dump_json(exclude_none=True),
                result.actual.model_dump_json(exclude_none=True),
                result.route.value,
                f"{result.confidence:.2f}",
            )
        console.print(failed)


@app.command("discover-greenhouse")
def discover_greenhouse(
    board_token: str,
    use_llm: bool = typer.Option(False, help="Use Ollama for accepted jobs"),
    json_output: bool = typer.Option(False, "--json", help="Print every result as JSON"),
) -> None:
    """Discover and score all public jobs on a Greenhouse board."""
    _score_discovered(GreenhouseDiscovery().discover(board_token), use_llm, json_output)


def _score_discovered(jobs: Iterable, use_llm: bool, json_output: bool = False) -> None:
    settings, store = _runtime()
    if not settings.profile_path.exists():
        console.print("Missing profile. Run: autojobsearch init")
        raise typer.Exit(1)
    profile = load_profile(settings.profile_path)
    llm = OllamaClient(settings.ollama_base_url, settings.ollama_model) if use_llm else None
    pipeline = Pipeline(store, profile, llm)
    results = []
    accepted = 0
    shortlisted = []
    for job in jobs:
        job_id, rules, assessment = pipeline.ingest_and_score(job)
        if rules.accepted:
            accepted += 1
        effective_score = assessment.score if assessment else rules.score
        if rules.accepted and effective_score >= profile.preferences.minimum_score:
            shortlisted.append((effective_score, job.title, job.company))
        results.append(
            {
                "id": job_id,
                "title": job.title,
                "company": job.company,
                "rules": rules.model_dump(),
                "assessment": assessment.model_dump() if assessment else None,
            }
        )
    if json_output:
        console.print_json(json.dumps(results))
        return

    summary = Table("Metric", "Count")
    summary.add_row("Discovered", str(len(results)))
    summary.add_row("Passed hard filters", str(accepted))
    summary.add_row("Shortlisted", str(len(shortlisted)))
    summary.add_row("Rejected", str(len(results) - len(shortlisted)))
    console.print(summary)

    if shortlisted:
        top = Table("Score", "Title", "Company", title="Top shortlisted roles")
        for score, title, company in sorted(shortlisted, reverse=True)[:20]:
            top.add_row(str(score), title, company)
        console.print(top)


async def _prepare_application(
    *, settings: Settings, job_id: int, url: str, headless: bool, fill: bool
) -> None:
    profile = load_profile(settings.profile_path)
    async with ApplicationBrowser(
        settings.expanded_home / "browser-profile", headless=headless
    ) as browser:
        await browser.open(url)
        fields = await browser.scan()
        plan = build_fill_plan(job_id, fields, profile)
        verified = await browser.fill(plan) if fill else []

    artifact_dir = settings.expanded_home / "applications" / str(job_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "plan.json").write_text(plan.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (artifact_dir / "verification.json").write_text(
        json.dumps([item.model_dump() for item in verified], indent=2) + "\n", encoding="utf-8"
    )
    digest_path = artifact_dir / "approval-digest.txt"
    if plan.ready_for_review:
        digest_path.write_text(plan_digest(plan) + "\n", encoding="utf-8")
    else:
        digest_path.unlink(missing_ok=True)

    summary = Table("Application preparation", "Result")
    summary.add_row("Scanned fields", str(len(fields)))
    summary.add_row("Planned actions", str(len(plan.actions)))
    summary.add_row("Verified actions", str(sum(item.matched for item in verified)))
    summary.add_row("Unresolved required", str(sum(field.required for field in plan.unresolved)))
    summary.add_row("Ready for review", str(plan.ready_for_review))
    digest = plan_digest(plan) if plan.ready_for_review else "blocked until required fields resolve"
    summary.add_row("Plan digest", digest)
    summary.add_row("Private artifacts", str(artifact_dir))
    console.print(summary)

    required = [field for field in plan.unresolved if field.required]
    if required:
        unresolved = Table("Type", "Required field", title="Manual answers needed")
        for field in required:
            unresolved.add_row(field.field_type, field.label or field.selector)
        console.print(unresolved)


@app.command("prepare-application")
def prepare_application_command(
    job_id: int,
    url: str,
    headless: bool = typer.Option(False, help="Run Chrome without showing a window"),
    fill: bool = typer.Option(False, help="Dry-fill planned fields and verify them; never submit"),
) -> None:
    """Scan an application, create a private plan, and optionally verify a dry fill."""
    settings, store = _runtime()
    _require_matching_job_url(store, job_id, url)
    if not settings.profile_path.exists():
        console.print("Missing profile. Run: autojobsearch init")
        raise typer.Exit(1)
    asyncio.run(
        _prepare_application(
            settings=settings, job_id=job_id, url=url, headless=headless, fill=fill
        )
    )


@app.command("browser-use-prepare")
def browser_use_prepare_command(
    job_id: int,
    url: str,
    profile_dir: Annotated[
        Path | None, typer.Option(help="Persistent Chrome profile for browser-use")
    ] = None,
    headless: bool = typer.Option(False, help="Run Chrome without showing a window"),
) -> None:
    """Scan an application and build its review plan using browser-use, never Playwright."""
    if profile_dir is None:
        console.print("Preparation blocked: pass --profile-dir for browser-use")
        raise typer.Exit(1)
    settings, store = _runtime()
    _require_matching_job_url(store, job_id, url)
    profile = load_profile(settings.profile_path)
    fields = asyncio.run(
        scan_with_browser_use(
            url=url,
            profile_dir=profile_dir.expanduser().resolve(),
            headless=headless,
        )
    )
    plan = build_fill_plan(job_id, fields, profile)
    artifact_dir = settings.expanded_home / "applications" / str(job_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "plan.json").write_text(
        plan.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    digest_path = artifact_dir / "approval-digest.txt"
    if plan.ready_for_review:
        digest_path.write_text(plan_digest(plan) + "\n", encoding="utf-8")
    else:
        digest_path.unlink(missing_ok=True)

    summary = Table("Browser-use preparation", "Result")
    summary.add_row("Scanned fields", str(len(fields)))
    summary.add_row("Planned actions", str(len(plan.actions)))
    summary.add_row("Unresolved required", str(sum(field.required for field in plan.unresolved)))
    summary.add_row("Ready for review", str(plan.ready_for_review))
    summary.add_row("Private artifacts", str(artifact_dir))
    console.print(summary)
    required = [field for field in plan.unresolved if field.required]
    if required:
        unresolved = Table("Type", "Required field", title="Manual answers needed")
        for field in required:
            unresolved.add_row(field.field_type, field.label or field.selector)
        console.print(unresolved)


async def _submit_application(
    *, settings: Settings, job_id: int, url: str, approval_digest: str, headless: bool
) -> BrowserExecutionResult:
    profile = load_profile(settings.profile_path)
    async with ApplicationBrowser(
        settings.expanded_home / "browser-profile", headless=headless
    ) as browser:
        await browser.open(url)
        fields = await browser.scan()
        plan = build_fill_plan(job_id, fields, profile)
        approval = PlanApproval(job_id=job_id, plan_digest=approval_digest)
        validate_approval(plan, approval)
        verified = await browser.fill(plan)
        result = await browser.submit(plan, approval, verified)

    artifact_dir = settings.expanded_home / "applications" / str(job_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "submission-result.json").write_text(
        result.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return result


@app.command("submit-application")
def submit_application_command(
    job_id: int,
    url: str,
    approval_digest: str = typer.Option(..., help="Digest from the reviewed current plan"),
    confirm_submit: bool = typer.Option(
        False, "--confirm-submit", help="Required explicit authorization to click submit"
    ),
    headless: bool = typer.Option(False, help="Run Chrome without showing a window"),
) -> None:
    """Rebuild, verify, and submit exactly one explicitly approved application plan."""
    if not confirm_submit:
        console.print("Submission blocked: pass --confirm-submit after reviewing the plan")
        raise typer.Exit(1)
    settings, store = _runtime()
    _require_matching_job_url(store, job_id, url)
    if not settings.profile_path.exists():
        console.print("Missing profile. Run: autojobsearch init")
        raise typer.Exit(1)
    try:
        result = asyncio.run(
            _submit_application(
                settings=settings,
                job_id=job_id,
                url=url,
                approval_digest=approval_digest,
                headless=headless,
            )
        )
    except ReviewRequiredError as exc:
        console.print(f"Submission blocked: {exc}")
        raise typer.Exit(1) from exc
    console.print(f"Submission result: {result.status.value}")
    if not result.submitted:
        console.print("No positive confirmation was observed; the application is not confirmed")


@app.command("browser-use-submit")
def browser_use_submit_command(
    job_id: int,
    url: str,
    approval_digest: str = typer.Option(..., help="Digest from the reviewed current plan"),
    confirm_submit: bool = typer.Option(
        False, "--confirm-submit", help="Required explicit authorization to click submit"
    ),
    model: str = typer.Option("qwen3.5:9b", help="Local Ollama browser model"),
    profile_dir: Annotated[
        Path | None, typer.Option(help="Persistent Chrome profile for browser-use")
    ] = None,
) -> None:
    """Submit one digest-locked plan through a browser-use agent."""
    if not confirm_submit:
        console.print("Submission blocked: pass --confirm-submit after reviewing the plan")
        raise typer.Exit(1)
    if profile_dir is None:
        console.print("Submission blocked: pass --profile-dir for browser-use")
        raise typer.Exit(1)
    settings, store = _runtime()
    _require_matching_job_url(store, job_id, url)
    artifact_dir = settings.expanded_home / "applications" / str(job_id)
    plan_path = artifact_dir / "plan.json"
    if not plan_path.exists():
        console.print("Submission blocked: prepare and review the application plan first")
        raise typer.Exit(1)
    plan = ApplicationPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    approval = PlanApproval(job_id=job_id, plan_digest=approval_digest)
    try:
        status, result = asyncio.run(
            submit_with_browser_use(
                url=url,
                plan=plan,
                approval=approval,
                model=model,
                ollama_base_url=settings.ollama_base_url,
                profile_dir=profile_dir.expanduser().resolve(),
                artifact_dir=artifact_dir,
            )
        )
    except ReviewRequiredError as exc:
        console.print(f"Submission blocked: {exc}")
        raise typer.Exit(1) from exc
    store.transition(
        job_id,
        status,
        {
            "executor": "browser-use",
            "model": model,
            "history": str(artifact_dir / "browser-use-history.json"),
        },
    )
    console.print(f"Submission result: {status.value}")
    console.print(result)


@app.command("indeed-apply")
def indeed_apply_command(
    job_id: int,
    url: str,
    resume_path: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, readable=True)
    ],
    profile_dir: Annotated[
        Path,
        typer.Option(
            exists=True, file_okay=False, readable=True, help="Real Chrome user-data directory"
        ),
    ],
    profile_directory: str = typer.Option("Default", help="Chrome profile directory name"),
    confirm_submit: bool = typer.Option(
        False, "--confirm-submit", help="Required authorization for an Indeed submission"
    ),
    model: str | None = typer.Option(None, help="Local Ollama browser model"),
    max_listings: int = typer.Option(5, min=1, max=25),
) -> None:
    """Search/apply through Indeed only, with local inference and strict stop states."""
    if not confirm_submit:
        console.print("Submission blocked: pass --confirm-submit")
        raise typer.Exit(1)
    settings, store = _runtime()
    _require_matching_job_url(store, job_id, url)
    profile = load_profile(settings.profile_path)
    config = JobBoardRunConfig(
        board=JobBoard.INDEED,
        search_url=url,
        resume_path=resume_path.expanduser().resolve(),
        max_applications=1,
        max_listings_to_check=max_listings,
    )
    model_name = model or settings.ollama_model
    artifact_dir = settings.expanded_home / "applications" / str(job_id)
    store.transition(job_id, JobStatus.FILLING, {"executor": "browser-use-indeed"})
    try:
        result = asyncio.run(
            run_indeed_apply(
                config=config,
                profile=profile,
                model=model_name,
                ollama_base_url=settings.ollama_base_url,
                profile_dir=profile_dir.expanduser().resolve(),
                profile_directory=profile_directory,
                artifact_dir=artifact_dir,
            )
        )
    except Exception as exc:
        store.transition(
            job_id,
            JobStatus.FAILED,
            {"executor": "browser-use-indeed", "error": type(exc).__name__, "detail": str(exc)},
        )
        console.print(f"Indeed run failed: {exc}")
        raise typer.Exit(1) from exc
    store.transition(
        job_id,
        result.status,
        {
            "executor": "browser-use-indeed",
            "model": model_name,
            "outcome": result.outcome.value,
            "history": str(result.history_path),
            "detail": result.detail,
        },
    )
    console.print(f"Indeed result: {result.outcome.value} ({result.status.value})")


@app.command("linkedin-easy-apply")
def linkedin_easy_apply_command(
    job_id: int,
    confirm_submit: bool = typer.Option(
        False, "--confirm-submit", help="Required authorization to submit this application"
    ),
    model: str | None = typer.Option(None, help="Local Ollama browser model"),
    profile_dir: Annotated[
        Path | None, typer.Option(help="Chrome user-data directory for the logged-in profile")
    ] = None,
    profile_name: str = typer.Option("Default", help="Chrome profile directory name"),
) -> None:
    """Apply to one stored LinkedIn listing through its inline Easy Apply modal."""
    if not confirm_submit:
        console.print("Submission blocked: pass --confirm-submit for this LinkedIn application")
        raise typer.Exit(1)
    if profile_dir is None:
        console.print("Submission blocked: pass your real Chrome --profile-dir")
        raise typer.Exit(1)

    settings, store = _runtime()
    if not settings.profile_path.exists():
        console.print("Missing profile. Run: autojobsearch init")
        raise typer.Exit(1)
    profile = load_profile(settings.profile_path)
    resume_path = Path(profile.documents.resume).expanduser()
    url = store.job_url(job_id)
    artifact_dir = settings.expanded_home / "applications" / str(job_id)
    model_name = model or settings.ollama_model

    store.transition(
        job_id,
        JobStatus.FILLING,
        {"executor": "browser-use-linkedin", "model": model_name, "url": url},
    )
    try:
        result = asyncio.run(
            run_linkedin_easy_apply(
                url=url,
                profile=profile,
                resume_path=resume_path,
                model=model_name,
                ollama_base_url=settings.ollama_base_url,
                profile_dir=profile_dir.expanduser().resolve(),
                profile_name=profile_name,
                artifact_dir=artifact_dir,
            )
        )
    except Exception as exc:
        store.transition(
            job_id,
            JobStatus.FAILED,
            {"executor": "browser-use-linkedin", "error": str(exc)},
        )
        console.print(f"LinkedIn run failed safely: {exc}")
        raise typer.Exit(1) from exc

    store.transition(
        job_id,
        result.status,
        {
            "executor": "browser-use-linkedin",
            "model": model_name,
            "outcome": result.outcome.value,
            "history": str(result.history_path),
        },
    )
    console.print(f"LinkedIn result: {result.outcome.value} ({result.status.value})")
    console.print(result.detail)


@app.command("discover-lever")
def discover_lever(
    site: str,
    use_llm: bool = typer.Option(False, help="Use Ollama for accepted jobs"),
    json_output: bool = typer.Option(False, "--json", help="Print every result as JSON"),
) -> None:
    """Discover and score all public jobs on a Lever site."""
    _score_discovered(LeverDiscovery().discover(site), use_llm, json_output)


@app.command("discover-ashby")
def discover_ashby(
    organization: str,
    use_llm: bool = typer.Option(False, help="Use Ollama for accepted jobs"),
    json_output: bool = typer.Option(False, "--json", help="Print every result as JSON"),
) -> None:
    """Discover and score all public jobs on an Ashby board."""
    _score_discovered(AshbyDiscovery().discover(organization), use_llm, json_output)


if __name__ == "__main__":
    app()

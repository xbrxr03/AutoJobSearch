from __future__ import annotations

import json
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .discovery import AshbyDiscovery, GreenhouseDiscovery, LeverDiscovery
from .llm import OllamaClient, OllamaError
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


@app.command("discover-greenhouse")
def discover_greenhouse(
    board_token: str,
    use_llm: bool = typer.Option(True, help="Use Ollama for accepted jobs"),
) -> None:
    """Discover and score all public jobs on a Greenhouse board."""
    settings, store = _runtime()
    if not settings.profile_path.exists():
        console.print("Missing profile. Run: autojobsearch init")
        raise typer.Exit(1)
    profile = load_profile(settings.profile_path)
    llm = OllamaClient(settings.ollama_base_url, settings.ollama_model) if use_llm else None
    pipeline = Pipeline(store, profile, llm)
    results = []
    for job in GreenhouseDiscovery().discover(board_token):
        job_id, rules, assessment = pipeline.ingest_and_score(job)
        results.append(
            {
                "id": job_id,
                "title": job.title,
                "company": job.company,
                "rules": rules.model_dump(),
                "assessment": assessment.model_dump() if assessment else None,
            }
        )
    console.print_json(json.dumps(results))


def _score_discovered(jobs, use_llm: bool) -> None:
    settings, store = _runtime()
    if not settings.profile_path.exists():
        console.print("Missing profile. Run: autojobsearch init")
        raise typer.Exit(1)
    profile = load_profile(settings.profile_path)
    llm = OllamaClient(settings.ollama_base_url, settings.ollama_model) if use_llm else None
    pipeline = Pipeline(store, profile, llm)
    results = []
    for job in jobs:
        job_id, rules, assessment = pipeline.ingest_and_score(job)
        results.append(
            {
                "id": job_id,
                "title": job.title,
                "company": job.company,
                "rules": rules.model_dump(),
                "assessment": assessment.model_dump() if assessment else None,
            }
        )
    console.print_json(json.dumps(results))


@app.command("discover-lever")
def discover_lever(
    site: str,
    use_llm: bool = typer.Option(True, help="Use Ollama for accepted jobs"),
) -> None:
    """Discover and score all public jobs on a Lever site."""
    _score_discovered(LeverDiscovery().discover(site), use_llm)


@app.command("discover-ashby")
def discover_ashby(
    organization: str,
    use_llm: bool = typer.Option(True, help="Use Ollama for accepted jobs"),
) -> None:
    """Discover and score all public jobs on an Ashby board."""
    _score_discovered(AshbyDiscovery().discover(organization), use_llm)


if __name__ == "__main__":
    app()

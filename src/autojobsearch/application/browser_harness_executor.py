from __future__ import annotations

import asyncio
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class BrowserHarnessError(RuntimeError):
    """Raised when the browser-use daemon subprocess does not return a result."""


@dataclass(frozen=True)
class BrowserHarnessAgentResult:
    final_result: str
    history_path: Path
    result_path: Path
    stdout: str
    stderr: str


class BrowserHarnessRunner(Protocol):
    def __call__(
        self,
        command: Sequence[str],
        *,
        input: str,
        cwd: Path,
        env: Mapping[str, str],
        timeout: int,
    ) -> subprocess.CompletedProcess[str]: ...


def _default_browser_harness_home(artifact_dir: Path) -> Path:
    if artifact_dir.parent.name == "applications":
        return artifact_dir.parent.parent / "browser-harness"
    return artifact_dir / "browser-harness"


def _subprocess_run(
    command: Sequence[str],
    *,
    input: str,
    cwd: Path,
    env: Mapping[str, str],
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        input=input,
        cwd=cwd,
        env=dict(env),
        timeout=timeout,
        text=True,
        capture_output=True,
        check=False,
    )


def _harness_script(config_path: Path) -> str:
    return f"""
from __future__ import annotations

import asyncio
import json
import traceback
from pathlib import Path

from browser_harness.daemon import get_ws_url
from browser_use import Agent, Browser, ChatOllama

CONFIG_PATH = Path({str(config_path)!r})


async def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    result_path = Path(config["result_path"])
    history_path = Path(config["history_path"])
    browser = None
    try:
        browser = Browser(
            cdp_url=get_ws_url(),
            keep_alive=True,
            allowed_domains=config["allowed_domains"],
        )
        llm = ChatOllama(
            model=config["model"],
            host=config["ollama_base_url"].rstrip("/"),
            timeout=180.0,
            ollama_options={{"temperature": 0, "num_ctx": 32768}},
        )
        agent = Agent(
            task=config["task"],
            llm=llm,
            browser=browser,
            available_file_paths=config["available_file_paths"],
            use_vision=True,
            use_judge=False,
            use_thinking=False,
            enable_planning=False,
            llm_timeout=180,
            step_timeout=240,
            llm_screenshot_size=(1024, 768),
            vision_detail_level="low",
            max_history_items=config["max_history_items"],
            directly_open_url=True,
            extend_system_message=config["extend_system_message"],
            max_actions_per_step=3,
            max_failures=4,
        )
        history = await agent.run(max_steps=config["max_steps"])
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history.save_to_file(history_path)
        final_result = history.final_result() or config["empty_result"]
        result_path.write_text(
            json.dumps(
                {{
                    "ok": True,
                    "final_result": final_result,
                    "history_path": str(history_path),
                }},
                indent=2,
            )
            + "\\n",
            encoding="utf-8",
        )
        print("AUTOJOBSEARCH_RESULT_JSON=" + json.dumps({{"result_path": str(result_path)}}))
        return 0
    except Exception as exc:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {{
                    "ok": False,
                    "error": type(exc).__name__,
                    "detail": str(exc),
                    "traceback": traceback.format_exc(),
                }},
                indent=2,
            )
            + "\\n",
            encoding="utf-8",
        )
        print("AUTOJOBSEARCH_RESULT_JSON=" + json.dumps({{"result_path": str(result_path)}}))
        return 1
    finally:
        if browser is not None:
            close = getattr(browser, "close", None)
            if close is not None:
                await close()


raise SystemExit(asyncio.run(main()))
""".lstrip()


async def run_browser_harness_agent(
    *,
    task: str,
    model: str,
    ollama_base_url: str,
    allowed_domains: Sequence[str],
    available_file_paths: Sequence[Path],
    artifact_dir: Path,
    history_filename: str,
    result_filename: str,
    extend_system_message: str,
    max_steps: int,
    max_history_items: int,
    empty_result: str,
    bh_home: Path | None = None,
    cwd: Path | None = None,
    command: Sequence[str] = ("uv", "run", "browser-use"),
    timeout: int = 3600,
    subprocess_runner: BrowserHarnessRunner | None = None,
) -> BrowserHarnessAgentResult:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_path = artifact_dir / result_filename
    history_path = artifact_dir / history_filename
    config_path = artifact_dir / "browser-harness-config.json"
    config = {
        "task": task,
        "model": model,
        "ollama_base_url": ollama_base_url,
        "allowed_domains": list(allowed_domains),
        "available_file_paths": [str(path) for path in available_file_paths],
        "history_path": str(history_path),
        "result_path": str(result_path),
        "extend_system_message": extend_system_message,
        "max_steps": max_steps,
        "max_history_items": max_history_items,
        "empty_result": empty_result,
    }
    result_path.unlink(missing_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    env = os.environ.copy()
    default_bh_home = bh_home or _default_browser_harness_home(artifact_dir)
    env["BH_HOME"] = str(default_bh_home.resolve())
    env["BH_TAB_MARKER"] = "0"
    env.setdefault("BROWSER_USE_SETUP_LOGGING", "false")

    runner = subprocess_runner or _subprocess_run
    process = await asyncio.to_thread(
        runner,
        command,
        input=_harness_script(config_path),
        cwd=(cwd or Path.cwd()),
        env=env,
        timeout=timeout,
    )
    stdout = process.stdout or ""
    stderr = process.stderr or ""
    if not result_path.exists():
        raise BrowserHarnessError(
            "browser-use harness did not write a structured result"
            f" (exit {process.returncode}). stderr: {stderr.strip()}"
        )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if process.returncode != 0 or not payload.get("ok"):
        detail = payload.get("detail") or stderr.strip() or stdout.strip()
        raise BrowserHarnessError(f"browser-use harness failed: {detail}")
    return BrowserHarnessAgentResult(
        final_result=str(payload.get("final_result") or empty_result),
        history_path=Path(payload.get("history_path") or history_path),
        result_path=result_path,
        stdout=stdout,
        stderr=stderr,
    )

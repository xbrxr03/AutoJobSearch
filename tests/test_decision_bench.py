import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from autojobsearch.decision_bench import load_decision_cases, run_decision_benchmark
from autojobsearch.decisions import (
    ChoiceDecisionOutput,
    DecisionEngine,
    DecisionRoute,
)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class StaticProvider:
    model = "static-local"

    def structured(self, *, system: str, prompt: str, schema: type[SchemaT]) -> SchemaT:
        assert "deterministic automation" in system
        return schema.model_validate(
            {
                "selected_id": "apply",
                "confidence": 0.9,
                "rationale": "Fixture result.",
                "evidence": ["fixture"],
            }
        )


def test_load_and_run_decision_benchmark(tmp_path: Path) -> None:
    trace = tmp_path / "decisions.jsonl"
    payload = {
        "name": "easy-apply",
        "request": {
            "kind": "choice",
            "task": "Pick apply button",
            "choices": [
                {"id": "apply", "label": "Easy Apply"},
                {"id": "save", "label": "Save"},
            ],
            "context": {"buttons": ["Easy Apply", "Save"]},
        },
        "expected": {"selected_id": "apply"},
    }
    trace.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    cases = load_decision_cases(trace)
    report = run_decision_benchmark(
        cases,
        DecisionEngine(StaticProvider(), provider_name="static", model="static-local"),
    )

    assert report.total == 1
    assert report.passed == 1
    assert report.failed == 0
    assert report.results[0].route == DecisionRoute.ACCEPT


def test_static_provider_shape_matches_choice_schema() -> None:
    output = StaticProvider().structured(
        system="deterministic automation",
        prompt="",
        schema=ChoiceDecisionOutput,
    )
    assert output.selected_id == "apply"

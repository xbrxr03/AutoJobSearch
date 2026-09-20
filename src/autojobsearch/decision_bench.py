from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import BaseModel, Field

from .decisions import (
    DecisionEngine,
    DecisionKind,
    DecisionOption,
    DecisionRequest,
    DecisionRoute,
)


class ExpectedDecision(BaseModel):
    selected_id: str | None = None
    answer: bool | None = None
    score: int | None = Field(default=None, ge=0, le=100)


class DecisionBenchmarkCase(BaseModel):
    name: str
    request: DecisionRequest
    expected: ExpectedDecision


class DecisionBenchmarkResult(BaseModel):
    name: str
    passed: bool
    route: DecisionRoute
    confidence: float
    expected: ExpectedDecision
    actual: ExpectedDecision
    elapsed_ms: int


class DecisionBenchmarkReport(BaseModel):
    total: int
    passed: int
    failed: int
    average_elapsed_ms: int
    results: list[DecisionBenchmarkResult]


def load_decision_cases(path: Path) -> list[DecisionBenchmarkCase]:
    cases: list[DecisionBenchmarkCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL case") from exc
        cases.append(DecisionBenchmarkCase.model_validate(payload))
    return cases


def run_decision_benchmark(
    cases: list[DecisionBenchmarkCase], engine: DecisionEngine
) -> DecisionBenchmarkReport:
    results: list[DecisionBenchmarkResult] = []
    for case in cases:
        started = perf_counter()
        request = case.request
        if request.kind == DecisionKind.CHOICE:
            decision = engine.choose(
                task=request.task,
                choices=request.choices,
                context=request.context,
                confidence_threshold=request.confidence_threshold,
                allow_abstain=request.allow_abstain,
            )
            actual = ExpectedDecision(selected_id=decision.selected_id)
        elif request.kind == DecisionKind.BOOLEAN:
            decision = engine.decide_boolean(
                task=request.task,
                context=request.context,
                confidence_threshold=request.confidence_threshold,
                allow_abstain=request.allow_abstain,
            )
            actual = ExpectedDecision(answer=decision.answer)
        else:
            decision = engine.score(
                task=request.task,
                context=request.context,
                min_score=request.min_score,
                max_score=request.max_score,
                confidence_threshold=request.confidence_threshold,
            )
            actual = ExpectedDecision(score=decision.score)
        elapsed_ms = round((perf_counter() - started) * 1000)
        results.append(
            DecisionBenchmarkResult(
                name=case.name,
                passed=actual == case.expected,
                route=decision.route,
                confidence=decision.confidence,
                expected=case.expected,
                actual=actual,
                elapsed_ms=elapsed_ms,
            )
        )

    passed = sum(result.passed for result in results)
    average = round(sum(result.elapsed_ms for result in results) / len(results)) if results else 0
    return DecisionBenchmarkReport(
        total=len(results),
        passed=passed,
        failed=len(results) - passed,
        average_elapsed_ms=average,
        results=results,
    )


def choice_case(
    *,
    name: str,
    task: str,
    choices: list[dict[str, str] | DecisionOption],
    context: dict[str, Any],
    expected_id: str | None,
) -> DecisionBenchmarkCase:
    return DecisionBenchmarkCase(
        name=name,
        request=DecisionRequest(
            kind=DecisionKind.CHOICE,
            task=task,
            choices=[
                choice
                if isinstance(choice, DecisionOption)
                else DecisionOption.model_validate(choice)
                for choice in choices
            ],
            context=context,
        ),
        expected=ExpectedDecision(selected_id=expected_id),
    )

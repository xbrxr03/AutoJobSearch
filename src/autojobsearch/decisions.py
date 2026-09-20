from __future__ import annotations

import json
from enum import StrEnum
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, Field, model_validator

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class DecisionError(RuntimeError):
    pass


class DecisionKind(StrEnum):
    CHOICE = "choice"
    BOOLEAN = "boolean"
    SCORE = "score"


class DecisionRoute(StrEnum):
    ACCEPT = "accept"
    REVIEW = "review"
    ABSTAIN = "abstain"


class DecisionOption(BaseModel):
    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.:-]+$")
    label: str = Field(min_length=1)
    description: str = ""


class DecisionRequest(BaseModel):
    kind: DecisionKind
    task: str = Field(min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)
    choices: list[DecisionOption] = Field(default_factory=list)
    min_score: int = Field(default=0, ge=0, le=100)
    max_score: int = Field(default=100, ge=0, le=100)
    confidence_threshold: float = Field(default=0.75, ge=0, le=1)
    allow_abstain: bool = True

    @model_validator(mode="after")
    def request_matches_kind(self) -> DecisionRequest:
        if self.kind == DecisionKind.CHOICE and not self.choices:
            raise ValueError("choice decisions require at least one option")
        if self.kind != DecisionKind.CHOICE and self.choices:
            raise ValueError("only choice decisions accept choices")
        if self.min_score > self.max_score:
            raise ValueError("min_score cannot exceed max_score")
        return self


class DecisionResult(BaseModel):
    kind: DecisionKind
    route: DecisionRoute
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1, max_length=500)
    selected_id: str | None = None
    answer: bool | None = None
    score: int | None = Field(default=None, ge=0, le=100)
    evidence: list[str] = Field(default_factory=list)
    provider: str = "local"
    model: str | None = None

    @property
    def needs_review(self) -> bool:
        return self.route in {DecisionRoute.REVIEW, DecisionRoute.ABSTAIN}


class ChoiceDecisionOutput(BaseModel):
    selected_id: str | None
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1, max_length=500)
    evidence: list[str] = Field(default_factory=list, max_length=5)


class BooleanDecisionOutput(BaseModel):
    answer: bool | None
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1, max_length=500)
    evidence: list[str] = Field(default_factory=list, max_length=5)


class ScoreDecisionOutput(BaseModel):
    score: int = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1, max_length=500)
    evidence: list[str] = Field(default_factory=list, max_length=5)


class StructuredDecisionProvider(Protocol):
    model: str

    def structured(self, *, system: str, prompt: str, schema: type[SchemaT]) -> SchemaT:
        ...


SYSTEM_PROMPT = """You are a local typed decision model for deterministic automation.
Make only the requested closed decision. Use the supplied context and choices only.
Do not invent facts. If the evidence is insufficient, return null where the schema allows it
and use low confidence. Return only schema-valid JSON."""


class DecisionEngine:
    def __init__(
        self,
        provider: StructuredDecisionProvider,
        *,
        provider_name: str = "ollama",
        model: str | None = None,
    ) -> None:
        self.provider = provider
        self.provider_name = provider_name
        self.model = model or getattr(provider, "model", None)

    def choose(
        self,
        *,
        task: str,
        choices: list[DecisionOption],
        context: dict[str, Any] | None = None,
        confidence_threshold: float = 0.75,
        allow_abstain: bool = True,
    ) -> DecisionResult:
        request = DecisionRequest(
            kind=DecisionKind.CHOICE,
            task=task,
            context=context or {},
            choices=choices,
            confidence_threshold=confidence_threshold,
            allow_abstain=allow_abstain,
        )
        output = self.provider.structured(
            system=SYSTEM_PROMPT,
            prompt=_decision_prompt(request),
            schema=ChoiceDecisionOutput,
        )
        valid_ids = {choice.id for choice in choices}
        if output.selected_id is None:
            route = DecisionRoute.ABSTAIN
        elif output.selected_id not in valid_ids:
            raise DecisionError(f"Decision selected unknown option: {output.selected_id}")
        elif output.confidence < confidence_threshold:
            route = DecisionRoute.REVIEW
        else:
            route = DecisionRoute.ACCEPT
        if route == DecisionRoute.ABSTAIN and not allow_abstain:
            route = DecisionRoute.REVIEW
        return DecisionResult(
            kind=DecisionKind.CHOICE,
            route=route,
            confidence=output.confidence,
            rationale=output.rationale,
            selected_id=output.selected_id,
            evidence=output.evidence,
            provider=self.provider_name,
            model=self.model,
        )

    def decide_boolean(
        self,
        *,
        task: str,
        context: dict[str, Any] | None = None,
        confidence_threshold: float = 0.75,
        allow_abstain: bool = True,
    ) -> DecisionResult:
        request = DecisionRequest(
            kind=DecisionKind.BOOLEAN,
            task=task,
            context=context or {},
            confidence_threshold=confidence_threshold,
            allow_abstain=allow_abstain,
        )
        output = self.provider.structured(
            system=SYSTEM_PROMPT,
            prompt=_decision_prompt(request),
            schema=BooleanDecisionOutput,
        )
        if output.answer is None:
            route = DecisionRoute.ABSTAIN if allow_abstain else DecisionRoute.REVIEW
        elif output.confidence < confidence_threshold:
            route = DecisionRoute.REVIEW
        else:
            route = DecisionRoute.ACCEPT
        return DecisionResult(
            kind=DecisionKind.BOOLEAN,
            route=route,
            confidence=output.confidence,
            rationale=output.rationale,
            answer=output.answer,
            evidence=output.evidence,
            provider=self.provider_name,
            model=self.model,
        )

    def score(
        self,
        *,
        task: str,
        context: dict[str, Any] | None = None,
        min_score: int = 0,
        max_score: int = 100,
        confidence_threshold: float = 0.75,
    ) -> DecisionResult:
        request = DecisionRequest(
            kind=DecisionKind.SCORE,
            task=task,
            context=context or {},
            min_score=min_score,
            max_score=max_score,
            confidence_threshold=confidence_threshold,
            allow_abstain=False,
        )
        output = self.provider.structured(
            system=SYSTEM_PROMPT,
            prompt=_decision_prompt(request),
            schema=ScoreDecisionOutput,
        )
        if not min_score <= output.score <= max_score:
            raise DecisionError(
                f"Decision score {output.score} is outside allowed range {min_score}-{max_score}"
            )
        route = (
            DecisionRoute.ACCEPT
            if output.confidence >= confidence_threshold
            else DecisionRoute.REVIEW
        )
        return DecisionResult(
            kind=DecisionKind.SCORE,
            route=route,
            confidence=output.confidence,
            rationale=output.rationale,
            score=output.score,
            evidence=output.evidence,
            provider=self.provider_name,
            model=self.model,
        )


def _decision_prompt(request: DecisionRequest) -> str:
    payload: dict[str, Any] = {
        "kind": request.kind.value,
        "task": request.task,
        "context": request.context,
        "confidence_threshold": request.confidence_threshold,
        "routing_contract": {
            "high_confidence": "The caller may proceed.",
            "low_confidence": "The caller will pause for review.",
            "null_answer": "The caller will treat this as an abstention or review.",
        },
    }
    if request.kind == DecisionKind.CHOICE:
        payload["choices"] = [choice.model_dump() for choice in request.choices]
        payload["instructions"] = [
            "Select exactly one supplied choice id when the evidence is sufficient.",
            "Never return a choice id that is not in choices.",
            "Return selected_id null when no supplied choice is supported.",
        ]
    elif request.kind == DecisionKind.BOOLEAN:
        payload["instructions"] = [
            "Return true or false only when the evidence supports it.",
            "Return answer null when the evidence is insufficient.",
        ]
    else:
        payload["score_range"] = {"min": request.min_score, "max": request.max_score}
        payload["instructions"] = [
            "Return an integer score inside the supplied inclusive range.",
            "Use lower confidence when the context is sparse or ambiguous.",
        ]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)

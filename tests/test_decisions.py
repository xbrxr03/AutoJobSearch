import json

import httpx
import pytest
import respx

from autojobsearch.decisions import DecisionEngine, DecisionError, DecisionOption, DecisionRoute
from autojobsearch.llm import OllamaClient


def options() -> list[DecisionOption]:
    return [
        DecisionOption(id="apply", label="Easy Apply"),
        DecisionOption(id="save", label="Save"),
        DecisionOption(id="dismiss", label="Dismiss"),
    ]


@respx.mock
def test_choice_decision_accepts_known_option() -> None:
    route = respx.post("http://127.0.0.1:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {
                    "content": json.dumps(
                        {
                            "selected_id": "apply",
                            "confidence": 0.91,
                            "rationale": "The button text directly matches the target action.",
                            "evidence": ["Visible button: Easy Apply"],
                        }
                    )
                }
            },
        )
    )

    engine = DecisionEngine(OllamaClient("http://127.0.0.1:11434", "qwen3.5:9b"))
    result = engine.choose(
        task="Choose the button that starts an application.",
        choices=options(),
        context={"visible_buttons": ["Easy Apply", "Save", "Dismiss"]},
    )

    assert result.route == DecisionRoute.ACCEPT
    assert result.selected_id == "apply"
    assert not result.needs_review
    request = json.loads(route.calls.last.request.content)
    prompt = json.loads(request["messages"][1]["content"])
    assert prompt["kind"] == "choice"
    assert {choice["id"] for choice in prompt["choices"]} == {"apply", "save", "dismiss"}
    assert request["think"] is False


@respx.mock
def test_low_confidence_choice_routes_to_review() -> None:
    respx.post("http://127.0.0.1:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {
                    "content": json.dumps(
                        {
                            "selected_id": "apply",
                            "confidence": 0.42,
                            "rationale": "The visible text is partially clipped.",
                        }
                    )
                }
            },
        )
    )

    engine = DecisionEngine(OllamaClient("http://127.0.0.1:11434", "qwen3.5:9b"))
    result = engine.choose(
        task="Choose the button that starts an application.",
        choices=options(),
        context={"visible_buttons": ["Apply...", "Save"]},
    )

    assert result.route == DecisionRoute.REVIEW
    assert result.needs_review


@respx.mock
def test_unknown_choice_id_is_rejected() -> None:
    respx.post("http://127.0.0.1:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {
                    "content": json.dumps(
                        {
                            "selected_id": "invented",
                            "confidence": 0.99,
                            "rationale": "Invalid id should be rejected by caller validation.",
                        }
                    )
                }
            },
        )
    )

    engine = DecisionEngine(OllamaClient("http://127.0.0.1:11434", "qwen3.5:9b"))
    with pytest.raises(DecisionError, match="unknown option"):
        engine.choose(
            task="Choose the button that starts an application.",
            choices=options(),
            context={"visible_buttons": ["Easy Apply"]},
        )


@respx.mock
def test_boolean_decision_can_abstain() -> None:
    respx.post("http://127.0.0.1:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {
                    "content": json.dumps(
                        {
                            "answer": None,
                            "confidence": 0.2,
                            "rationale": "The page text does not mention a confirmation.",
                        }
                    )
                }
            },
        )
    )

    engine = DecisionEngine(OllamaClient("http://127.0.0.1:11434", "qwen3.5:9b"))
    result = engine.decide_boolean(
        task="Does the page show application submission confirmation?",
        context={"visible_text": "Thanks for visiting."},
    )

    assert result.route == DecisionRoute.ABSTAIN
    assert result.answer is None


@respx.mock
def test_score_decision_enforces_requested_range() -> None:
    respx.post("http://127.0.0.1:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {
                    "content": json.dumps(
                        {
                            "score": 87,
                            "confidence": 0.88,
                            "rationale": "The role is an exact title and location match.",
                        }
                    )
                }
            },
        )
    )

    engine = DecisionEngine(OllamaClient("http://127.0.0.1:11434", "qwen3.5:9b"))
    with pytest.raises(DecisionError, match="outside allowed range"):
        engine.score(
            task="Score job fit after deterministic filters.",
            context={"title": "Software Engineer"},
            min_score=0,
            max_score=80,
        )

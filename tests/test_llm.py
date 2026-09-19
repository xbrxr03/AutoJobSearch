import json

import httpx
import pytest
import respx

from autojobsearch.llm import OllamaClient, OllamaError
from autojobsearch.models import ApplicantProfile, FitAssessment, JobPosting


def sample_profile() -> ApplicantProfile:
    return ApplicantProfile.model_validate(
        {
            "person": {
                "first_name": "Jane",
                "last_name": "Doe",
                "email": "jane@example.com",
                "phone": "+1 555 010 0100",
                "city": "Toronto",
                "region": "Ontario",
                "country": "Canada",
            },
            "preferences": {
                "target_titles": ["software engineer"],
                "locations": ["Toronto"],
            },
            "facts": [
                {
                    "id": "fact-1",
                    "kind": "project",
                    "text": "Built a Python service.",
                    "tags": ["python"],
                }
            ],
        }
    )


def sample_job() -> JobPosting:
    return JobPosting(
        source="fixture",
        external_id="1",
        url="https://example.com/jobs/1",
        title="Software Engineer",
        company="Example",
        location="Toronto",
        description="Build Python services.",
    )


@respx.mock
def test_list_models() -> None:
    respx.get("http://127.0.0.1:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "qwen3.5:9b"}]})
    )
    client = OllamaClient("http://127.0.0.1:11434", "qwen3.5:9b")
    assert client.list_models() == ["qwen3.5:9b"]


@respx.mock
def test_structured_assessment_and_fact_validation() -> None:
    assessment = FitAssessment(
        score=84,
        recommendation="strong_match",
        matched_requirements=["Python"],
        missing_requirements=[],
        evidence_fact_ids=["fact-1"],
        explanation="The supplied project demonstrates Python experience.",
    )
    route = respx.post("http://127.0.0.1:11434/api/chat").mock(
        return_value=httpx.Response(
            200, json={"message": {"content": assessment.model_dump_json()}}
        )
    )
    client = OllamaClient("http://127.0.0.1:11434", "qwen3.5:9b")
    result = client.assess_fit(sample_job(), sample_profile())
    assert result.score == 84
    request = json.loads(route.calls.last.request.content)
    assert request["think"] is False
    assert request["options"]["num_predict"] == 1024


@respx.mock
def test_rejects_unknown_evidence_fact() -> None:
    payload = {
        "score": 95,
        "recommendation": "strong_match",
        "matched_requirements": ["Kubernetes"],
        "missing_requirements": [],
        "evidence_fact_ids": ["invented-fact"],
        "explanation": "Unsupported claim.",
    }
    respx.post("http://127.0.0.1:11434/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"content": json.dumps(payload)}})
    )
    client = OllamaClient("http://127.0.0.1:11434", "qwen3.5:9b")
    with pytest.raises(OllamaError, match="unknown profile facts"):
        client.assess_fit(sample_job(), sample_profile())


def test_assessment_rejects_score_recommendation_mismatch() -> None:
    with pytest.raises(ValueError, match="requires recommendation"):
        FitAssessment(
            matched_requirements=["Python"],
            missing_requirements=[],
            evidence_fact_ids=["fact-1"],
            score=90,
            recommendation="skip",
            explanation="Contradictory output.",
        )


def test_assessment_rejects_unsupported_match() -> None:
    with pytest.raises(ValueError, match="require evidence"):
        FitAssessment(
            matched_requirements=["Python"],
            missing_requirements=[],
            evidence_fact_ids=[],
            score=70,
            recommendation="match",
            explanation="Unsupported match.",
        )


def test_assessment_rejects_requirement_in_both_lists() -> None:
    with pytest.raises(ValueError, match="both matched and missing"):
        FitAssessment(
            matched_requirements=["Python"],
            missing_requirements=["python"],
            evidence_fact_ids=["fact-1"],
            score=50,
            recommendation="borderline",
            explanation="Contradictory classification.",
        )


@respx.mock
def test_structured_retries_once_after_validation_error() -> None:
    invalid = {
        "matched_requirements": [],
        "missing_requirements": ["Python"],
        "evidence_fact_ids": ["fact-1"],
        "score": 0,
        "recommendation": "skip",
        "explanation": "Evidence contradicts the empty match list.",
    }
    corrected = {
        **invalid,
        "evidence_fact_ids": [],
        "explanation": "Python is unsupported, so no evidence is cited.",
    }
    route = respx.post("http://127.0.0.1:11434/api/chat").mock(
        side_effect=[
            httpx.Response(200, json={"message": {"content": json.dumps(invalid)}}),
            httpx.Response(200, json={"message": {"content": json.dumps(corrected)}}),
        ]
    )
    result = OllamaClient("http://127.0.0.1:11434", "qwen3.5:9b").structured(
        system="test", prompt="test", schema=FitAssessment
    )
    assert result.evidence_fact_ids == []
    assert route.call_count == 2

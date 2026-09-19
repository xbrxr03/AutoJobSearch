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
        recommendation="apply",
        matched_requirements=["Python"],
        missing_requirements=[],
        evidence_fact_ids=["fact-1"],
        explanation="The supplied project demonstrates Python experience.",
    )
    respx.post("http://127.0.0.1:11434/api/chat").mock(
        return_value=httpx.Response(
            200, json={"message": {"content": assessment.model_dump_json()}}
        )
    )
    client = OllamaClient("http://127.0.0.1:11434", "qwen3.5:9b")
    result = client.assess_fit(sample_job(), sample_profile())
    assert result.score == 84


@respx.mock
def test_rejects_unknown_evidence_fact() -> None:
    payload = {
        "score": 95,
        "recommendation": "apply",
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

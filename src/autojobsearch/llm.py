from __future__ import annotations

import json
from typing import TypeVar

import httpx
from pydantic import BaseModel

from .models import ApplicantProfile, FitAssessment, JobPosting

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: float = 180.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def list_models(self) -> list[str]:
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama is unavailable at {self.base_url}: {exc}") from exc
        return [item["name"] for item in response.json().get("models", [])]

    def structured(self, *, system: str, prompt: str, schema: type[SchemaT]) -> SchemaT:
        payload = {
            "model": self.model,
            "stream": False,
            "format": schema.model_json_schema(),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": 0, "num_ctx": 16384},
        }
        try:
            response = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
            content = response.json()["message"]["content"]
            return schema.model_validate_json(content)
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise OllamaError(f"Invalid Ollama response for {schema.__name__}: {exc}") from exc

    def assess_fit(self, job: JobPosting, profile: ApplicantProfile) -> FitAssessment:
        facts = [fact.model_dump() for fact in profile.facts]
        prompt = json.dumps(
            {
                "job": {
                    "title": job.title,
                    "company": job.company,
                    "location": job.location,
                    "description": job.description,
                },
                "applicant_facts": facts,
                "instructions": [
                    "Use only the supplied applicant facts as evidence.",
                    "Never invent skills, credentials, dates, or experience.",
                    "Every evidence_fact_id must exactly match a supplied fact id.",
                    "Return a conservative 0-100 fit score.",
                ],
            },
            ensure_ascii=False,
        )
        result = self.structured(
            system="You are a conservative job-fit evaluator. Return only schema-valid JSON.",
            prompt=prompt,
            schema=FitAssessment,
        )
        valid_ids = {fact.id for fact in profile.facts}
        unknown = set(result.evidence_fact_ids) - valid_ids
        if unknown:
            raise OllamaError(f"Model cited unknown profile facts: {sorted(unknown)}")
        return result

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
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "keep_alive": "10m",
            "format": schema.model_json_schema(),
            "messages": messages,
            "options": {"temperature": 0, "num_ctx": 16384, "num_predict": 1024},
        }
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = httpx.post(
                    f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise OllamaError(f"Ollama request failed for {schema.__name__}: {exc}") from exc

            try:
                content = response.json()["message"]["content"]
                return schema.model_validate_json(content)
            except (KeyError, ValueError) as exc:
                last_error = exc
                if attempt == 0:
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": content if "content" in locals() else "",
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"That response failed schema validation: {exc}. "
                                    "Correct every reported issue and return only valid JSON."
                                ),
                            },
                        ]
                    )

        raise OllamaError(f"Invalid Ollama response for {schema.__name__}: {last_error}")

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
                    "Identify each concrete job requirement and put it in exactly one of "
                    "matched_requirements or missing_requirements.",
                    "A requirement is matched only when a supplied fact directly supports it; "
                    "otherwise it is missing.",
                    "Fill matched_requirements, missing_requirements, and evidence_fact_ids "
                    "before choosing the score. Keep all fields semantically consistent.",
                    "Use this exact score rubric: 0-39 skip, 40-59 borderline, 60-79 match, "
                    "80-95 strong_match. Never return a score above 95.",
                    "If no applicant facts are supplied, score 0, recommend skip, cite no "
                    "evidence, and list all concrete requirements as missing.",
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

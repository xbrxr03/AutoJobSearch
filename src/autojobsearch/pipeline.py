from __future__ import annotations

from .llm import OllamaClient
from .models import ApplicantProfile, FitAssessment, HardFilterResult, JobPosting, JobStatus
from .scoring import hard_filter
from .storage import Store


class Pipeline:
    def __init__(self, store: Store, profile: ApplicantProfile, llm: OllamaClient | None) -> None:
        self.store = store
        self.profile = profile
        self.llm = llm

    def ingest_and_score(
        self, job: JobPosting
    ) -> tuple[int, HardFilterResult, FitAssessment | None]:
        job_id = self.store.upsert_job(job)
        rules = hard_filter(job, self.profile)
        if not rules.accepted:
            self.store.transition(job_id, JobStatus.REJECTED, rules.model_dump())
            return job_id, rules, None

        assessment = self.llm.assess_fit(job, self.profile) if self.llm else None
        effective_score = assessment.score if assessment else rules.score
        status = (
            JobStatus.SHORTLISTED
            if effective_score >= self.profile.preferences.minimum_score
            else JobStatus.REJECTED
        )
        payload = assessment.model_dump() if assessment else rules.model_dump()
        self.store.transition(job_id, status, payload)
        return job_id, rules, assessment

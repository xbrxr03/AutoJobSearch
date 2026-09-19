from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, field_validator


class JobStatus(StrEnum):
    DISCOVERED = "discovered"
    REJECTED = "rejected"
    SHORTLISTED = "shortlisted"
    MATERIALS_READY = "materials_ready"
    READY_FOR_REVIEW = "ready_for_review"
    APPROVED = "approved"
    FILLING = "filling"
    MANUAL_ACTION = "manual_action"
    SUBMISSION_CONFIRMED = "submission_confirmed"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


class ApplicantFact(BaseModel):
    id: str
    kind: str
    text: str
    tags: list[str] = Field(default_factory=list)


class Person(BaseModel):
    first_name: str
    last_name: str
    email: str
    phone: str
    city: str
    region: str
    country: str
    linkedin_url: str = ""
    github_url: str = ""
    portfolio_url: str = ""


class Preferences(BaseModel):
    target_titles: list[str]
    locations: list[str]
    remote_allowed: bool = True
    max_required_years: int = 3
    minimum_score: int = Field(default=60, ge=0, le=100)
    blocked_keywords: list[str] = Field(default_factory=list)
    blocked_companies: list[str] = Field(default_factory=list)


class ApplicantProfile(BaseModel):
    person: Person
    preferences: Preferences
    facts: list[ApplicantFact] = Field(default_factory=list)
    approved_answers: dict[str, str] = Field(default_factory=dict)


class JobPosting(BaseModel):
    source: str
    external_id: str
    url: HttpUrl
    title: str
    company: str
    location: str = ""
    description: str = ""
    is_remote: bool | None = None
    employment_type: str | None = None
    date_posted: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("title", "company")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class HardFilterResult(BaseModel):
    accepted: bool
    score: int = Field(ge=0, le=100)
    reasons: list[str] = Field(default_factory=list)


class FitAssessment(BaseModel):
    score: int = Field(ge=0, le=100)
    recommendation: str
    matched_requirements: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    evidence_fact_ids: list[str] = Field(default_factory=list)
    explanation: str


class ApplicationEvidence(BaseModel):
    confirmation_url: str
    confirmation_text: str
    screenshot_path: str | None = None
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_positive(self) -> bool:
        text = self.confirmation_text.casefold()
        markers = (
            "application submitted",
            "application has been submitted",
            "thank you for applying",
            "we received your application",
            "application received",
        )
        return any(marker in text for marker in markers)


class FormField(BaseModel):
    selector: str
    label: str
    field_type: str
    required: bool = False
    options: list[str] = Field(default_factory=list)


class FillAction(BaseModel):
    selector: str
    label: str
    value: str
    source: str
    requires_review: bool = False


class ApplicationPlan(BaseModel):
    job_id: int
    actions: list[FillAction] = Field(default_factory=list)
    unresolved: list[FormField] = Field(default_factory=list)

    @property
    def ready_for_review(self) -> bool:
        return not any(field.required for field in self.unresolved)


class PlanApproval(BaseModel):
    job_id: int
    plan_digest: str
    approved_by: str = "local-user"
    approved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class FillVerification(BaseModel):
    selector: str
    expected: str
    actual: str
    matched: bool


class BrowserExecutionResult(BaseModel):
    verified: list[FillVerification] = Field(default_factory=list)
    submitted: bool = False
    evidence: ApplicationEvidence | None = None
    status: JobStatus

from __future__ import annotations

import hashlib

from ..models import ApplicationPlan, PlanApproval


class ReviewRequiredError(RuntimeError):
    pass


def plan_digest(plan: ApplicationPlan) -> str:
    canonical = plan.model_dump_json(exclude_none=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def approve_plan(plan: ApplicationPlan, approved_by: str = "local-user") -> PlanApproval:
    if not plan.ready_for_review:
        raise ReviewRequiredError("Cannot approve a plan with unresolved required fields")
    return PlanApproval(
        job_id=plan.job_id,
        plan_digest=plan_digest(plan),
        approved_by=approved_by,
    )


def validate_approval(plan: ApplicationPlan, approval: PlanApproval | None) -> None:
    if not plan.ready_for_review:
        raise ReviewRequiredError("Submission is blocked by unresolved required fields")
    if approval is None:
        raise ReviewRequiredError("Submission requires an explicit plan approval")
    if approval.job_id != plan.job_id or approval.plan_digest != plan_digest(plan):
        raise ReviewRequiredError("Approval does not match the current application plan")

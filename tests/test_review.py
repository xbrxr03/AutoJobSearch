import pytest

from autojobsearch.application import (
    ReviewRequiredError,
    approve_plan,
    validate_approval,
)
from autojobsearch.models import ApplicationPlan, FillAction, FormField


def plan() -> ApplicationPlan:
    return ApplicationPlan(
        job_id=7,
        actions=[
            FillAction(
                selector="#email",
                label="Email",
                value="jane@example.com",
                source="profile.person.email",
            )
        ],
    )


def test_approval_is_bound_to_exact_plan() -> None:
    original = plan()
    approval = approve_plan(original)
    validate_approval(original, approval)

    changed = original.model_copy(deep=True)
    changed.actions[0].value = "other@example.com"
    with pytest.raises(ReviewRequiredError, match="does not match"):
        validate_approval(changed, approval)


def test_unresolved_required_fields_cannot_be_approved() -> None:
    blocked = ApplicationPlan(
        job_id=7,
        unresolved=[
            FormField(
                selector="#question",
                label="Unknown required question",
                field_type="text",
                required=True,
            )
        ],
    )
    with pytest.raises(ReviewRequiredError, match="unresolved required"):
        approve_plan(blocked)


def test_submission_without_approval_is_rejected() -> None:
    with pytest.raises(ReviewRequiredError, match="explicit plan approval"):
        validate_approval(plan(), None)


def test_fabricated_approval_cannot_bypass_unresolved_fields() -> None:
    blocked = ApplicationPlan(
        job_id=7,
        unresolved=[
            FormField(
                selector="#question",
                label="Unknown required question",
                field_type="text",
                required=True,
            )
        ],
    )
    fabricated = approve_plan(plan())
    fabricated.job_id = 7
    with pytest.raises(ReviewRequiredError, match="blocked by unresolved"):
        validate_approval(blocked, fabricated)

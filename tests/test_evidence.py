from autojobsearch.models import ApplicationEvidence


def test_requires_positive_confirmation_text() -> None:
    positive = ApplicationEvidence(
        confirmation_url="https://example.com/thanks",
        confirmation_text="Thank you for applying. We received your application.",
    )
    ambiguous = ApplicationEvidence(
        confirmation_url="https://example.com/apply",
        confirmation_text="No validation errors found.",
    )
    assert positive.is_positive
    assert not ambiguous.is_positive

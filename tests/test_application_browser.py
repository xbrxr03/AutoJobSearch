from autojobsearch.application.browser import ApplicationBrowser


def test_phone_verification_ignores_display_formatting() -> None:
    assert ApplicationBrowser._matches("6475229553", "(647) 522-9553", "tel")


def test_phone_verification_still_rejects_different_number() -> None:
    assert not ApplicationBrowser._matches("6475229553", "(647) 522-9554", "tel")

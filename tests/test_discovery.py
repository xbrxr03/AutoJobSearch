import httpx
import respx

from autojobsearch.discovery import GreenhouseDiscovery


@respx.mock
def test_greenhouse_discovery_normalizes_public_jobs() -> None:
    route = respx.get("https://boards-api.greenhouse.io/v1/boards/example/jobs").mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "id": 123,
                        "title": "Software Engineer",
                        "absolute_url": "https://job-boards.greenhouse.io/example/jobs/123",
                        "updated_at": "2026-09-18T12:00:00-04:00",
                        "location": {"name": "Toronto, Ontario"},
                        "content": "<p>Build useful software &amp; ship it.</p>",
                    }
                ]
            },
        )
    )

    jobs = GreenhouseDiscovery().discover("example")

    assert route.called
    assert len(jobs) == 1
    assert jobs[0].external_id == "123"
    assert jobs[0].description == "Build useful software & ship it."

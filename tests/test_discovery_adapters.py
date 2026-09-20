import httpx
import respx

from autojobsearch.discovery import AshbyDiscovery, LeverDiscovery


@respx.mock
def test_lever_discovery() -> None:
    respx.get("https://api.lever.co/v0/postings/example").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "id": "abc",
                    "text": "Software Developer",
                    "hostedUrl": "https://jobs.lever.co/example/abc",
                    "applyUrl": "https://jobs.lever.co/example/abc/apply",
                    "descriptionPlain": "Build APIs.",
                    "lists": [
                        {
                            "text": "Requirements",
                            "content": "<ul><li>2+ years with Python.</li></ul>",
                        }
                    ],
                    "workplaceType": "remote",
                    "categories": {"location": "Toronto", "commitment": "Full-time"},
                }
            ],
        )
    )
    jobs = LeverDiscovery().discover("example")
    assert jobs[0].external_id == "abc"
    assert jobs[0].location == "Toronto"
    assert jobs[0].is_remote is True
    assert "Requirements" in jobs[0].description
    assert "2+ years with Python." in jobs[0].description
    assert jobs[0].metadata["workplace_type"] == "remote"


@respx.mock
def test_ashby_discovery() -> None:
    respx.get("https://api.ashbyhq.com/posting-api/job-board/example").mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "id": "job-1",
                        "title": "Software Engineer",
                        "jobUrl": "https://jobs.ashbyhq.com/example/job-1",
                        "applyUrl": "https://jobs.ashbyhq.com/example/job-1/application",
                        "location": "Remote Canada",
                        "descriptionPlain": "Build reliable services.",
                        "isRemote": True,
                        "workplaceType": "Remote",
                        "employmentType": "FullTime",
                    }
                ]
            },
        )
    )
    jobs = AshbyDiscovery().discover("example")
    assert jobs[0].is_remote is True
    assert jobs[0].metadata["apply_url"].endswith("/application")
    assert jobs[0].metadata["workplace_type"] == "Remote"

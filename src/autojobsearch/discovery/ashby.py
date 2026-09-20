from __future__ import annotations

import httpx

from ..models import JobPosting


class AshbyDiscovery:
    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def discover(self, organization: str) -> list[JobPosting]:
        response = httpx.get(
            f"https://api.ashbyhq.com/posting-api/job-board/{organization}",
            timeout=self.timeout,
        )
        response.raise_for_status()
        jobs = []
        for item in response.json().get("jobs", []):
            jobs.append(
                JobPosting(
                    source="ashby",
                    external_id=item["id"],
                    url=item["jobUrl"],
                    title=item["title"],
                    company=organization,
                    location=item.get("location", ""),
                    description=item.get("descriptionPlain", ""),
                    is_remote=item.get("isRemote"),
                    employment_type=item.get("employmentType"),
                    metadata={
                        "organization": organization,
                        "apply_url": item.get("applyUrl"),
                        "workplace_type": item.get("workplaceType"),
                    },
                )
            )
        return jobs

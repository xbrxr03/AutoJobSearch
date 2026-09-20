from __future__ import annotations

import re

import httpx

from ..models import JobPosting


def _plain_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value)).strip()


class LeverDiscovery:
    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def discover(self, site: str) -> list[JobPosting]:
        response = httpx.get(
            f"https://api.lever.co/v0/postings/{site}",
            params={"mode": "json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        jobs = []
        for item in response.json():
            categories = item.get("categories") or {}
            description_parts = [
                item.get("descriptionPlain") or item.get("description") or "",
                item.get("additionalPlain") or item.get("additional") or "",
            ]
            jobs.append(
                JobPosting(
                    source="lever",
                    external_id=item["id"],
                    url=item.get("hostedUrl") or item["applyUrl"],
                    title=item["text"],
                    company=site,
                    location=categories.get("location", ""),
                    description=_plain_text(" ".join(description_parts)),
                    is_remote=item.get("workplaceType", "").casefold() == "remote",
                    employment_type=categories.get("commitment"),
                    metadata={
                        "site": site,
                        "apply_url": item.get("applyUrl"),
                        "workplace_type": item.get("workplaceType"),
                    },
                )
            )
        return jobs

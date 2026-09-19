from __future__ import annotations

import html
import re
from datetime import datetime

import httpx

from ..models import JobPosting


def _plain_text(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", html.unescape(value))
    return re.sub(r"\s+", " ", without_tags).strip()


class GreenhouseDiscovery:
    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def discover(self, board_token: str) -> list[JobPosting]:
        url = f"https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs"
        response = httpx.get(url, params={"content": "true"}, timeout=self.timeout)
        response.raise_for_status()
        jobs: list[JobPosting] = []
        for item in response.json().get("jobs", []):
            jobs.append(
                JobPosting(
                    source="greenhouse",
                    external_id=str(item["id"]),
                    url=item["absolute_url"],
                    title=item["title"],
                    company=board_token,
                    location=(item.get("location") or {}).get("name", ""),
                    description=_plain_text(item.get("content", "")),
                    date_posted=(
                        datetime.fromisoformat(item["updated_at"])
                        if item.get("updated_at")
                        else None
                    ),
                    metadata={"board_token": board_token},
                )
            )
        return jobs

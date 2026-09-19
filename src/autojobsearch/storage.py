from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .models import JobPosting, JobStatus

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY,
    fingerprint TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL,
    external_id TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    company TEXT NOT NULL,
    location TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    job_id INTEGER NOT NULL REFERENCES jobs(id),
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def fingerprint(job: JobPosting) -> str:
    stable = f"{job.source}|{job.external_id}|{str(job.url).split('?')[0].rstrip('/')}"
    return hashlib.sha256(stable.casefold().encode()).hexdigest()


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    def upsert_job(self, job: JobPosting) -> int:
        job_fingerprint = fingerprint(job)
        payload = json.dumps(job.model_dump(mode="json"), sort_keys=True)
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    fingerprint, source, external_id, url, title, company,
                    location, description, status, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(fingerprint) DO UPDATE SET
                    title=excluded.title,
                    company=excluded.company,
                    location=excluded.location,
                    description=excluded.description,
                    payload_json=excluded.payload_json,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    job_fingerprint,
                    job.source,
                    job.external_id,
                    str(job.url),
                    job.title,
                    job.company,
                    job.location,
                    job.description,
                    JobStatus.DISCOVERED.value,
                    payload,
                ),
            )
            row = connection.execute(
                "SELECT id FROM jobs WHERE fingerprint = ?", (job_fingerprint,)
            ).fetchone()
            assert row is not None
            return int(row["id"])

    def transition(self, job_id: int, status: JobStatus, payload: dict | None = None) -> None:
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE jobs SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status.value, job_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown job id: {job_id}")
            connection.execute(
                "INSERT INTO events (job_id, event_type, payload_json) VALUES (?, ?, ?)",
                (job_id, status.value, json.dumps(payload or {}, sort_keys=True)),
            )

    def job_url(self, job_id: int) -> str:
        with self.connect() as connection:
            row = connection.execute("SELECT url FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown job id: {job_id}")
            return str(row["url"])

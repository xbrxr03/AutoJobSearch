from __future__ import annotations

import json
from pathlib import Path

from .models import ApplicantProfile


def load_profile(path: Path) -> ApplicantProfile:
    return ApplicantProfile.model_validate_json(path.read_text(encoding="utf-8"))


def save_profile(profile: ApplicantProfile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")

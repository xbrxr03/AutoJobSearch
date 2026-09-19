from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    home: Path = Field(default=Path("~/.autojobsearch"), alias="AUTOJOBSEARCH_HOME")
    ollama_base_url: str = Field(default="http://127.0.0.1:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="qwen3.5:9b", alias="OLLAMA_MODEL")
    review_before_submit: bool = Field(default=True, alias="AUTOJOBSEARCH_REVIEW_BEFORE_SUBMIT")

    @property
    def expanded_home(self) -> Path:
        return self.home.expanduser().resolve()

    @property
    def database_path(self) -> Path:
        return self.expanded_home / "autojobsearch.sqlite3"

    @property
    def profile_path(self) -> Path:
        return self.expanded_home / "profile.json"

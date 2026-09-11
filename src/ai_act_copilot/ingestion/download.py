"""Idempotent corpus downloads.

Raw files are never versioned: this module fetches them into ``data/raw/`` and records a
checksum per file, so ``aiact ingest`` is reproducible and re-running it costs nothing.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict

from ai_act_copilot.ingestion.sources import Source
from ai_act_copilot.models import Language
from ai_act_copilot.observability.tracing import observe

MANIFEST_FILENAME = "downloads.json"
_USER_AGENT = "ai-act-copilot/0.1 (+https://github.com/ludvax/ai-act-copilot)"


class DownloadRecord(BaseModel):
    """What was fetched, from where, and what it hashed to."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    language: Language
    url: str
    filename: str
    sha256: str
    size_bytes: int
    fetched_at: datetime


@dataclass(frozen=True, slots=True)
class FetchResult:
    record: DownloadRecord
    from_cache: bool


class Downloader:
    """Fetches sources into ``raw_dir`` and maintains the checksum manifest."""

    def __init__(self, raw_dir: Path, client: httpx.Client | None = None) -> None:
        self.raw_dir = raw_dir
        self._client = client
        self._records = self._load_records()

    @observe(name="download", capture_input=False, capture_output=False)
    def fetch(self, source: Source, language: Language, *, force: bool = False) -> FetchResult:
        filename = source.raw_filename(language)
        path = self.raw_dir / filename
        cached = self._records.get(filename)
        if (
            not force
            and cached is not None
            and path.is_file()
            and _sha256(path.read_bytes()) == cached.sha256
        ):
            return FetchResult(cached, from_cache=True)

        url = source.url_for(language)
        payload = self._get(url, headers=source.request_headers(language))
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

        record = DownloadRecord(
            source_id=source.id,
            language=language,
            url=url,
            filename=filename,
            sha256=_sha256(payload),
            size_bytes=len(payload),
            fetched_at=datetime.now(UTC),
        )
        self._records[filename] = record
        self._save_records()
        return FetchResult(record, from_cache=False)

    def read(self, source: Source, language: Language) -> bytes:
        path = self.raw_dir / source.raw_filename(language)
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing; run `aiact ingest --download` first")
        return path.read_bytes()

    def record_for(self, source: Source, language: Language) -> DownloadRecord | None:
        return self._records.get(source.raw_filename(language))

    def _get(self, url: str, headers: dict[str, str]) -> bytes:
        request_headers = {"User-Agent": _USER_AGENT, **headers}
        if self._client is not None:
            response = self._client.get(url, headers=request_headers)
            response.raise_for_status()
            return response.content
        with httpx.Client(follow_redirects=True, timeout=120.0) as client:
            response = client.get(url, headers=request_headers)
            response.raise_for_status()
            return response.content

    @property
    def _manifest_path(self) -> Path:
        return self.raw_dir / MANIFEST_FILENAME

    def _load_records(self) -> dict[str, DownloadRecord]:
        if not self._manifest_path.is_file():
            return {}
        raw = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        return {key: DownloadRecord.model_validate(value) for key, value in raw.items()}

    def _save_records(self) -> None:
        payload = {key: json.loads(value.model_dump_json()) for key, value in self._records.items()}
        self._manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()

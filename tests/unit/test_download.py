from pathlib import Path

import httpx
import pytest

from ai_act_copilot.ingestion.download import Downloader
from ai_act_copilot.ingestion.sources import Source, SourceFormat
from ai_act_copilot.models import Language

PAYLOAD = b"<html>text</html>"


def _source() -> Source:
    return Source(
        id="sample",
        title="Sample Act",
        short_title="Sample",
        format=SourceFormat.EURLEX_XHTML,
        celex="32024R1689",
        languages=(Language.EN,),
        license="Reuse authorised",
        attribution="© European Union",
    )


def _client(requests: list[httpx.Request], payload: bytes = PAYLOAD) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_writes_file_and_records_checksum(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    downloader = Downloader(tmp_path, client=_client(requests))

    result = downloader.fetch(_source(), Language.EN)

    assert not result.from_cache
    assert (tmp_path / "sample.en.xhtml").read_bytes() == PAYLOAD
    assert result.record.size_bytes == len(PAYLOAD)
    assert requests[0].headers["accept"] == "application/xhtml+xml"
    assert requests[0].headers["accept-language"] == "eng"


def test_second_fetch_uses_the_cached_file(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    downloader = Downloader(tmp_path, client=_client(requests))
    downloader.fetch(_source(), Language.EN)

    result = Downloader(tmp_path, client=_client(requests)).fetch(_source(), Language.EN)

    assert result.from_cache
    assert len(requests) == 1  # the manifest survived the new Downloader instance


def test_force_redownloads(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    downloader = Downloader(tmp_path, client=_client(requests))
    downloader.fetch(_source(), Language.EN)

    result = downloader.fetch(_source(), Language.EN, force=True)

    assert not result.from_cache
    assert len(requests) == 2


def test_read_without_download_explains_what_to_run(tmp_path: Path) -> None:
    downloader = Downloader(tmp_path)

    with pytest.raises(FileNotFoundError, match="aiact ingest --download"):
        downloader.read(_source(), Language.EN)


def test_source_without_celex_requires_a_url_per_language() -> None:
    with pytest.raises(ValueError, match="no celex and no url"):
        Source(
            id="guide",
            title="Guide",
            short_title="Guide",
            format=SourceFormat.PDF,
            languages=(Language.EN, Language.FR),
            license="Open",
            attribution="Someone",
            urls={Language.EN: "http://example.invalid/guide.pdf"},
        )

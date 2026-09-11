from pathlib import Path

import httpx

from ai_act_copilot.chunking.tokenizer import WordTokenCounter
from ai_act_copilot.config import Settings
from ai_act_copilot.ingestion.download import Downloader
from ai_act_copilot.ingestion.pipeline import ingest
from ai_act_copilot.models import ChunkStrategy, Language
from ai_act_copilot.store.sqlite import CorpusStore
from tests.conftest import FIXTURES

MANIFEST = """
sources:
  - id: sample
    title: Sample Act of 2024
    short_title: Sample Act
    format: eurlex-xhtml
    celex: "32024R1689"
    languages: [en]
    license: Reuse authorised with acknowledgement of the source
    attribution: © European Union
"""


def _settings(tmp_path: Path) -> Settings:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "sources.yaml").write_text(MANIFEST, encoding="utf-8")
    return Settings(data_dir=data_dir, chunk_max_tokens=80, chunk_min_tokens=10)


def _downloader(settings: Settings) -> Downloader:
    payload = (FIXTURES / "sample_act.xhtml").read_bytes()
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=payload))
    )
    return Downloader(settings.raw_dir, client=client)


def test_ingest_downloads_parses_chunks_and_stores(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    reports = ingest(
        settings,
        download=True,
        counter=WordTokenCounter(),
        downloader=_downloader(settings),
        strategy=ChunkStrategy.STRUCTURAL,
    )

    assert len(reports) == 1
    report = reports[0]
    assert report.document_id == "sample:en"
    assert report.provisions == 5  # 2 recitals + 2 articles + 1 annex
    assert report.chunks >= 5
    assert report.downloaded
    assert report.average_tokens > 0

    with CorpusStore(settings.database_path) as store:
        chunks = list(store.chunks(strategy=ChunkStrategy.STRUCTURAL, language=Language.EN))
        provision = store.provision("sample:art:1", Language.EN)

    assert {pid for chunk in chunks for pid in chunk.provision_ids} >= {
        "sample:rct:1",
        "sample:art:1",
        "sample:anx:I",
    }
    assert provision is not None
    assert provision.title == "Subject matter"


def test_ingest_is_idempotent_and_reuses_the_cache(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    downloader = _downloader(settings)
    first = ingest(settings, download=True, counter=WordTokenCounter(), downloader=downloader)

    second = ingest(settings, download=True, counter=WordTokenCounter(), downloader=downloader)

    assert first[0].chunks == second[0].chunks
    assert not second[0].downloaded
    with CorpusStore(settings.database_path) as store:
        assert len(store.document_summaries()) == 1


def test_fixed_strategy_produces_headerless_chunks(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    ingest(
        settings,
        download=True,
        counter=WordTokenCounter(),
        downloader=_downloader(settings),
        strategy=ChunkStrategy.FIXED,
    )

    with CorpusStore(settings.database_path) as store:
        chunks = list(store.chunks(strategy=ChunkStrategy.FIXED))

    assert chunks
    assert all(chunk.header == "" for chunk in chunks)

from __future__ import annotations

from pathlib import Path

import pytest

from tests.corpus import build_corpus
from waymark.config import Settings
from waymark.graph.index import build_graph
from waymark.ingest.pipeline import Ingestor
from waymark.nlp.annotator import Annotator
from waymark.reasoning.engine import ReasoningEngine


@pytest.fixture(scope="session")
def corpus_files(tmp_path_factory) -> list[Path]:
    return build_corpus(tmp_path_factory.mktemp("corpus"), long_pages=120)


@pytest.fixture(scope="session")
def settings(tmp_path_factory) -> Settings:
    return Settings(data_dir=tmp_path_factory.mktemp("data"))


@pytest.fixture(scope="session")
def annotator() -> Annotator:
    return Annotator("en_core_web_md")


@pytest.fixture(scope="session")
def ingested(corpus_files, settings, annotator):
    ingestor = Ingestor(settings, annotator)
    return {path.name: ingestor.ingest(path, path.stem) for path in corpus_files}


@pytest.fixture(scope="session")
def graph(ingested):
    notes = [note for result in ingested.values() for note in result.notes]
    return build_graph(notes)


@pytest.fixture(scope="session")
def engine(graph):
    return ReasoningEngine(graph, Annotator("en_core_web_md", role="query"))

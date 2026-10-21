from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from waymark.config import Settings
from waymark.graph.index import KnowledgeGraph, build_graph
from waymark.graph.store import Library, file_digest, new_id
from waymark.ingest.pipeline import Ingestor
from waymark.nlp.annotator import Annotator
from waymark.reasoning.engine import ReasoningEngine

log = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    collection: str
    document: str
    filename: str
    status: str = "queued"
    stage: str = "queued"
    progress: float = 0.0
    error: str | None = None
    created: float = field(default_factory=time.time)
    finished: float | None = None

    def public(self) -> dict:
        return {
            "id": self.id,
            "collection": self.collection,
            "document": self.document,
            "filename": self.filename,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 3),
            "error": self.error,
            "created": self.created,
            "finished": self.finished,
        }


class Runtime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.library = Library(settings.data_dir)
        self.library.ensure_default()
        self._annotator: Annotator | None = None
        self._reader: Annotator | None = None
        self._annotator_lock = threading.Lock()
        self._engines: dict[str, tuple[str, ReasoningEngine]] = {}
        self._engine_locks: dict[str, threading.Lock] = {}
        self._jobs: dict[str, Job] = {}
        self._jobs_lock = threading.Lock()
        self._pending: dict[str, int] = {}
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="waymark-ingest")

    @property
    def annotator(self) -> Annotator:
        with self._annotator_lock:
            if self._annotator is None:
                self._annotator = Annotator(self.settings.spacy_model, batch_size=self.settings.nlp_batch_size)
            return self._annotator

    @property
    def reader(self) -> Annotator:
        with self._annotator_lock:
            if self._reader is None:
                self._reader = Annotator(self.settings.spacy_model, batch_size=8, role="query")
            return self._reader

    def warm(self) -> None:
        self._worker.submit(lambda: (self.annotator, self.reader))

    def shutdown(self) -> None:
        self._worker.shutdown(wait=False, cancel_futures=True)
        self.library.close()

    def submit(self, collection: str, filename: str, suffix: str, staged: Path) -> Job | None:
        digest = file_digest(staged)
        if self.library.find_duplicate(collection, digest[0]) is not None:
            staged.unlink(missing_ok=True)
            return None
        doc = self.library.add_document(collection, filename, suffix, staged, digest)
        job = Job(id=new_id(), collection=collection, document=doc.id, filename=filename)
        with self._jobs_lock:
            self._jobs[job.id] = job
            self._pending[collection] = self._pending.get(collection, 0) + 1
            self._trim_jobs()
        self._worker.submit(self._run, job)
        return job

    def _trim_jobs(self) -> None:
        finished = sorted((j for j in self._jobs.values() if j.finished), key=lambda j: j.finished)
        for job in finished[:-200]:
            self._jobs.pop(job.id, None)

    def job(self, job_id: str) -> Job | None:
        with self._jobs_lock:
            return self._jobs.get(job_id)

    def jobs(self, collection: str) -> list[Job]:
        with self._jobs_lock:
            return [j for j in self._jobs.values() if j.collection == collection]

    def _run(self, job: Job) -> None:
        library = self.library
        doc = library.document(job.document)
        if doc is None:
            self._finish(job, "failed", "document was removed before indexing")
            return
        job.status = "running"
        library.set_status(doc.id, "running")

        def progress(stage: str, fraction: float) -> None:
            job.stage = stage
            job.progress = max(job.progress, min(0.97, fraction))

        try:
            ingestor = Ingestor(self.settings, self.annotator)
            result = ingestor.ingest(library.file_path(doc), doc.id, progress)
            if library.document(doc.id) is None:
                self._finish(job, "failed", "document was removed while indexing")
                return
            meta = {
                "title": result.title,
                "ocr_pages": result.ocr_pages,
                "figures": result.figures,
                "warnings": result.warnings[:20],
                "seconds": round(result.seconds, 2),
            }
            if result.report is not None:
                meta["report"] = result.report
            library.store_notes(doc.id, result.kind, result.pages, result.notes, meta)
            job.stage = "linking graph"
            job.progress = 0.98
            self._after_change(job.collection)
            self._finish(job, "done", None)
        except Exception as exc:
            log.exception("ingest failed for %s", job.filename)
            library.set_status(doc.id, "failed", str(exc)[:400])
            self._finish(job, "failed", str(exc)[:400])
            self._after_change(job.collection)

    def _finish(self, job: Job, status: str, error: str | None) -> None:
        job.status = status
        job.error = error
        job.stage = "indexed" if status == "done" else "failed"
        job.progress = 1.0
        job.finished = time.time()

    def _after_change(self, collection: str) -> None:
        with self._jobs_lock:
            self._pending[collection] = max(0, self._pending.get(collection, 1) - 1)
            remaining = self._pending[collection]
        self.invalidate(collection)
        if remaining == 0:
            try:
                self.engine(collection)
            except Exception:
                log.exception("graph rebuild failed for %s", collection)

    def invalidate(self, collection: str) -> None:
        self._engines.pop(collection, None)

    def engine(self, collection: str) -> ReasoningEngine | None:
        version = self.library.version(collection)
        cached = self._engines.get(collection)
        if cached and cached[0] == version:
            return cached[1]
        lock = self._engine_locks.setdefault(collection, threading.Lock())
        with lock:
            cached = self._engines.get(collection)
            if cached and cached[0] == version:
                return cached[1]
            graph: KnowledgeGraph | None = self.library.load_cached(collection, version)
            if graph is None:
                notes = self.library.notes(collection)
                if not notes:
                    self._engines.pop(collection, None)
                    return None
                graph = build_graph(notes)
                try:
                    self.library.save_cached(collection, version, graph)
                except OSError:
                    log.warning("could not persist graph cache for %s", collection)
            engine = ReasoningEngine(graph, self.reader)
            self._engines[collection] = (version, engine)
            return engine

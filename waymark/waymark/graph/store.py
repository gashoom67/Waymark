from __future__ import annotations

import hashlib
import json
import pickle
import sqlite3
import threading
import time
import uuid
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from waymark import __version__
from waymark.nlp.annotator import PassageNote

SCHEMA = """
CREATE TABLE IF NOT EXISTS collections (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    collection TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    suffix TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    pages INTEGER DEFAULT 0,
    passages INTEGER DEFAULT 0,
    meta TEXT DEFAULT '{}',
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS documents_by_collection ON documents(collection);
CREATE TABLE IF NOT EXISTS passages (
    doc TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    payload BLOB NOT NULL,
    PRIMARY KEY (doc, seq)
);
"""


@dataclass
class DocumentRow:
    id: str
    collection: str
    name: str
    kind: str
    suffix: str
    bytes: int
    sha256: str
    status: str
    error: str | None
    pages: int
    passages: int
    meta: dict
    created: float

    def public(self) -> dict:
        meta = dict(self.meta)
        meta.pop("report", None)
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "bytes": self.bytes,
            "status": self.status,
            "error": self.error,
            "pages": self.pages,
            "passages": self.passages,
            "has_report": "report" in self.meta,
            "meta": meta,
            "created": self.created,
        }


def new_id() -> str:
    return uuid.uuid4().hex[:16]


def file_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


class Library:
    def __init__(self, root: Path):
        self.root = root
        self.files = root / "files"
        self.cache = root / "cache"
        self.files.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(root / "waymark.sqlite3", check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(SCHEMA)
        self._recover()

    def _recover(self) -> None:
        with self._tx() as db:
            db.execute("UPDATE documents SET status='failed', error='interrupted before indexing finished' WHERE status IN ('queued', 'running')")

    @contextmanager
    def _tx(self):
        with self._lock:
            self._db.execute("BEGIN")
            try:
                yield self._db
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            else:
                self._db.execute("COMMIT")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def collections(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                """SELECT c.id, c.name, c.created,
                          COUNT(d.id) AS documents,
                          COALESCE(SUM(CASE WHEN d.status='ready' THEN d.pages ELSE 0 END), 0) AS pages
                   FROM collections c LEFT JOIN documents d ON d.collection = c.id
                   GROUP BY c.id ORDER BY c.created"""
            ).fetchall()
        return [dict(r) for r in rows]

    def collection(self, cid: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT id, name, created FROM collections WHERE id=?", (cid,)).fetchone()
        return dict(row) if row else None

    def create_collection(self, name: str) -> dict:
        cid = new_id()
        clean = " ".join(name.split())[:80] or "Untitled"
        with self._tx() as db:
            db.execute("INSERT INTO collections (id, name, created) VALUES (?, ?, ?)", (cid, clean, time.time()))
        return {"id": cid, "name": clean}

    def rename_collection(self, cid: str, name: str) -> bool:
        clean = " ".join(name.split())[:80] or "Untitled"
        with self._tx() as db:
            cur = db.execute("UPDATE collections SET name=? WHERE id=?", (clean, cid))
        return cur.rowcount > 0

    def delete_collection(self, cid: str) -> bool:
        docs = self.documents(cid)
        with self._tx() as db:
            cur = db.execute("DELETE FROM collections WHERE id=?", (cid,))
        for doc in docs:
            self._remove_file(doc)
        self.drop_cache(cid)
        return cur.rowcount > 0

    def ensure_default(self) -> dict:
        existing = self.collections()
        if existing:
            return existing[0]
        return self.create_collection("My library")

    def _row(self, row: sqlite3.Row) -> DocumentRow:
        data = dict(row)
        data["meta"] = json.loads(data.get("meta") or "{}")
        return DocumentRow(**data)

    def documents(self, cid: str) -> list[DocumentRow]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM documents WHERE collection=? ORDER BY created", (cid,)).fetchall()
        return [self._row(r) for r in rows]

    def document(self, doc_id: str) -> DocumentRow | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        return self._row(row) if row else None

    def file_path(self, doc: DocumentRow) -> Path:
        return self.files / f"{doc.id}{doc.suffix}"

    def add_document(self, cid: str, name: str, suffix: str, source: Path, digest: tuple[str, int] | None = None) -> DocumentRow:
        doc_id = new_id()
        sha, size = digest or file_digest(source)
        target = self.files / f"{doc_id}{suffix}"
        source.replace(target)
        now = time.time()
        with self._tx() as db:
            db.execute(
                "INSERT INTO documents (id, collection, name, kind, suffix, bytes, sha256, status, created) VALUES (?,?,?,?,?,?,?,?,?)",
                (doc_id, cid, name, "pending", suffix, size, sha, "queued", now),
            )
        return self.document(doc_id)

    def find_duplicate(self, cid: str, sha256: str) -> DocumentRow | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM documents WHERE collection=? AND sha256=? AND status IN ('ready', 'queued', 'running')", (cid, sha256)
            ).fetchone()
        return self._row(row) if row else None

    def set_status(self, doc_id: str, status: str, error: str | None = None) -> None:
        with self._tx() as db:
            db.execute("UPDATE documents SET status=?, error=? WHERE id=?", (status, error, doc_id))

    def store_notes(self, doc_id: str, kind: str, pages: int, notes: list[PassageNote], meta: dict) -> None:
        payloads = [(doc_id, i, zlib.compress(json.dumps(n.pack(), separators=(",", ":")).encode("utf-8"), 3)) for i, n in enumerate(notes)]
        with self._tx() as db:
            db.execute("DELETE FROM passages WHERE doc=?", (doc_id,))
            db.executemany("INSERT INTO passages (doc, seq, payload) VALUES (?,?,?)", payloads)
            db.execute(
                "UPDATE documents SET kind=?, pages=?, passages=?, meta=?, status='ready', error=NULL WHERE id=?",
                (kind, pages, len(notes), json.dumps(meta), doc_id),
            )

    def notes(self, cid: str) -> list[PassageNote]:
        with self._lock:
            rows = self._db.execute(
                """SELECT p.payload FROM passages p JOIN documents d ON d.id = p.doc
                   WHERE d.collection=? AND d.status='ready' ORDER BY d.created, p.doc, p.seq""",
                (cid,),
            ).fetchall()
        return [PassageNote.unpack(json.loads(zlib.decompress(r[0]))) for r in rows]

    def delete_document(self, doc_id: str) -> DocumentRow | None:
        doc = self.document(doc_id)
        if doc is None:
            return None
        with self._tx() as db:
            db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        self._remove_file(doc)
        self.drop_cache(doc.collection)
        return doc

    def _remove_file(self, doc: DocumentRow) -> None:
        try:
            self.file_path(doc).unlink(missing_ok=True)
        except OSError:
            pass

    def version(self, cid: str) -> str:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, passages FROM documents WHERE collection=? AND status='ready' ORDER BY id", (cid,)
            ).fetchall()
        joined = __version__ + "|" + "|".join(f"{r[0]}:{r[1]}" for r in rows)
        return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]

    def cache_path(self, cid: str, version: str) -> Path:
        return self.cache / f"{cid}-{version}.pkl"

    def load_cached(self, cid: str, version: str):
        path = self.cache_path(cid, version)
        if not path.exists():
            return None
        try:
            with path.open("rb") as fh:
                return pickle.load(fh)
        except Exception:
            path.unlink(missing_ok=True)
            return None

    def save_cached(self, cid: str, version: str, graph) -> None:
        self.drop_cache(cid)
        path = self.cache_path(cid, version)
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(graph, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    def drop_cache(self, cid: str) -> None:
        for path in self.cache.glob(f"{cid}-*.pkl"):
            path.unlink(missing_ok=True)

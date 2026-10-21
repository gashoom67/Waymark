from __future__ import annotations

import logging
import os
import re
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from waymark import __version__
from waymark.config import Settings, get_settings
from waymark.graph.explore import entity_detail, overview, search_entities, subgraph
from waymark.ingest.pdf import render_page_png
from waymark.ingest.pipeline import SUPPORTED
from waymark.server.runtime import Runtime

log = logging.getLogger(__name__)
WEB_ROOT = Path(__file__).resolve().parent.parent / "web"
_unsafe = re.compile(r"[^\w .()\-\[\]]+", re.UNICODE)


class AskBody(BaseModel):
    question: str = Field(min_length=2, max_length=600)
    max_hops: int = Field(default=4, ge=1, le=6)


class CollectionBody(BaseModel):
    name: str = Field(min_length=1, max_length=80)


def _safe_name(name: str) -> str:
    base = os.path.basename(name or "upload")
    cleaned = _unsafe.sub("_", base).strip(" ._") or "upload"
    return cleaned[:160]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = Runtime(settings)
        app.state.runtime = runtime
        runtime.warm()
        yield
        runtime.shutdown()

    app = FastAPI(title="Waymark", version=__version__, lifespan=lifespan, docs_url="/api/docs", openapi_url="/api/openapi.json")

    def runtime_of(request: Request) -> Runtime:
        return request.app.state.runtime

    def require_collection(runtime: Runtime, cid: str) -> dict:
        found = runtime.library.collection(cid)
        if found is None:
            raise HTTPException(status_code=404, detail="collection not found")
        return found

    @app.exception_handler(ValueError)
    async def value_error(_: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/api/health")
    def health(request: Request):
        runtime = runtime_of(request)
        return {"ok": True, "version": __version__, "model": settings.spacy_model, "ocr": settings.ocr_enabled, "collections": len(runtime.library.collections())}

    @app.get("/api/collections")
    def list_collections(request: Request):
        return runtime_of(request).library.collections()

    @app.post("/api/collections", status_code=201)
    def make_collection(request: Request, body: CollectionBody):
        return runtime_of(request).library.create_collection(body.name)

    @app.patch("/api/collections/{cid}")
    def rename_collection(request: Request, cid: str, body: CollectionBody):
        runtime = runtime_of(request)
        require_collection(runtime, cid)
        runtime.library.rename_collection(cid, body.name)
        return runtime.library.collection(cid)

    @app.delete("/api/collections/{cid}", status_code=204)
    def remove_collection(request: Request, cid: str):
        runtime = runtime_of(request)
        require_collection(runtime, cid)
        if any(j.status in {"queued", "running"} for j in runtime.jobs(cid)):
            raise HTTPException(status_code=409, detail="wait for indexing to finish before deleting this collection")
        runtime.library.delete_collection(cid)
        runtime.invalidate(cid)
        runtime.library.ensure_default()
        return Response(status_code=204)

    @app.get("/api/collections/{cid}")
    def collection_detail(request: Request, cid: str):
        runtime = runtime_of(request)
        found = require_collection(runtime, cid)
        docs = [d.public() for d in runtime.library.documents(cid)]
        jobs = [j.public() for j in runtime.jobs(cid) if j.status in {"queued", "running"} or (j.finished and j.status == "failed")]
        return found | {"documents": docs, "jobs": jobs}

    @app.get("/api/collections/{cid}/stats")
    async def collection_stats(request: Request, cid: str):
        runtime = runtime_of(request)
        require_collection(runtime, cid)
        engine = await run_in_threadpool(runtime.engine, cid)
        if engine is None:
            return {"empty": True}
        return overview(engine.graph)

    @app.post("/api/collections/{cid}/documents", status_code=202)
    async def upload(request: Request, cid: str, files: list[UploadFile] = File(...)):
        runtime = runtime_of(request)
        require_collection(runtime, cid)
        limit = settings.max_upload_mb * 1024 * 1024
        accepted, rejected = [], []
        staging = settings.data_dir / "incoming"
        staging.mkdir(parents=True, exist_ok=True)
        for upload_file in files:
            name = _safe_name(upload_file.filename or "upload")
            suffix = Path(name).suffix.lower()
            if suffix not in SUPPORTED:
                rejected.append({"name": name, "reason": f"unsupported type {suffix or '(none)'}"})
                await upload_file.close()
                continue
            fd, tmp_name = tempfile.mkstemp(dir=staging, suffix=suffix)
            size = 0
            too_big = False
            with os.fdopen(fd, "wb") as out:
                while chunk := await upload_file.read(1 << 20):
                    size += len(chunk)
                    if size > limit:
                        too_big = True
                        break
                    out.write(chunk)
            await upload_file.close()
            tmp = Path(tmp_name)
            if too_big or size == 0:
                tmp.unlink(missing_ok=True)
                rejected.append({"name": name, "reason": "file is empty" if size == 0 else f"larger than {settings.max_upload_mb} MB"})
                continue
            job = await run_in_threadpool(runtime.submit, cid, name, suffix, tmp)
            if job is None:
                rejected.append({"name": name, "reason": "already in this collection"})
                continue
            accepted.append(job.public())
        if not accepted and rejected:
            raise HTTPException(status_code=400, detail={"rejected": rejected})
        return {"jobs": accepted, "rejected": rejected}

    @app.delete("/api/collections/{cid}/documents/{doc_id}", status_code=204)
    def delete_document(request: Request, cid: str, doc_id: str):
        runtime = runtime_of(request)
        doc = runtime.library.document(doc_id)
        if doc is None or doc.collection != cid:
            raise HTTPException(status_code=404, detail="document not found")
        if doc.status in {"queued", "running"}:
            raise HTTPException(status_code=409, detail="document is still being indexed")
        runtime.library.delete_document(doc_id)
        runtime.invalidate(cid)
        return Response(status_code=204)

    @app.get("/api/jobs/{job_id}")
    def job_status(request: Request, job_id: str):
        job = runtime_of(request).job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job.public()

    @app.get("/api/collections/{cid}/documents/{doc_id}/report")
    def document_report(request: Request, cid: str, doc_id: str):
        doc = runtime_of(request).library.document(doc_id)
        if doc is None or doc.collection != cid:
            raise HTTPException(status_code=404, detail="document not found")
        return doc.public() | {"report": doc.meta.get("report")}

    @app.get("/api/collections/{cid}/documents/{doc_id}/page/{page}")
    async def document_page(request: Request, cid: str, doc_id: str, page: int):
        runtime = runtime_of(request)
        doc = runtime.library.document(doc_id)
        if doc is None or doc.collection != cid:
            raise HTTPException(status_code=404, detail="document not found")
        path = runtime.library.file_path(doc)
        if not path.exists():
            raise HTTPException(status_code=404, detail="original file is missing")
        if doc.kind == "image":
            return FileResponse(path)
        if doc.kind != "pdf":
            raise HTTPException(status_code=415, detail="page previews exist for PDFs and images only")
        try:
            png = await run_in_threadpool(render_page_png, path, page)
        except IndexError:
            raise HTTPException(status_code=404, detail="page out of range")
        return Response(content=png, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})

    @app.post("/api/collections/{cid}/ask")
    async def ask(request: Request, cid: str, body: AskBody):
        runtime = runtime_of(request)
        require_collection(runtime, cid)
        engine = await run_in_threadpool(runtime.engine, cid)
        if engine is None:
            raise HTTPException(status_code=409, detail="this collection has no indexed documents yet")
        result = await run_in_threadpool(engine.ask, body.question, body.max_hops)
        names = {d.id: d.name for d in runtime.library.documents(cid)}
        for chain in result["chains"]:
            for hop in chain["hops"]:
                hop["sentence"]["doc_name"] = names.get(hop["sentence"]["doc"], hop["sentence"]["doc"])
        for passage in result["passages"]:
            passage["doc_name"] = names.get(passage["doc"], passage["doc"])
        result.pop("ranked_passages", None)
        focus = [h["to"]["id"] for c in result["chains"][:2] for h in c["hops"] if h.get("to")]
        focus += [h["from"]["id"] for c in result["chains"][:2] for h in c["hops"] if h.get("from")]
        result["graph"] = subgraph(engine.graph, list(dict.fromkeys(focus))[:14], limit=40, per_node=3)
        return result

    @app.get("/api/collections/{cid}/entities")
    async def entities(request: Request, cid: str, q: str = "", limit: int = 24, concepts: bool = True):
        runtime = runtime_of(request)
        require_collection(runtime, cid)
        engine = await run_in_threadpool(runtime.engine, cid)
        if engine is None:
            return []
        return search_entities(engine.graph, q[:120], limit=max(1, min(limit, 100)), include_concepts=concepts)

    @app.get("/api/collections/{cid}/entities/{eid}")
    async def entity(request: Request, cid: str, eid: int):
        runtime = runtime_of(request)
        require_collection(runtime, cid)
        engine = await run_in_threadpool(runtime.engine, cid)
        if engine is None or not 0 <= eid < engine.graph.n_entities:
            raise HTTPException(status_code=404, detail="entity not found")
        names = {d.id: d.name for d in runtime.library.documents(cid)}
        return entity_detail(engine.graph, eid, names)

    @app.get("/api/collections/{cid}/graph")
    async def graph_view(request: Request, cid: str, focus: str = "", limit: int = 60):
        runtime = runtime_of(request)
        require_collection(runtime, cid)
        engine = await run_in_threadpool(runtime.engine, cid)
        if engine is None:
            return {"nodes": [], "edges": []}
        ids = [int(x) for x in focus.split(",") if x.strip().isdigit()][:20]
        return subgraph(engine.graph, ids, limit=max(5, min(limit, 150)))

    if WEB_ROOT.exists():
        app.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(WEB_ROOT / "index.html", headers={"Cache-Control": "no-cache"})

    return app

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from waymark.config import Settings
from waymark.server.app import create_app


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    settings = Settings(data_dir=tmp_path_factory.mktemp("api"))
    with TestClient(create_app(settings)) as c:
        yield c


def _wait(client, cid, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        detail = client.get(f"/api/collections/{cid}").json()
        if detail["documents"] and all(d["status"] in {"ready", "failed"} for d in detail["documents"]) and not [j for j in detail["jobs"] if j["status"] in {"queued", "running"}]:
            return detail
        time.sleep(0.3)
    raise AssertionError("indexing did not finish")


def test_full_flow(client, corpus_files):
    assert client.get("/api/health").json()["ok"] is True
    cid = client.post("/api/collections", json={"name": "Desert papers"}).json()["id"]
    files = [("files", (p.name, p.read_bytes())) for p in corpus_files if p.name != "shifting_sands_survey.pdf"]
    files.append(("files", ("notes.exe", b"nope")))
    response = client.post(f"/api/collections/{cid}/documents", files=files)
    assert response.status_code == 202
    body = response.json()
    assert len(body["jobs"]) == len(corpus_files) - 1
    assert body["rejected"][0]["name"] == "notes.exe"
    detail = _wait(client, cid)
    assert all(d["status"] == "ready" for d in detail["documents"])
    again = client.post(f"/api/collections/{cid}/documents", files=[files[0]])
    assert again.status_code == 400
    assert "already" in again.json()["detail"]["rejected"][0]["reason"]

    answer = client.post(f"/api/collections/{cid}/ask", json={"question": "Who founded the company where Jordan Hale works?"}).json()
    assert answer["answer"]["text"] == "Rashid Al Suwaidi"
    assert answer["graph"]["nodes"]
    assert answer["chains"][0]["hops"][0]["sentence"]["doc_name"].endswith(".pdf")

    found = client.get(f"/api/collections/{cid}/entities", params={"q": "Chen"}).json()
    names = {e["name"] for e in found}
    assert {"Lin Chen", "Wei Chen"} <= names
    lin = next(e for e in found if e["name"] == "Lin Chen")
    detail_entity = client.get(f"/api/collections/{cid}/entities/{lin['id']}").json()
    assert any(s["name"] == "Wei Chen" for s in detail_entity["senses"])

    pdf = next(d for d in detail["documents"] if d["kind"] == "pdf")
    page = client.get(f"/api/collections/{cid}/documents/{pdf['id']}/page/1")
    assert page.status_code == 200 and page.headers["content-type"] == "image/png"
    assert client.get(f"/api/collections/{cid}/documents/{pdf['id']}/page/999").status_code == 404

    image = next(d for d in detail["documents"] if d["kind"] == "image")
    report = client.get(f"/api/collections/{cid}/documents/{image['id']}/report").json()
    assert "Mercury Labs" in report["report"]["text"]

    graph = client.get(f"/api/collections/{cid}/graph").json()
    assert graph["nodes"] and "edges" in graph

    assert client.delete(f"/api/collections/{cid}/documents/{pdf['id']}").status_code == 204
    remaining = client.get(f"/api/collections/{cid}").json()["documents"]
    assert pdf["id"] not in {d["id"] for d in remaining}


def test_bad_requests(client):
    assert client.post("/api/collections/nope/ask", json={"question": "hello there"}).status_code == 404
    cid = client.post("/api/collections", json={"name": "Empty"}).json()["id"]
    assert client.post(f"/api/collections/{cid}/ask", json={"question": "anything?"}).status_code == 409
    assert client.post(f"/api/collections/{cid}/ask", json={"question": ""}).status_code == 422
    assert client.get("/").status_code == 200

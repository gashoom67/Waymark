from __future__ import annotations

from collections import Counter

import numpy as np
from rapidfuzz import fuzz, process

from waymark.graph.index import KnowledgeGraph
from waymark.text import surface_key


def entity_card(graph: KnowledgeGraph, eid: int) -> dict:
    res = graph.res
    return {
        "id": int(eid),
        "name": res.entity_name[eid],
        "label": res.entity_label[eid],
        "class": res.entity_class[eid],
        "kind": res.entity_kind[eid],
        "qualifier": res.entity_qualifier[eid],
        "mentions": int(len(res.entity_mentions[eid])),
        "passages": int(graph.entity_df[eid]),
    }


def search_entities(graph: KnowledgeGraph, query: str, limit: int = 20, include_concepts: bool = True) -> list[dict]:
    res = graph.res
    key = surface_key(query)
    if not key:
        order = np.argsort(-graph.entity_df)
        out = []
        for eid in order:
            if not include_concepts and res.entity_kind[eid] == "concept":
                continue
            out.append(entity_card(graph, int(eid)))
            if len(out) >= limit:
                break
        return out
    exact = graph.alias_index.get(key, [])
    seen = set(exact)
    scored: list[tuple[float, int]] = [(200.0 + float(graph.entity_df[e]), e) for e in exact]
    if graph.name_keys is None:
        graph.name_keys = [surface_key(n) for n in res.entity_name]
    names = graph.name_keys
    for match, score, eid in process.extract(key, names, scorer=fuzz.WRatio, limit=limit * 3, score_cutoff=70):
        if eid in seen:
            continue
        seen.add(eid)
        scored.append((float(score) + np.log1p(graph.entity_df[eid]), eid))
    scored.sort(reverse=True)
    out = []
    for _, eid in scored:
        if not include_concepts and res.entity_kind[eid] == "concept":
            continue
        out.append(entity_card(graph, int(eid)))
        if len(out) >= limit:
            break
    return out


def entity_detail(graph: KnowledgeGraph, eid: int, doc_names: dict[str, str] | None = None) -> dict:
    res = graph.res
    corpus = graph.corpus
    card = entity_card(graph, eid)
    mids = res.entity_mentions[eid]
    surfaces = Counter(corpus.m_surface[m] for m in mids)
    samples = []
    seen_sents = set()
    for m in mids:
        sid = int(corpus.m_sent[m])
        if sid in seen_sents:
            continue
        seen_sents.add(sid)
        pid = int(corpus.sent_passage[sid])
        passage = corpus.passages[pid]
        samples.append({
            "sentence": corpus.sentence_text(sid),
            "doc": passage.doc_id,
            "doc_name": (doc_names or {}).get(passage.doc_id, passage.doc_id),
            "page": passage.page,
            "surface": corpus.m_surface[m],
        })
        if len(samples) >= 10:
            break
    nbrs, weights = graph.neighbours(eid)
    order = np.argsort(-weights)[:16]
    neighbours = [entity_card(graph, int(nbrs[i])) | {"weight": float(weights[i])} for i in order]
    siblings = set()
    for alias in res.entity_aliases[eid]:
        for other in graph.alias_index.get(alias, []):
            if other != eid:
                siblings.add(other)
    senses = [entity_card(graph, int(o)) for o in sorted(siblings, key=lambda o: -len(res.entity_mentions[o]))[:12]]
    docs = Counter(corpus.passage_doc[int(corpus.sent_passage[int(corpus.m_sent[m])])] for m in mids)
    return card | {
        "aliases": res.entity_aliases[eid],
        "surfaces": [{"text": t, "count": c} for t, c in surfaces.most_common(8)],
        "samples": samples,
        "neighbours": neighbours,
        "senses": senses,
        "documents": [{"id": d, "name": (doc_names or {}).get(d, d), "mentions": c} for d, c in docs.most_common(10)],
        "gender_hint": float(graph.entity_gender[eid]) if len(graph.entity_gender) > eid else 0.0,
    }


def subgraph(graph: KnowledgeGraph, focus: list[int], limit: int = 60, per_node: int = 6) -> dict:
    res = graph.res
    nodes: dict[int, float] = {}
    for eid in focus:
        if 0 <= eid < graph.n_entities:
            nodes[eid] = 10.0
    for eid in list(nodes):
        nbrs, weights = graph.neighbours(eid)
        if not len(nbrs):
            continue
        score = weights * np.asarray([graph.specificity(int(n)) for n in nbrs])
        for i in np.argsort(-score)[:per_node]:
            n = int(nbrs[i])
            if res.entity_class[n] == "NORP":
                continue
            nodes[n] = max(nodes.get(n, 0.0), float(score[i]))
            if len(nodes) >= limit:
                break
    if not focus:
        ranking = graph.entity_df * np.asarray([0.4 if k == "concept" else 1.0 for k in res.entity_kind])
        for eid in np.argsort(-ranking)[:limit]:
            nodes[int(eid)] = float(ranking[eid])
    ids = sorted(nodes, key=lambda e: -nodes[e])[:limit]
    index = set(ids)
    edges = []
    co = graph.cooccurrence
    for eid in ids:
        start, stop = co.indptr[eid], co.indptr[eid + 1]
        for n, w in zip(co.indices[start:stop], co.data[start:stop]):
            n = int(n)
            if n in index and n > eid:
                edges.append({"source": eid, "target": n, "weight": float(w)})
    return {
        "nodes": [entity_card(graph, e) | {"focus": e in focus, "score": nodes[e]} for e in ids],
        "edges": edges,
    }


def overview(graph: KnowledgeGraph) -> dict:
    res = graph.res
    classes = Counter(res.entity_class)
    return {
        "passages": graph.n_passages,
        "sentences": graph.corpus.n_sentences,
        "entities": graph.n_entities,
        "mentions": int(graph.corpus.n_mentions),
        "edges": int(graph.cooccurrence.nnz // 2),
        "classes": dict(classes),
        "resolver": res.stats,
        "build_seconds": round(graph.build_seconds, 2),
    }

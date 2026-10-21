from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from waymark.graph.index import KnowledgeGraph
from waymark.reasoning.question import Question


@dataclass
class Retrieval:
    passage_scores: np.ndarray
    entity_scores: np.ndarray
    ranked: np.ndarray
    lexical: np.ndarray
    walk: np.ndarray

    def top(self, k: int) -> list[int]:
        return [int(p) for p in self.ranked[:k]]


class Walker:
    def __init__(
        self,
        graph: KnowledgeGraph,
        damping: float = 0.5,
        cooccurrence_weight: float = 0.4,
        blend: float = 0.6,
        iterations: int = 30,
    ):
        self.graph = graph
        self.damping = damping
        self.blend = blend
        self.iterations = iterations
        n_e, n_p = graph.n_entities, graph.n_passages
        spec = np.asarray([graph.specificity(e) for e in range(n_e)], dtype=np.float32)
        ep = graph.entity_passage.copy().astype(np.float32)
        ep.data = np.log1p(ep.data)
        pe = sparse.csr_matrix(ep.T @ sparse.diags(spec))
        ee = graph.cooccurrence.copy().astype(np.float32)
        ee.data = np.log1p(ee.data) * cooccurrence_weight
        ee = sparse.csr_matrix(ee @ sparse.diags(spec))
        upper = sparse.hstack([ee, ep], format="csr")
        lower = sparse.hstack([pe, sparse.csr_matrix((n_p, n_p), dtype=np.float32)], format="csr")
        adj = sparse.vstack([upper, lower], format="csr")
        out = np.asarray(adj.sum(axis=1)).ravel()
        out[out == 0] = 1.0
        self.transition_t = sparse.csr_matrix((sparse.diags(1.0 / out) @ adj).T)
        self.n_e = n_e
        self.n_p = n_p
        self.spec = spec

    def pagerank(self, seed: np.ndarray) -> np.ndarray:
        total = seed.sum()
        if total <= 0:
            return np.zeros_like(seed)
        restart = seed / total
        rank = restart.copy()
        for _ in range(self.iterations):
            nxt = (1.0 - self.damping) * restart + self.damping * (self.transition_t @ rank)
            if np.abs(nxt - rank).sum() < 1e-7:
                rank = nxt
                break
            rank = nxt
        return rank

    def retrieve(self, question: Question, extra_terms: dict[int, float] | None = None) -> Retrieval:
        graph = self.graph
        terms = dict(question.terms)
        if extra_terms:
            for t, w in extra_terms.items():
                terms[t] = terms.get(t, 0.0) + w
        lexical = graph.passage_bm25.score(terms).astype(np.float64)
        seed = np.zeros(self.n_e + self.n_p, dtype=np.float64)
        anchors = question.anchors
        if anchors:
            share = 1.0 / len(anchors)
            for a in anchors:
                damp = 0.3 if a.soft else 1.0
                for eid, w in a.candidates[:3]:
                    seed[eid] += share * damp * w * (0.5 + float(self.spec[eid]))
        if lexical.max() > 0:
            top = np.argsort(-lexical)[:20]
            weights = lexical[top] / lexical[top].sum()
            mass = 0.35 if anchors else 1.0
            seed[self.n_e + top] += mass * weights
        rank = self.pagerank(seed)
        walk = rank[self.n_e:]
        ent = rank[: self.n_e]
        lex_n = lexical / lexical.max() if lexical.max() > 0 else lexical
        walk_n = walk / walk.max() if walk.max() > 0 else walk
        scores = self.blend * walk_n + (1.0 - self.blend) * lex_n
        ranked = np.argsort(-scores, kind="stable")
        return Retrieval(passage_scores=scores, entity_scores=ent, ranked=ranked, lexical=lex_n, walk=walk_n)

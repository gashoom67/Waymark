from __future__ import annotations

import heapq
import time
from dataclasses import dataclass

import numpy as np

from waymark.graph.index import KnowledgeGraph
from waymark.nlp.annotator import Annotator
from waymark.reasoning.chains import Candidate, Chain, ChainSearch, CueMatcher, Hop
from waymark.reasoning.lexicon import DEMONYMS
from waymark.reasoning.question import Question, QuestionReader
from waymark.reasoning.values import lifespan_years
from waymark.reasoning.walker import Retrieval, Walker
from waymark.text import surface_key

POOL_SIZE = 30


@dataclass
class Answer:
    text: str
    confidence: float
    kind: str
    entity: int | None = None


class ReasoningEngine:
    def __init__(self, graph: KnowledgeGraph, annotator: Annotator):
        self.graph = graph
        self.annotator = annotator
        self.reader = QuestionReader(graph, annotator)
        self.walker = Walker(graph)

    def ask(self, text: str, max_hops: int = 4) -> dict:
        started = time.perf_counter()
        question = self.reader.read(text)
        retrieval = self.walker.retrieve(question)
        pool = set(retrieval.top(POOL_SIZE))
        prior = retrieval.passage_scores
        matcher = CueMatcher(self.graph, self.annotator, question.cues)
        search = ChainSearch(self.graph, matcher, prior, pool)
        search.configure(question.plan, [w for w in question.tail if w not in {"same", "both"}], question.focus)
        search.present_tense = question.present
        if question.mode == "connect":
            answer, chains, candidates = self._connect(question, retrieval)
        elif question.mode == "choose":
            answer, chains, candidates = self._choose(question, search)
        elif question.mode in {"same", "both"}:
            answer, chains, candidates = self._compare(question, search)
        else:
            answer, chains, candidates = self._lookup(question, search, retrieval, max_hops)
        evidence = self._evidence(chains, retrieval)
        elapsed = (time.perf_counter() - started) * 1000.0
        return self._render(question, answer, chains, candidates, evidence, retrieval, elapsed)

    def _starts(self, question: Question, retrieval: Retrieval) -> list[tuple[int, float]]:
        starts: dict[int, float] = {}
        hard = [a for a in question.anchors if not a.soft]
        anchors = hard or question.anchors
        for a in anchors:
            for eid, w in a.candidates[:3]:
                weight = w * (0.4 if a.soft else 1.0)
                starts[eid] = max(starts.get(eid, 0.0), weight)
        if not starts:
            ent = retrieval.entity_scores
            top = np.argsort(-ent)[:6]
            peak = float(ent[top[0]]) if len(top) and ent[top[0]] > 0 else 1.0
            for eid in top:
                if ent[eid] > 0:
                    starts[int(eid)] = float(ent[eid]) / peak * 0.6
            for pid in retrieval.top(5):
                topic = int(self.graph.passage_topic[pid])
                if topic >= 0:
                    starts[topic] = max(starts.get(topic, 0.0), 0.5)
        return sorted(starts.items(), key=lambda kv: -kv[1])[:8]

    def _lookup(self, question: Question, search: ChainSearch, retrieval: Retrieval, max_hops: int):
        grounded = any(not a.soft for a in question.anchors)
        if not grounded and (not question.cues or float(retrieval.lexical.max(initial=0.0)) <= 0.0):
            return None, [], []
        if question.mode == "verify" and grounded:
            return self._verify(question, search)
        starts = self._starts(question, retrieval)
        search.prefer_countries = question.head in {"country", "nationality", "citizenship"}
        forbid = {e for a in question.anchors if not a.soft for e, _ in a.candidates}
        candidates = search.run(starts, question.expected, max_hops, forbid)
        if not candidates:
            return None, [], []
        best = candidates[0]
        if not grounded and search.has_tail and best.chain.coverage(search.weights) < 0.34:
            return None, [], []
        confidence = self._confidence(candidates)
        answer = Answer(text=best.text, confidence=confidence, kind=best.kind, entity=best.entity)
        return answer, [c.chain for c in candidates[:3]], candidates[:6]

    def _verify(self, question: Question, search: ChainSearch):
        anchors = [a for a in question.anchors if not a.soft]
        best_cov, best_chain = 0.0, None
        for anchor in anchors[:2]:
            eid = anchor.best
            for sid in search.sentences_for(eid):
                cov = search.matcher.sentence(sid)
                score = float(cov @ search.weights) if search.has_tail else 0.0
                others = {a.best for a in anchors if a is not anchor}
                if others and not others & set(search.sentence_entities(sid)):
                    score *= 0.6
                if score > best_cov:
                    best_cov = score
                    best_chain = Chain(score=score, entity=eid, covered=cov, hops=[Hop(sid, eid, None, score)])
        verdict = best_cov >= 0.7
        answer = Answer(text="yes" if verdict else "no", confidence=round(min(0.9, 0.45 + abs(best_cov - 0.7)), 3), kind="bool")
        return answer, [best_chain] if best_chain else [], []

    def _confidence(self, candidates: list[Candidate]) -> float:
        scores = np.asarray([c.score for c in candidates[:6]], dtype=np.float64)
        shifted = np.exp((scores - scores.max()) * 2.0)
        return float(shifted[0] / shifted.sum())

    def _option_values(self, question: Question, search: ChainSearch, expected: frozenset[str]):
        results = []
        option_ids = {e for opt in question.options for e, _ in opt.candidates}
        for opt in question.options:
            starts = [(e, w) for e, w in opt.candidates[:2]]
            hops = len(search.plan) + 1
            cands = search.run(starts, expected, hops, option_ids)
            results.append((opt, cands))
        return results

    def _choose(self, question: Question, search: ChainSearch):
        direction = question.direction
        if direction == "lifespan":
            spans = []
            for opt in question.options:
                years = None
                for pid in self.graph.entity_homes[opt.best][:2]:
                    years = lifespan_years(self.graph.corpus.passages[int(pid)].text)
                    if years is not None:
                        break
                spans.append((opt, years))
            known = [(o, y) for o, y in spans if y is not None]
            if known:
                winner = max(known, key=lambda x: x[1])[0]
                conf = 0.85 if len(known) == len(spans) else 0.55
                return Answer(text=self.graph.res.entity_name[winner.best], confidence=conf, kind="entity", entity=winner.best), [], []
            direction = "late"
        expected = frozenset({"DATE"}) if direction in {"early", "late", None} else frozenset({"NUMBER"})
        per_option = self._option_values(question, search, expected)
        scored = []
        chains: list[Chain] = []
        for opt, cands in per_option:
            usable = [c for c in cands if c.value is not None]
            if usable:
                scored.append((opt, usable[0]))
                chains.append(usable[0].chain)
        if not scored:
            fallback = question.options[0]
            return Answer(text=self.graph.res.entity_name[fallback.best], confidence=0.2, kind="entity", entity=fallback.best), [], []
        if len(scored) == 1:
            opt, cand = scored[0]
            return Answer(text=self.graph.res.entity_name[opt.best], confidence=0.3, kind="entity", entity=opt.best), chains, [cand]
        reverse = direction in {"late", "more"}
        ordered = sorted(scored, key=lambda x: x[1].value, reverse=reverse)
        winner = ordered[0][0]
        tie = ordered[0][1].value == ordered[1][1].value
        confidence = 0.4 if tie else min(0.95, 0.55 + 0.1 * min(ordered[0][1].score, ordered[1][1].score))
        answer = Answer(text=self.graph.res.entity_name[winner.best], confidence=max(0.3, confidence), kind="entity", entity=winner.best)
        return answer, chains, [c for _, c in scored]

    def _compare(self, question: Question, search: ChainSearch):
        if question.mode == "same":
            head = None
            for word in ("nationality", "country", "citizenship", "city", "language", "genre", "religion"):
                if word in question.cues:
                    head = word
                    break
            expected = frozenset({"NORP", "PLACE"}) if head in {"nationality", "citizenship", "country", None} else frozenset({"PLACE"}) if head == "city" else frozenset()
            search.prefer_countries = head in {"nationality", "citizenship", "country", None}
            per_option = self._option_values(question, search, expected)
            picks = []
            chains = []
            for opt, cands in per_option:
                if cands:
                    picks.append(self._normalise_place(cands[0].text))
                    chains.append(cands[0].chain)
            if len(picks) < 2:
                return Answer(text="no", confidence=0.3, kind="bool"), chains, []
            same = picks[0] == picks[1]
            return Answer(text="yes" if same else "no", confidence=0.75, kind="bool"), chains, []
        hits = 0
        chains = []
        for opt in question.options:
            best = 0.0
            best_chain = None
            for pid in self.graph.entity_homes[opt.best][:2].tolist():
                for sid in self.graph.corpus.passage_sents[pid]:
                    cov = float(search.matcher.sentence(sid) @ search.weights) if len(search.weights) else 0.0
                    if cov > best:
                        best = cov
                        best_chain = Chain(score=cov, entity=opt.best, covered=search.matcher.sentence(sid), hops=[])
            if best >= 0.6:
                hits += 1
            if best_chain is not None:
                chains.append(best_chain)
        answer = Answer(text="yes" if hits == len(question.options) else "no", confidence=0.6, kind="bool")
        return answer, chains, []

    def _normalise_place(self, text: str) -> str:
        key = surface_key(text)
        return DEMONYMS.get(key, key)

    def _connect(self, question: Question, retrieval: Retrieval):
        anchors = [a for a in question.anchors if not a.soft][:2]
        src, dst = anchors[0].best, anchors[1].best
        paths = self.connections(src, dst, retrieval.entity_scores)
        chains = []
        for score, nodes in paths:
            hops = [Hop(self._shared_sentence(a, b, retrieval), a, b, 0.0) for a, b in zip(nodes, nodes[1:])]
            chains.append(Chain(score=score, entity=dst, covered=np.zeros(0), hops=hops))
        if not chains:
            return Answer(text="No connecting path found", confidence=0.0, kind="none"), [], []
        names = " → ".join(self.graph.res.entity_name[e] for e in paths[0][1])
        return Answer(text=names, confidence=min(0.95, 0.5 + 0.1 * len(paths)), kind="path"), chains, []

    def connections(self, src: int, dst: int, entity_scores: np.ndarray, max_depth: int = 4, limit: int = 3) -> list[tuple[float, list[int]]]:
        graph = self.graph
        heap: list[tuple[float, list[int]]] = [(0.0, [src])]
        found: list[tuple[float, list[int]]] = []
        expanded = 0
        best_cost: dict[tuple[int, int], float] = {}
        while heap and len(found) < limit and expanded < 20000:
            cost, path = heapq.heappop(heap)
            node = path[-1]
            if node == dst:
                if not any(set(p[1:-1]) == set(path[1:-1]) for _, p in found):
                    found.append((1.0 / (1.0 + cost), path))
                continue
            if len(path) > max_depth:
                continue
            expanded += 1
            nbrs, weights = graph.neighbours(node)
            if len(nbrs) > 400:
                order = np.argsort(-weights)[:400]
                nbrs, weights = nbrs[order], weights[order]
            for nxt, w in zip(nbrs.tolist(), weights.tolist()):
                if nxt in path:
                    continue
                if nxt != dst and graph.res.entity_class[nxt] == "NORP":
                    continue
                step = 1.0 / np.log1p(w + 1.0) + 0.6 * (1.0 - graph.specificity(nxt)) - 0.2 * float(entity_scores[nxt] / (entity_scores.max() or 1.0))
                new_cost = cost + max(step, 0.05)
                sig = (nxt, len(path))
                if best_cost.get(sig, 1e9) <= new_cost - 1e-9 and nxt != dst:
                    continue
                best_cost[sig] = new_cost
                heapq.heappush(heap, (new_cost, path + [nxt]))
        return found

    def _shared_sentence(self, a: int, b: int, retrieval: Retrieval) -> int:
        graph = self.graph
        sa = set(graph.entity_sentences(a).tolist())
        shared = [s for s in graph.entity_sentences(b).tolist() if s in sa]
        if not shared:
            return int(graph.entity_sentences(a)[0])
        return max(shared, key=lambda s: retrieval.passage_scores[int(graph.corpus.sent_passage[s])])

    def _evidence(self, chains: list[Chain], retrieval: Retrieval) -> list[int]:
        ordered: list[int] = []
        for chain in chains:
            for hop in chain.hops:
                pid = int(self.graph.corpus.sent_passage[hop.sentence])
                if pid not in ordered:
                    ordered.append(pid)
        for pid in retrieval.top(10):
            if pid not in ordered:
                ordered.append(pid)
        return ordered

    def _entity_ref(self, eid: int | None) -> dict | None:
        if eid is None or eid < 0:
            return None
        res = self.graph.res
        return {"id": int(eid), "name": res.entity_name[eid], "type": res.entity_label[eid], "class": res.entity_class[eid]}

    def _sentence_ref(self, sid: int) -> dict:
        corpus = self.graph.corpus
        pid = int(corpus.sent_passage[sid])
        passage = corpus.passages[pid]
        return {"id": int(sid), "text": corpus.sentence_text(sid), "passage": pid, "doc": passage.doc_id, "page": passage.page, "origin": passage.origin}

    def _render(self, question, answer, chains, candidates, evidence, retrieval, elapsed) -> dict:
        corpus = self.graph.corpus
        rendered_chains = []
        for chain in chains:
            rendered_chains.append({
                "score": round(float(chain.score), 4),
                "hops": [
                    {"from": self._entity_ref(h.source), "to": self._entity_ref(h.target), "sentence": self._sentence_ref(h.sentence)}
                    for h in chain.hops
                ],
            })
        return {
            "question": question.text,
            "mode": question.mode,
            "expects": sorted(question.expected),
            "answer": None if answer is None else {
                "text": answer.text,
                "confidence": round(float(answer.confidence), 3),
                "kind": answer.kind,
                "entity": self._entity_ref(answer.entity),
            },
            "alternatives": [
                {"text": c.text, "score": round(float(c.score), 3), "kind": c.kind} for c in candidates[1:5]
            ],
            "anchors": [
                {"text": a.text, "soft": a.soft, "entity": self._entity_ref(a.best), "options": [self._entity_ref(e) | {"weight": round(w, 3)} for e, w in a.candidates[:3]]}
                for a in question.anchors
            ],
            "cues": list(question.cues.keys()),
            "chains": rendered_chains,
            "passages": [
                {
                    "id": pid,
                    "doc": corpus.passages[pid].doc_id,
                    "page": corpus.passages[pid].page,
                    "text": corpus.passages[pid].text,
                    "score": round(float(retrieval.passage_scores[pid]), 4),
                }
                for pid in evidence[:8]
            ],
            "ranked_passages": evidence,
            "elapsed_ms": round(elapsed, 1),
        }

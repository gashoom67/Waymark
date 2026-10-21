from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from waymark.graph.anaphora import resolve_anaphora
from waymark.graph.corpus import Corpus
from waymark.graph.resolver import EntityResolver, Resolution
from waymark.nlp.annotator import PassageNote
from waymark.text import name_core

log = logging.getLogger(__name__)

_male = re.compile(r"\b(?:he|him|his|himself)\b", re.IGNORECASE)
_female = re.compile(r"\b(?:she|her|hers|herself)\b", re.IGNORECASE)


@dataclass
class Bm25:
    weights: sparse.csc_matrix
    idf: np.ndarray

    @classmethod
    def fit(cls, rows: list[np.ndarray], vocab_size: int, k1: float = 1.2, b: float = 0.75) -> "Bm25":
        indptr = [0]
        indices: list[int] = []
        for terms in rows:
            indices.extend(int(t) for t in terms)
            indptr.append(len(indices))
        tf = sparse.csr_matrix(
            (np.ones(len(indices), dtype=np.float32), np.asarray(indices, dtype=np.int64), np.asarray(indptr)),
            shape=(len(rows), max(1, vocab_size)),
        )
        tf.sum_duplicates()
        n_docs = max(1, len(rows))
        lengths = np.asarray(tf.sum(axis=1)).ravel()
        avg = float(lengths.mean()) if len(lengths) else 1.0
        df = np.bincount(tf.indices, minlength=tf.shape[1]).astype(np.float32)
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        coo = tf.tocoo()
        denom = coo.data + k1 * (1.0 - b + b * lengths[coo.row] / max(avg, 1e-6))
        data = coo.data * (k1 + 1.0) / denom * idf[coo.col]
        weights = sparse.csc_matrix((data.astype(np.float32), (coo.row, coo.col)), shape=tf.shape)
        return cls(weights=weights, idf=idf)

    def score(self, term_weights: dict[int, float]) -> np.ndarray:
        if not term_weights:
            return np.zeros(self.weights.shape[0], dtype=np.float32)
        cols = np.asarray(list(term_weights.keys()), dtype=np.int64)
        mult = np.asarray(list(term_weights.values()), dtype=np.float32)
        sub = self.weights[:, cols]
        return np.asarray(sub @ mult).ravel()


@dataclass
class KnowledgeGraph:
    corpus: Corpus
    res: Resolution
    entity_passage: sparse.csr_matrix
    passage_entity: sparse.csr_matrix
    entity_sentence: sparse.csr_matrix
    sentence_entity: sparse.csr_matrix
    cooccurrence: sparse.csr_matrix
    passage_bm25: Bm25
    sentence_bm25: Bm25
    entity_df: np.ndarray
    alias_index: dict[str, list[int]]
    token_index: dict[str, list[int]]
    max_alias_len: int
    passage_topic: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    entity_homes: list[np.ndarray] = field(default_factory=list)
    entity_gender: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    name_keys: list[str] | None = None
    build_seconds: float = 0.0

    @property
    def n_entities(self) -> int:
        return len(self.res.entity_name)

    @property
    def n_passages(self) -> int:
        return len(self.corpus.passages)

    def entity_sentences(self, eid: int) -> np.ndarray:
        row = self.entity_sentence.getrow(eid)
        return row.indices

    def sentence_entities(self, sid: int) -> np.ndarray:
        return self.sentence_entity.indices[self.sentence_entity.indptr[sid]:self.sentence_entity.indptr[sid + 1]]

    def specificity(self, eid: int) -> float:
        return float(1.0 / np.log(np.e + self.entity_df[eid]))

    def neighbours(self, eid: int) -> tuple[np.ndarray, np.ndarray]:
        start, stop = self.cooccurrence.indptr[eid], self.cooccurrence.indptr[eid + 1]
        return self.cooccurrence.indices[start:stop], self.cooccurrence.data[start:stop]


def build_graph(passages: list[PassageNote], anaphora: bool = True) -> KnowledgeGraph:
    started = time.perf_counter()
    corpus = Corpus.build(passages)
    res = EntityResolver(corpus).resolve()
    n_ent = len(res.entity_name)
    n_sent = corpus.n_sentences
    n_pass = len(corpus.passages)

    ent = res.mention_entity
    valid = ent >= 0
    links = resolve_anaphora(corpus, res) if anaphora else []
    res.stats["anaphora_links"] = len(links)
    m_sent = np.concatenate([corpus.m_sent[valid], np.asarray([s for s, _ in links], dtype=np.int32)])
    m_ent = np.concatenate([ent[valid], np.asarray([e for _, e in links], dtype=np.int64)])
    weights = np.concatenate([np.ones(int(valid.sum()), dtype=np.float32), np.full(len(links), 0.8, dtype=np.float32)])
    sent_ent = sparse.csr_matrix(
        (weights, (m_sent, m_ent)),
        shape=(n_sent, n_ent),
    )
    sent_ent.sum_duplicates()
    ent_sent = sparse.csr_matrix(sent_ent.T)
    sent_pass = sparse.csr_matrix(
        (np.ones(n_sent, dtype=np.float32), (np.arange(n_sent), corpus.sent_passage)),
        shape=(n_sent, n_pass),
    )
    ent_pass = sparse.csr_matrix(ent_sent @ sent_pass)
    pass_ent = sparse.csr_matrix(ent_pass.T)
    binary = sent_ent.copy()
    binary.data[:] = 1.0
    co = sparse.csr_matrix(binary.T @ binary)
    co.setdiag(0)
    co.eliminate_zeros()
    df = np.diff(sparse.csr_matrix(ent_pass > 0).indptr).astype(np.float32)

    passage_terms = []
    for span in corpus.passage_sents:
        if len(span):
            passage_terms.append(np.concatenate([corpus.sent_terms[s] for s in span]))
        else:
            passage_terms.append(np.zeros(0, dtype=np.int32))
    vocab_size = len(corpus.lemmas)
    passage_bm25 = Bm25.fit(passage_terms, vocab_size)
    sentence_bm25 = Bm25.fit(corpus.sent_terms, vocab_size, k1=1.0, b=0.5)

    alias_index: dict[str, list[int]] = defaultdict(list)
    token_index: dict[str, list[int]] = defaultdict(list)
    max_len = 1
    for eid, aliases in enumerate(res.entity_aliases):
        for alias in aliases:
            alias_index[alias].append(eid)
            max_len = max(max_len, len(alias.split()))
        if res.entity_kind[eid] == "named":
            for tok in set(t for alias in aliases for t in name_core(alias)):
                if len(tok) > 2:
                    token_index[tok].append(eid)
    topic, homes, gender = _topics(corpus, res, ent_pass)
    graph = KnowledgeGraph(
        corpus=corpus,
        res=res,
        entity_passage=ent_pass,
        passage_entity=pass_ent,
        entity_sentence=ent_sent,
        sentence_entity=sent_ent,
        cooccurrence=co,
        passage_bm25=passage_bm25,
        sentence_bm25=sentence_bm25,
        entity_df=df,
        alias_index=dict(alias_index),
        token_index=dict(token_index),
        max_alias_len=min(max_len, 12),
        passage_topic=topic,
        entity_homes=homes,
        entity_gender=gender,
    )
    graph.build_seconds = time.perf_counter() - started
    log.info("graph built: %d passages, %d entities in %.1fs", n_pass, n_ent, graph.build_seconds)
    return graph


def _topics(corpus: Corpus, res: Resolution, ent_pass: sparse.csr_matrix) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    n_pass = len(corpus.passages)
    n_ent = len(res.entity_name)
    topic = np.full(n_pass, -1, dtype=np.int64)
    for mid in range(corpus.n_mentions):
        if corpus.m_kind[mid] == "title":
            topic[corpus.mention_passage(mid)] = res.mention_entity[mid]
    pass_ent = sparse.csr_matrix(ent_pass.T)
    for pid in range(n_pass):
        if topic[pid] >= 0:
            continue
        start, stop = pass_ent.indptr[pid], pass_ent.indptr[pid + 1]
        if stop == start:
            continue
        ents = pass_ent.indices[start:stop]
        counts = pass_ent.data[start:stop]
        named = [(c, -i, e) for i, (e, c) in enumerate(zip(ents, counts)) if res.entity_kind[e] == "named" and res.entity_class[e] != "NORP"]
        if named:
            best = max(named)
            if best[0] >= 2:
                topic[pid] = best[2]
                continue
        span = corpus.passage_sents[pid]
        if len(span):
            for m in corpus.sent_mentions[span.start]:
                eid = int(res.mention_entity[m])
                if eid >= 0 and res.entity_kind[eid] == "named" and res.entity_class[eid] != "NORP":
                    topic[pid] = eid
                    break
    homes: list[list[tuple[float, int]]] = [[] for _ in range(n_ent)]
    for pid, eid in enumerate(topic):
        if eid >= 0:
            homes[eid].append((1e6, pid))
    for eid in range(n_ent):
        start, stop = ent_pass.indptr[eid], ent_pass.indptr[eid + 1]
        if stop - start > 60:
            order = np.argsort(-ent_pass.data[start:stop])[:60]
            pairs = zip(ent_pass.indices[start:stop][order], ent_pass.data[start:stop][order])
        else:
            pairs = zip(ent_pass.indices[start:stop], ent_pass.data[start:stop])
        homes[eid].extend((float(c), int(p)) for p, c in pairs if c >= 2)
    home_arrays = []
    for rows in homes:
        seen, ordered = set(), []
        for _, pid in sorted(rows, reverse=True):
            if pid not in seen:
                seen.add(pid)
                ordered.append(pid)
            if len(ordered) >= 6:
                break
        home_arrays.append(np.asarray(ordered, dtype=np.int64))
    gender = np.zeros(n_ent, dtype=np.float32)
    for eid in range(n_ent):
        if res.entity_class[eid] not in {"PERSON", "MISC"}:
            continue
        male = female = 0
        for pid in home_arrays[eid][:3]:
            text = corpus.passages[int(pid)].text
            male += len(_male.findall(text))
            female += len(_female.findall(text))
        if male + female >= 2:
            gender[eid] = (male - female) / (male + female)
    return topic, home_arrays, gender

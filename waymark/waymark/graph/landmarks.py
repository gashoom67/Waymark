from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from waymark.graph.corpus import Corpus
from waymark.text import name_core


@dataclass
class ContextField:
    sentence_vectors: sparse.csr_matrix
    landmark_keys: dict[str, int]
    term_idf: np.ndarray

    def mention_rows(self, corpus: Corpus, mention_ids: list[int] | np.ndarray) -> sparse.csr_matrix:
        sents = corpus.m_sent[np.asarray(mention_ids, dtype=np.int64)]
        return self.sentence_vectors[sents]

    def centroid(self, corpus: Corpus, mention_ids: list[int] | np.ndarray) -> sparse.csr_matrix:
        rows = self.mention_rows(corpus, mention_ids)
        summed = sparse.csr_matrix(rows.sum(axis=0))
        return row_normalize(summed)


def row_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sparse.csr_matrix(sparse.diags(1.0 / norms) @ matrix)


def is_landmark_key(corpus: Corpus, mid: int) -> bool:
    kind = corpus.m_kind[mid]
    if kind == "title":
        return True
    if kind == "concept":
        return False
    return len(name_core(corpus.m_key[mid])) >= 2


def window_operator(corpus: Corpus, radius: int = 1, falloff: float = 0.5) -> sparse.csr_matrix:
    n = corpus.n_sentences
    rows, cols, vals = [], [], []
    for sid in range(n):
        for nid in corpus.neighbours(sid, radius):
            rows.append(sid)
            cols.append(nid)
            vals.append(1.0 if nid == sid else falloff ** abs(nid - sid))
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)


def build_context_field(corpus: Corpus, landmark_weight: float = 1.6) -> ContextField:
    n_sent = corpus.n_sentences
    vocab_size = max(1, len(corpus.lemmas))
    indptr = [0]
    indices: list[int] = []
    for terms in corpus.sent_terms:
        indices.extend(int(t) for t in terms)
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.float32)
    tf = sparse.csr_matrix((data, np.asarray(indices, dtype=np.int64), np.asarray(indptr)), shape=(n_sent, vocab_size))
    tf.sum_duplicates()
    df = np.bincount(tf.indices, minlength=vocab_size).astype(np.float32)
    idf = np.log((n_sent + 1.0) / (df + 1.0)) + 1.0
    tf.data = np.log1p(tf.data)
    weighted = row_normalize(sparse.csr_matrix(tf @ sparse.diags(idf)))

    landmark_counts: Counter[str] = Counter()
    for mid in range(corpus.n_mentions):
        if is_landmark_key(corpus, mid):
            landmark_counts[corpus.m_key[mid]] += 1
    landmark_keys = {key: i for i, key in enumerate(sorted(landmark_counts))}
    n_land = max(1, len(landmark_keys))
    l_rows, l_cols = [], []
    for mid in range(corpus.n_mentions):
        col = landmark_keys.get(corpus.m_key[mid])
        if col is not None:
            l_rows.append(int(corpus.m_sent[mid]))
            l_cols.append(col)
    land = sparse.csr_matrix((np.ones(len(l_rows), dtype=np.float32), (l_rows, l_cols)), shape=(n_sent, n_land))
    land.sum_duplicates()
    land.data[:] = 1.0
    land_df = np.bincount(land.indices, minlength=n_land).astype(np.float32)
    land_idf = np.log((n_sent + 1.0) / (land_df + 1.0)) + 1.0
    land = row_normalize(sparse.csr_matrix(land @ sparse.diags(land_idf)))

    window = window_operator(corpus)
    stacked = sparse.hstack([window @ weighted, landmark_weight * (window @ land)], format="csr")
    return ContextField(sentence_vectors=row_normalize(stacked), landmark_keys=landmark_keys, term_idf=idf)

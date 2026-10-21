from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from waymark.nlp.annotator import PassageNote
from waymark.text import surface_key

COARSE = {
    "PERSON": "PERSON",
    "GPE": "PLACE",
    "LOC": "PLACE",
    "FAC": "PLACE",
    "ORG": "ORG",
    "NORP": "NORP",
    "LANGUAGE": "NORP",
    "PRODUCT": "WORK",
    "WORK_OF_ART": "WORK",
    "LAW": "WORK",
    "EVENT": "EVENT",
    "MISC": "MISC",
    "TITLE": "MISC",
    "CONCEPT": "CONCEPT",
}


_qualifier = re.compile(r"\s*\(([^)]*)\)\s*")


def split_title(title: str) -> tuple[str, str]:
    qualifiers = [q.strip() for q in _qualifier.findall(title)]
    bare = _qualifier.sub(" ", title).strip()
    return bare or title, " ".join(qualifiers)


def coarse(label: str) -> str:
    return COARSE.get(label, "MISC")


def compatible(a: str, b: str) -> bool:
    if a == b or a == "MISC" or b == "MISC":
        return True
    loose = {frozenset({"ORG", "PLACE"}), frozenset({"WORK", "EVENT"}), frozenset({"ORG", "WORK"})}
    return frozenset({a, b}) in loose


@dataclass
class Corpus:
    passages: list[PassageNote]
    vocab: dict[str, int] = field(default_factory=dict)
    lemmas: list[str] = field(default_factory=list)
    sent_passage: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    sent_order: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    sent_terms: list[np.ndarray] = field(default_factory=list)
    passage_sents: list[range] = field(default_factory=list)
    passage_doc: list[str] = field(default_factory=list)
    m_sent: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    m_key: list[str] = field(default_factory=list)
    m_label: list[str] = field(default_factory=list)
    m_class: list[str] = field(default_factory=list)
    m_kind: list[str] = field(default_factory=list)
    m_surface: list[str] = field(default_factory=list)
    m_span: list[tuple[int, int]] = field(default_factory=list)
    m_qualifier: list[str] = field(default_factory=list)
    m_modifier: list[bool] = field(default_factory=list)
    sent_mentions: list[list[int]] = field(default_factory=list)
    sent_values: list[list[list]] = field(default_factory=list)

    @classmethod
    def build(cls, passages: list[PassageNote]) -> "Corpus":
        corpus = cls(passages=passages)
        corpus._index()
        return corpus

    def term_id(self, lemma: str) -> int:
        idx = self.vocab.get(lemma)
        if idx is None:
            idx = len(self.lemmas)
            self.vocab[lemma] = idx
            self.lemmas.append(lemma)
        return idx

    def _index(self) -> None:
        sent_passage: list[int] = []
        sent_order: list[int] = []
        m_sent: list[int] = []
        for pid, passage in enumerate(self.passages):
            self.passage_doc.append(passage.doc_id)
            first = len(sent_passage)
            bare_title, qualifier = split_title(passage.title) if passage.title else ("", "")
            title_key = surface_key(bare_title) if bare_title else ""
            for order, sent in enumerate(passage.sentences):
                sid = len(sent_passage)
                sent_passage.append(pid)
                sent_order.append(order)
                terms = [self.term_id(lem) for lem in sent.lemmas]
                if order == 0 and passage.title:
                    terms.extend(self.term_id(lem) for lem in surface_key(passage.title).split() if len(lem) > 1)
                self.sent_terms.append(np.asarray(terms, dtype=np.int32))
                ids: list[int] = []
                if order == 0 and title_key:
                    ids.append(len(m_sent))
                    m_sent.append(sid)
                    self.m_key.append(title_key)
                    self.m_label.append("TITLE")
                    self.m_class.append("MISC")
                    self.m_kind.append("title")
                    self.m_surface.append(bare_title)
                    self.m_span.append((-1, -1))
                    self.m_qualifier.append(surface_key(qualifier))
                    self.m_modifier.append(False)
                for mention in sent.mentions:
                    if mention.kind == "title":
                        continue
                    ids.append(len(m_sent))
                    m_sent.append(sid)
                    self.m_key.append(mention.key)
                    self.m_label.append(mention.label)
                    self.m_class.append(coarse(mention.label))
                    self.m_kind.append(mention.kind)
                    self.m_surface.append(mention.surface)
                    self.m_span.append((mention.start, mention.end))
                    self.m_qualifier.append("")
                    self.m_modifier.append(mention.modifier)
                self.sent_mentions.append(ids)
                self.sent_values.append(sent.values)
            self.passage_sents.append(range(first, len(sent_passage)))
        self.sent_passage = np.asarray(sent_passage, dtype=np.int32)
        self.sent_order = np.asarray(sent_order, dtype=np.int32)
        self.m_sent = np.asarray(m_sent, dtype=np.int32)

    @property
    def n_sentences(self) -> int:
        return len(self.sent_terms)

    @property
    def n_mentions(self) -> int:
        return len(self.m_key)

    def sentence_text(self, sid: int) -> str:
        passage = self.passages[int(self.sent_passage[sid])]
        sent = passage.sentences[int(self.sent_order[sid])]
        return passage.text[sent.start:sent.end].strip()

    def mention_passage(self, mid: int) -> int:
        return int(self.sent_passage[self.m_sent[mid]])

    def mention_doc(self, mid: int) -> str:
        return self.passage_doc[self.mention_passage(mid)]

    def neighbours(self, sid: int, radius: int = 1) -> range:
        pid = int(self.sent_passage[sid])
        span = self.passage_sents[pid]
        lo = max(span.start, sid - radius)
        hi = min(span.stop, sid + radius + 1)
        return range(lo, hi)

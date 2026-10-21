from __future__ import annotations

import re

from waymark.graph.corpus import Corpus
from waymark.graph.resolver import Resolution
from waymark.text import tokens_of

DEFINITE_HEADS = frozenset("""
institute station laboratory lab company firm university college survey report study paper bank ministry museum city
town film book album band team club organisation organization agency centre center project programme program school
hospital refinery plant group foundation council committee court government office bureau department river island
village region country mosque airport port fund institution archive library journal newspaper magazine festival
treaty agreement law act party movement army navy dynasty kingdom empire republic province state
""".split())
ORG_HEADS = frozenset({"company", "firm", "organisation", "organization", "group", "agency", "institution", "bank"})

_person_pronoun = re.compile(r"^\W*(he|she|his|her|him)\b", re.IGNORECASE)
_thing_pronoun = re.compile(r"^\W*(it|its)\b", re.IGNORECASE)
_definite = re.compile(r"^\W*(?:the|this|that)\s+([a-z]+)\b", re.IGNORECASE)
_inner_definite = re.compile(r"\b(?:the|this|that)\s+([a-z]+)\b")


def resolve_anaphora(corpus: Corpus, res: Resolution) -> list[tuple[int, int]]:
    links: list[tuple[int, int]] = []
    ent = res.mention_entity
    last_doc = None
    history: list[list[int]] = []
    for pid, passage in enumerate(corpus.passages):
        if passage.doc_id != last_doc:
            history = []
            last_doc = passage.doc_id
        for sid in corpus.passage_sents[pid]:
            sentence = corpus.sentence_text(sid)
            present = []
            for m in corpus.sent_mentions[sid]:
                eid = int(ent[m])
                if eid >= 0 and res.entity_kind[eid] == "named" and res.entity_class[eid] != "NORP" and eid not in present:
                    present.append(eid)
            target = _antecedent(sentence, history, res)
            if target is not None and target not in present:
                links.append((sid, target))
                present.insert(0, target)
            for match in _inner_definite.finditer(sentence.lower()):
                head = match.group(1)
                if head in DEFINITE_HEADS:
                    found = _by_head(head, history, res)
                    if found is not None and found not in present:
                        links.append((sid, found))
                        present.append(found)
            history.append(present)
            if len(history) > 14:
                history.pop(0)
    return links


def _antecedent(sentence: str, history: list[list[int]], res: Resolution) -> int | None:
    if not history:
        return None
    if _person_pronoun.match(sentence):
        return _recent(history, res, lambda cls: cls == "PERSON", depth=3)
    if _thing_pronoun.match(sentence):
        return _recent(history, res, lambda cls: cls not in {"PERSON", "CONCEPT", "NORP"}, depth=2)
    match = _definite.match(sentence)
    if match:
        head = match.group(1).lower()
        if head in DEFINITE_HEADS:
            return _by_head(head, history, res)
    return None


def _recent(history: list[list[int]], res: Resolution, accept, depth: int) -> int | None:
    for present in reversed(history[-depth:]):
        for eid in present:
            if accept(res.entity_class[eid]):
                return eid
    return None


def _by_head(head: str, history: list[list[int]], res: Resolution) -> int | None:
    stem = head[:-1] if head.endswith("s") else head
    for present in reversed(history):
        for eid in present:
            words = tokens_of(res.entity_name[eid])
            if stem in words or head in words or (stem == "lab" and "laboratory" in words) or (stem == "laboratory" and "lab" in words):
                return eid
            if head in ORG_HEADS and res.entity_class[eid] == "ORG":
                return eid
    return None

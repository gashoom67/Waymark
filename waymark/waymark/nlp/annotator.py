from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span

from waymark.text import STOPWORDS, squash, surface_key

log = logging.getLogger(__name__)

NAMED_LABELS = frozenset({"PERSON", "ORG", "GPE", "LOC", "FAC", "NORP", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"})
VALUE_LABELS = frozenset({"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "CARDINAL", "ORDINAL"})
CALENDAR = frozenset({
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov",
    "dec", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
})
WEAK_MODIFIERS = frozenset({
    "other", "such", "many", "several", "various", "same", "different", "certain", "own", "new", "first",
    "second", "third", "last", "next", "previous", "following", "main", "major", "few", "more", "most", "much",
    "some", "any", "each", "every", "all", "both", "whole", "entire", "particular", "specific", "overall",
})


@dataclass(slots=True)
class Mention:
    start: int
    end: int
    surface: str
    label: str
    kind: str
    key: str
    modifier: bool = False

    def pack(self) -> list:
        row = [self.start, self.end, self.surface, self.label, self.kind, self.key]
        if self.modifier:
            row.append(1)
        return row

    @classmethod
    def unpack(cls, row: list) -> "Mention":
        return cls(int(row[0]), int(row[1]), str(row[2]), str(row[3]), str(row[4]), str(row[5]), bool(row[6]) if len(row) > 6 else False)


@dataclass(slots=True)
class SentenceNote:
    start: int
    end: int
    lemmas: list[str] = field(default_factory=list)
    mentions: list[Mention] = field(default_factory=list)
    values: list[list] = field(default_factory=list)

    def pack(self) -> dict:
        return {
            "s": self.start,
            "e": self.end,
            "l": self.lemmas,
            "m": [m.pack() for m in self.mentions],
            "v": self.values,
        }

    @classmethod
    def unpack(cls, row: dict) -> "SentenceNote":
        return cls(
            start=int(row["s"]),
            end=int(row["e"]),
            lemmas=list(row["l"]),
            mentions=[Mention.unpack(m) for m in row["m"]],
            values=[list(v) for v in row["v"]],
        )


@dataclass(slots=True)
class PassageInput:
    doc_id: str
    page: int
    text: str
    title: str = ""
    origin: str = "text"


@dataclass(slots=True)
class PassageNote:
    doc_id: str
    page: int
    text: str
    title: str
    origin: str
    sentences: list[SentenceNote]

    def pack(self) -> dict:
        return {
            "doc": self.doc_id,
            "page": self.page,
            "text": self.text,
            "title": self.title,
            "origin": self.origin,
            "sents": [s.pack() for s in self.sentences],
        }

    @classmethod
    def unpack(cls, row: dict) -> "PassageNote":
        return cls(
            doc_id=row["doc"],
            page=int(row["page"]),
            text=row["text"],
            title=row.get("title", ""),
            origin=row.get("origin", "text"),
            sentences=[SentenceNote.unpack(s) for s in row["sents"]],
        )


def _content_lemma(tok) -> str | None:
    if tok.is_punct or tok.is_space or tok.like_url or tok.like_email:
        return None
    if not (tok.is_alpha or tok.like_num):
        if not any(ch.isalnum() for ch in tok.text):
            return None
    lemma = tok.lemma_.lower().strip()
    if not lemma or len(lemma) < 2 and not lemma.isdigit():
        return None
    if lemma in STOPWORDS or tok.lower_ in STOPWORDS:
        return None
    if lemma.startswith(("-", "'", "’")) or lemma == "-pron-":
        return None
    return lemma


def lemma_list(tokens: Iterable) -> list[str]:
    out = []
    for tok in tokens:
        lemma = _content_lemma(tok)
        if lemma:
            out.append(lemma)
    return out


def _trim_named(span: Span) -> Span | None:
    start, end = span.start, span.end
    doc = span.doc
    while start < end and (doc[start].lower_ in {"the", "a", "an"} or doc[start].is_punct):
        start += 1
    while end > start and (doc[end - 1].is_punct or doc[end - 1].tag_ == "POS" or doc[end - 1].lower_ in {"'s", "’s"}):
        end -= 1
    if end <= start:
        return None
    trimmed = doc[start:end]
    if not any(ch.isalpha() for ch in trimmed.text):
        return None
    return trimmed


def _concept_span(chunk: Span) -> Span | None:
    doc = chunk.doc
    start, end = chunk.start, chunk.end
    while start < end:
        tok = doc[start]
        if tok.pos_ in {"DET", "PRON", "PART", "NUM", "ADP", "CCONJ", "PUNCT"} or tok.lower_ in WEAK_MODIFIERS or tok.lower_ in STOPWORDS:
            start += 1
            continue
        break
    while end > start and (doc[end - 1].is_punct or doc[end - 1].tag_ == "POS"):
        end -= 1
    if end - start < 2 or end - start > 5:
        return None
    span = doc[start:end]
    root = span.root
    if root.pos_ not in {"NOUN", "PROPN"}:
        return None
    content = [t for t in span if t.is_alpha and not t.is_stop and len(t) > 1]
    if len(content) < 2:
        return None
    return span


class Annotator:
    _lock = threading.Lock()
    _cache: dict[str, Language] = {}

    def __init__(self, model: str = "en_core_web_md", batch_size: int = 48, role: str = "ingest"):
        self.model_name = model
        self.batch_size = batch_size
        self.nlp = self._load(model, role)
        self.has_vectors = bool(self.nlp.vocab.vectors.shape[0])
        self._parse_lock = threading.Lock()

    @classmethod
    def _load(cls, model: str, role: str = "ingest") -> Language:
        key = f"{model}:{role}"
        with cls._lock:
            if key in cls._cache:
                return cls._cache[key]
            try:
                nlp = spacy.load(model)
            except OSError:
                fallback = "en_core_web_sm"
                try:
                    if model == fallback:
                        raise OSError(model)
                    log.warning("spaCy model %s not found, falling back to %s", model, fallback)
                    nlp = spacy.load(fallback)
                except OSError as exc:
                    raise RuntimeError(
                        f"spaCy model '{model}' is not installed. Run: python -m spacy download {model}"
                    ) from exc
            nlp.max_length = 2_000_000
            cls._cache[key] = nlp
            return nlp

    def annotate(
        self,
        passages: list[PassageInput],
        progress: Callable[[int, int], None] | None = None,
    ) -> Iterator[PassageNote]:
        total = len(passages)
        texts = (p.text for p in passages)
        for done, (item, doc) in enumerate(zip(passages, self.nlp.pipe(texts, batch_size=self.batch_size)), start=1):
            yield self._note(item, doc)
            if progress and (done % 16 == 0 or done == total):
                progress(done, total)

    def parse(self, text: str) -> Doc:
        with self._parse_lock:
            return self.nlp(text)

    def _note(self, item: PassageInput, doc: Doc) -> PassageNote:
        sentences: list[SentenceNote] = []
        ents = list(doc.ents)
        ent_tokens = set()
        for ent in ents:
            ent_tokens.update(range(ent.start, ent.end))
        chunks_by_sent: dict[int, list[Span]] = {}
        try:
            noun_chunks = list(doc.noun_chunks)
        except ValueError:
            noun_chunks = []
        for chunk in noun_chunks:
            chunks_by_sent.setdefault(chunk.sent.start, []).append(chunk)
        for sent in doc.sents:
            note = SentenceNote(start=sent.start_char, end=sent.end_char, lemmas=lemma_list(sent))
            covered: set[int] = set()
            for ent in sent.ents:
                if ent.label_ in VALUE_LABELS:
                    covered.update(range(ent.start, ent.end))
                if ent.label_ in VALUE_LABELS:
                    note.values.append([ent.start_char - sent.start_char, ent.end_char - sent.start_char, ent.text, ent.label_])
                    continue
                if ent.label_ not in NAMED_LABELS:
                    continue
                span = _trim_named(ent)
                if span is None:
                    continue
                key = surface_key(span.text)
                if not key or len(key) < 2:
                    continue
                covered.update(range(span.start, span.end))
                root = span.root
                modifier = root.dep_ == "compound" and root.head.pos_ == "NOUN" and root.head.i >= span.end
                note.mentions.append(Mention(span.start_char, span.end_char, squash(span.text), ent.label_, "named", key, modifier))
            self._proper_runs(sent, covered, note)
            for chunk in chunks_by_sent.get(sent.start, []):
                if any(i in ent_tokens or i in covered for i in range(chunk.start, chunk.end)):
                    continue
                span = _concept_span(chunk)
                if span is None:
                    continue
                key = " ".join(t.lemma_.lower() for t in span if not t.is_punct)
                note.mentions.append(Mention(span.start_char, span.end_char, squash(span.text), "CONCEPT", "concept", key))
            note.mentions.sort(key=lambda m: m.start)
            sentences.append(note)
        return PassageNote(
            doc_id=item.doc_id,
            page=item.page,
            text=doc.text,
            title=item.title,
            origin=item.origin,
            sentences=sentences,
        )

    def _proper_runs(self, sent: Span, covered: set[int], note: SentenceNote) -> None:
        doc = sent.doc
        i = sent.start
        while i < sent.end:
            tok = doc[i]
            if tok.pos_ == "PROPN" and i not in covered and tok.is_alpha and (tok.is_title or tok.is_upper) and tok.lower_ not in CALENDAR:
                j = i
                while j < sent.end and doc[j].pos_ == "PROPN" and j not in covered and doc[j].lower_ not in CALENDAR and (doc[j].is_alpha or doc[j].text in {".", "-"}):
                    j += 1
                span = doc[i:j]
                if len(span.text) > 2 and not (i == sent.start and j - i == 1 and tok.lower_ in STOPWORDS):
                    key = surface_key(span.text)
                    if key and len(key) > 2:
                        note.mentions.append(Mention(span.start_char, span.end_char, squash(span.text), "MISC", "named", key))
                        covered.update(range(i, j))
                i = max(j, i + 1)
            else:
                i += 1

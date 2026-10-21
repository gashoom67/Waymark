from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from rapidfuzz import fuzz
from scipy import sparse

from waymark.graph.index import KnowledgeGraph
from waymark.graph.landmarks import row_normalize
from waymark.nlp.annotator import Annotator, lemma_list
from waymark.text import QUESTION_NOISE, STOPWORDS, surface_key, tokens_of

PLACE_NOUNS = frozenset("country city state town place location region province village capital continent county island nation district municipality area neighborhood neighbourhood birthplace hometown kingdom empire territory".split())
ORG_NOUNS = frozenset("company organization organisation university college school team club band group label party agency institution network channel newspaper magazine publisher firm corporation studio league church order army".split())
WORK_NOUNS = frozenset("film movie book novel album song show series game play opera poem painting single episode documentary work sitcom musical newspaper magazine".split())
DATE_NOUNS = frozenset("year date decade century day month era".split())
PERSON_NOUNS = frozenset("""person man woman actor actress director author writer singer composer player king queen
president leader founder father mother son daughter wife husband spouse child brother sister producer artist
musician scientist politician poet painter architect coach manager ruler emperor monarch prince princess
grandfather grandmother uncle aunt nephew niece successor predecessor ceo owner creator inventor star member
performer screenwriter novelist philosopher researcher professor teacher student""".split())
NUMBER_NOUNS = frozenset("number amount population percentage percent total count".split())
NORP_NOUNS = frozenset("nationality language religion ethnicity citizenship".split())

EARLY_WORDS = frozenset("first earlier earliest older oldest before sooner previously".split())
LATE_WORDS = frozenset("later latest last younger youngest after recent newer newest".split())
MORE_WORDS = frozenset("more most larger largest bigger biggest higher highest longer longest taller tallest greater greatest".split())
LESS_WORDS = frozenset("less least fewer fewest smaller smallest lower lowest shorter shortest".split())
COMPARISON_WORDS = EARLY_WORDS | LATE_WORDS | MORE_WORDS | LESS_WORDS
AUX_START = frozenset("is are was were do does did has have had can could will would should".split())
CLASSIFIER_NOUNS = frozenset("film movie song album book novel series show game play opera single episode documentary sitcom musical magazine newspaper poem painting track record".split())
ATTRIBUTE_NOUNS = frozenset("place birth death date year nationality country city cause name age location origin height population time day month century decade birthplace hometown capital".split())
RELATION_NOUNS = frozenset("""father mother parent son daughter child wife husband spouse brother sister sibling grandfather grandmother
grandson granddaughter grandchild uncle aunt nephew niece stepmother stepfather director producer composer performer
author writer creator founder owner ceo president leader member coach manager star editor publisher designer architect
successor predecessor heir partner screenwriter lyricist singer actor actress host presenter employer label team
company university school college headquarters""".split())
COMPOUND_RELATIONS: dict[str, list[tuple[str, ...]]] = {
    "paternal grandfather": [("father",), ("father",)],
    "paternal grandmother": [("father",), ("mother",)],
    "maternal grandfather": [("mother",), ("father",)],
    "maternal grandmother": [("mother",), ("mother",)],
    "grandfather": [("father", "mother", "parent"), ("father",)],
    "grandmother": [("father", "mother", "parent"), ("mother",)],
    "grandson": [("son", "daughter", "child"), ("son",)],
    "granddaughter": [("son", "daughter", "child"), ("daughter",)],
    "grandchild": [("son", "daughter", "child"), ("son", "daughter", "child")],
    "father in law": [("wife", "husband", "spouse"), ("father",)],
    "mother in law": [("wife", "husband", "spouse"), ("mother",)],
    "son in law": [("daughter",), ("husband",)],
    "daughter in law": [("son",), ("wife",)],
    "brother in law": [("wife", "husband", "spouse"), ("brother",)],
    "sister in law": [("wife", "husband", "spouse"), ("sister",)],
    "stepmother": [("father",), ("wife",)],
    "stepfather": [("mother",), ("husband",)],
    "uncle": [("father", "mother", "parent"), ("brother",)],
    "aunt": [("father", "mother", "parent"), ("sister",)],
    "nephew": [("brother", "sister", "sibling"), ("son",)],
    "niece": [("brother", "sister", "sibling"), ("daughter",)],
    "spouse": [("wife", "husband", "spouse", "marry")],
    "parent": [("father", "mother", "parent")],
    "sibling": [("brother", "sister", "sibling")],
    "child": [("son", "daughter", "child")],
}
ORG_AGENT_VERBS = frozenset("""fund own operate run sponsor publish finance acquire manage build install produce manufacture
distribute release develop organise organize host back employ supply construct design maintain administer""".split())
CONNECTION_WORDS = frozenset("connect connected connection link linked links relate related relationship relation between common".split())

_word = re.compile(r"[^\W_]+", re.UNICODE)
_paren = re.compile(r"\(([^)]*)\)")


@dataclass
class Anchor:
    text: str
    start: int
    end: int
    candidates: list[tuple[int, float]]
    soft: bool = False

    @property
    def best(self) -> int:
        return self.candidates[0][0]


@dataclass
class Question:
    text: str
    wh: str
    expected: frozenset[str]
    head: str | None
    anchors: list[Anchor]
    cues: dict[str, float]
    terms: dict[int, float]
    mode: str = "lookup"
    plan: list[tuple[str, ...]] = field(default_factory=list)
    tail: dict[str, float] = field(default_factory=dict)
    focus: str | None = None
    present: bool = False
    direction: str | None = None
    options: list[Anchor] = field(default_factory=list)
    lemmas: list[str] = field(default_factory=list)


def _expected_for(head: str | None, wh: str) -> frozenset[str]:
    if wh in {"who", "whom", "whose"}:
        return frozenset({"PERSON"})
    if wh == "where":
        return frozenset({"PLACE"})
    if wh == "when":
        return frozenset({"DATE"})
    if wh in {"how many", "how much"}:
        return frozenset({"NUMBER"})
    if wh == "yesno":
        return frozenset({"BOOL"})
    if head:
        if head in DATE_NOUNS:
            return frozenset({"DATE"})
        if head in PLACE_NOUNS:
            return frozenset({"PLACE"})
        if head in ORG_NOUNS:
            return frozenset({"ORG"})
        if head in WORK_NOUNS:
            return frozenset({"WORK"})
        if head in PERSON_NOUNS:
            return frozenset({"PERSON"})
        if head in NUMBER_NOUNS:
            return frozenset({"NUMBER"})
        if head in NORP_NOUNS:
            return frozenset({"NORP", "PLACE"})
    return frozenset()


class QuestionReader:
    def __init__(self, graph: KnowledgeGraph, annotator: Annotator):
        self.graph = graph
        self.annotator = annotator

    def read(self, text: str) -> Question:
        clean = " ".join(text.strip().split())
        doc = self.annotator.parse(clean)
        wh, head = self._wh(doc)
        anchors = self._anchors(clean, doc)
        spans = [(a.start, a.end) for a in anchors if not a.soft]
        spans.extend((m.start(), m.end()) for m in _paren.finditer(clean))
        free = [t for t in doc if not any(lo <= t.idx < hi for lo, hi in spans)]
        cue_lemmas: list[str] = []
        for lemma in lemma_list(free):
            if lemma in QUESTION_NOISE or lemma in STOPWORDS:
                continue
            cue_lemmas.append(lemma)
        if head and head not in cue_lemmas:
            cue_lemmas.append(head)
        corpus = self.graph.corpus
        idf = self.graph.sentence_bm25.idf
        cues: dict[str, float] = {}
        for lemma in cue_lemmas:
            tid = corpus.vocab.get(lemma)
            weight = float(idf[tid]) if tid is not None and tid < len(idf) else 2.0
            cues[lemma] = max(cues.get(lemma, 0.0), max(weight, 0.5))
        terms: dict[int, float] = {}
        for lemma in lemma_list(doc):
            if lemma in QUESTION_NOISE:
                continue
            tid = corpus.vocab.get(lemma)
            if tid is not None:
                terms[tid] = terms.get(tid, 0.0) + 1.0
        for a in anchors:
            for tok in tokens_of(a.text):
                tid = corpus.vocab.get(tok)
                if tid is not None and tok not in STOPWORDS:
                    terms[tid] = max(terms.get(tid, 0.0), 1.0)
        expected = _expected_for(head, wh)
        if wh == "where" and set(cues) & {"work", "study", "educate", "attend", "teach", "employ", "graduate", "train", "enrol", "enroll"}:
            expected = frozenset({"ORG", "PLACE"})
        question = Question(
            text=clean,
            wh=wh,
            expected=expected,
            head=head,
            anchors=anchors,
            cues=cues,
            terms=terms,
            lemmas=[t.lemma_.lower() for t in doc],
        )
        self._disambiguate(question, doc)
        self._mode(question, free)
        self._plan(question, doc, free)
        question.focus = self._focus(doc, free)
        lowered_words = {t.lower_ for t in doc}
        question.present = bool(lowered_words & {"does", "do", "is", "are", "currently", "now"}) and not bool(lowered_words & {"previously", "formerly", "former", "was", "were", "did", "before", "earlier"})
        if question.wh in {"who", "whom"} and question.focus in ORG_AGENT_VERBS:
            question.expected = frozenset({"PERSON", "ORG"})
        return question

    def _focus(self, doc, free) -> str | None:
        free_idx = {t.i for t in free}
        for tok in doc:
            if tok.lower_ not in {"who", "what", "whom"}:
                continue
            j = tok.i + 1
            if j < len(doc) and doc[j].lemma_ in {"be"}:
                j += 1
                while j < len(doc) and doc[j].pos_ in {"DET", "ADJ", "NUM"} and doc[j].i in free_idx:
                    j += 1
                if j < len(doc) and doc[j].pos_ == "NOUN" and doc[j].i in free_idx:
                    lemma = doc[j].lemma_.lower()
                    return None if lemma in ATTRIBUTE_NOUNS or lemma in CLASSIFIER_NOUNS else lemma
                return None
            if j < len(doc) and doc[j].pos_ == "VERB" and doc[j].i in free_idx:
                return doc[j].lemma_.lower()
            return None
        return None

    def _wh(self, doc) -> tuple[str, str | None]:
        words = [t.lower_ for t in doc]
        if not words:
            return "none", None
        for i, w in enumerate(words):
            if w == "how" and i + 1 < len(words) and words[i + 1] in {"many", "much"}:
                return f"how {words[i + 1]}", None
        for tok in doc:
            if tok.lower_ in {"which", "what"}:
                head = None
                if tok.dep_ == "det" and tok.head.pos_ in {"NOUN", "PROPN"}:
                    head = tok.head.lemma_.lower()
                else:
                    nxt = tok.i + 1
                    while nxt < len(doc) and doc[nxt].pos_ in {"ADJ", "ADV"}:
                        nxt += 1
                    if nxt < len(doc) and doc[nxt].pos_ == "NOUN":
                        head = doc[nxt].lemma_.lower()
                    elif tok.lower_ == "what" and tok.i + 1 < len(doc) and doc[tok.i + 1].lemma_ == "be":
                        for t in doc[tok.i + 2:]:
                            if t.dep_ in {"attr", "nsubj", "pobj"} and t.pos_ == "NOUN":
                                head = t.lemma_.lower()
                                break
                if head in {"kind", "type", "sort"}:
                    for t in doc[tok.i:]:
                        if t.lower_ == "of" and t.i + 1 < len(doc):
                            head = doc[t.i + 1].lemma_.lower()
                            break
                return tok.lower_, head
            if tok.lower_ in {"who", "whom", "whose", "when", "where", "why", "how"}:
                return tok.lower_, None
        if words[0] in AUX_START:
            return "yesno", None
        return "none", None

    def _anchors(self, text: str, doc) -> list[Anchor]:
        graph = self.graph
        pieces: list[tuple[str, int, int]] = []
        for match in _word.finditer(text):
            for tok in tokens_of(match.group(0)):
                pieces.append((tok, match.start(), match.end()))
        blocked = set()
        for paren in _paren.finditer(text):
            blocked.update(i for i, (_, a, _) in enumerate(pieces) if paren.start() < a < paren.end())
        n = len(pieces)
        used = [i in blocked for i in range(n)]
        propn = {t.idx for t in doc if t.pos_ == "PROPN"}
        found: list[Anchor] = []
        for size in range(min(graph.max_alias_len, n), 0, -1):
            for i in range(0, n - size + 1):
                if any(used[i:i + size]):
                    continue
                window = pieces[i:i + size]
                key = surface_key(" ".join(w for w, _, _ in window))
                if not key:
                    continue
                eids = graph.alias_index.get(key)
                if not eids:
                    continue
                start, end = window[0][1], window[-1][2]
                raw = text[start:end]
                words = key.split()
                capital = any(text[a:b][:1].isupper() and a > 0 for _, a, b in window) or any(a in propn for _, a, _ in window)
                digits = any(w.isdigit() for w in words)
                named = [e for e in eids if graph.res.entity_kind[e] == "named"]
                concepts = [e for e in eids if graph.res.entity_kind[e] == "concept"]
                if len(words) == 1:
                    word = words[0]
                    if word in STOPWORDS or word in QUESTION_NOISE or len(word) < 3 or not (capital or digits) or not named:
                        continue
                    eids = named
                elif not (capital or digits):
                    if not concepts:
                        continue
                    eids = concepts
                elif named:
                    eids = named
                if all(w in STOPWORDS or w in QUESTION_NOISE for w in words):
                    continue
                soft = all(graph.res.entity_kind[e] == "concept" for e in eids)
                found.append(Anchor(text=raw, start=start, end=end, candidates=[(e, 1.0) for e in eids], soft=soft))
                for k in range(i, i + size):
                    used[k] = True
        for span in self._loose_spans(doc):
            if any(not (span.end_char <= a.start or span.start_char >= a.end) for a in found):
                continue
            if any(paren.start() < span.start_char < paren.end() for paren in _paren.finditer(text)):
                continue
            match = self._fuzzy(span.text)
            if match:
                found.append(Anchor(text=span.text, start=span.start_char, end=span.end_char, candidates=match))
        found.sort(key=lambda a: a.start)
        return found

    def _loose_spans(self, doc):
        spans = [e for e in doc.ents if e.label_ not in {"DATE", "TIME", "CARDINAL", "ORDINAL", "PERCENT", "MONEY", "QUANTITY"}]
        i = 0
        while i < len(doc):
            if doc[i].pos_ == "PROPN" or (doc[i].text[:1].isupper() and i > 0 and doc[i].is_alpha):
                j = i
                while j < len(doc) and (doc[j].pos_ == "PROPN" or (doc[j].text[:1].isupper() and doc[j].is_alpha)):
                    j += 1
                spans.append(doc[i:j])
                i = j
            else:
                i += 1
        return spans

    def _fuzzy(self, text: str) -> list[tuple[int, float]]:
        graph = self.graph
        toks = [t for t in tokens_of(text) if t not in STOPWORDS and len(t) > 2]
        if not toks:
            return []
        pool: set[int] = set()
        for t in toks:
            pool.update(graph.token_index.get(t, [])[:400])
        if not pool:
            return []
        target = surface_key(text)
        scored = []
        for eid in pool:
            best = max(fuzz.token_set_ratio(target, alias) for alias in graph.res.entity_aliases[eid])
            if best >= 88:
                scored.append((eid, best / 100.0))
        scored.sort(key=lambda x: -x[1])
        return scored[:6]

    def _disambiguate(self, question: Question, doc) -> None:
        graph = self.graph
        if not question.anchors:
            return
        corpus = graph.corpus
        field_ = graph.res.context
        vocab_size = len(field_.term_idf)
        lemma_ids = [corpus.vocab[l] for l in lemma_list(doc) if l in corpus.vocab]
        cols = [i for i in lemma_ids if i < vocab_size]
        data = [float(field_.term_idf[i]) for i in cols]
        width = field_.sentence_vectors.shape[1]
        for a in question.anchors:
            for other in question.anchors:
                if other is a:
                    continue
                key = surface_key(other.text)
                land = field_.landmark_keys.get(key)
                if land is not None:
                    cols.append(vocab_size + land)
                    data.append(3.0)
        vec = row_normalize(sparse.csr_matrix((data, ([0] * len(cols), cols)), shape=(1, width))) if cols else None
        folded = set(tokens_of(question.text))
        for a in question.anchors:
            if len(a.candidates) == 1:
                continue
            eids = [e for e, _ in a.candidates]
            prior = np.asarray([w for _, w in a.candidates], dtype=np.float64)
            after = set(tokens_of(question.text[a.end:a.end + 40]))
            for k, eid in enumerate(eids):
                qual = graph.res.entity_qualifier[eid].split()
                if qual:
                    near = len(set(qual) & after) / len(qual)
                    anywhere = len(set(qual) & folded) / len(qual)
                    prior[k] *= 1.0 + 4.0 * near + 1.0 * anywhere
            if vec is not None:
                sims = np.asarray((graph.res.profiles[eids] @ vec.T).todense()).ravel()
            else:
                sims = np.zeros(len(eids))
            mass = np.log1p(np.asarray([len(graph.res.entity_mentions[e]) for e in eids], dtype=np.float64))
            score = prior * (1.0 + 4.0 * sims) + 0.05 * mass
            order = np.argsort(-score)
            total = float(score.sum()) or 1.0
            a.candidates = [(eids[i], float(score[i] / total)) for i in order]

    def _mode(self, question: Question, free) -> None:
        lowered = question.text.lower()
        words = {t.lower_ for t in free}
        joined = " " + " ".join(t.lower_ for t in free) + " "
        named = [a for a in question.anchors if not a.soft]
        if words & CONNECTION_WORDS and len(named) >= 2 and question.wh in {"how", "what", "none", "which"} and not question.expected:
            question.mode = "connect"
            return
        comparative = bool(words & COMPARISON_WORDS) or ("live" in question.cues and "long" in question.cues)
        if len(named) >= 2 and comparative and " or " not in joined and (" and " in joined or " out of " in joined):
            question.options = named[-2:]
            question.mode = "choose"
            question.direction = self._direction(words, question)
            return
        if len(named) >= 2 and " or " in joined:
            tail = lowered.rsplit(",", 1)[-1] if "," in lowered else lowered
            options = [a for a in named if a.start >= len(question.text) - len(tail) - 1] or named
            if len(options) >= 2:
                question.options = options[-2:]
                question.mode = "choose"
                question.direction = self._direction(words, question)
                return
        if question.wh == "yesno" and len(named) >= 2:
            if "same" in words or "both" in words or "also" in words:
                question.options = named[:2] if "same" in words else named
                question.mode = "same" if "same" in words else "both"
                return
        if question.wh == "yesno":
            question.mode = "verify"

    def _direction(self, words: set[str], question: Question) -> str | None:
        if "live" in question.cues and ("long" in question.cues or "longer" in words):
            return "lifespan"
        if words & EARLY_WORDS:
            return "early"
        if words & LATE_WORDS:
            return "late"
        if words & MORE_WORDS:
            return "more"
        if words & LESS_WORDS:
            return "less"
        return None

    def _plan(self, question: Question, doc, free) -> None:
        words = [(t.lower_, t.lemma_.lower(), t.idx, t.pos_) for t in doc]
        free_idx = {t.idx for t in free}
        relations: list[str] = []
        if question.mode in {"choose", "same", "both"}:
            for i, (low, lemma, idx, pos) in enumerate(words):
                if idx not in free_idx:
                    continue
                if lemma in RELATION_NOUNS and pos in {"NOUN", "PROPN"} and lemma not in CLASSIFIER_NOUNS:
                    phrase = self._compound(words, i)
                    if phrase not in relations:
                        relations.append(phrase)
        else:
            hard = [a for a in question.anchors if not a.soft]
            if hard:
                relations = self._relations_around(words, hard[0])
        plan: list[tuple[str, ...]] = []
        for rel in relations:
            plan.extend(COMPOUND_RELATIONS.get(rel, [tuple(rel.split()[-1:])]))
        question.plan = plan[:4]
        used = {w for group in question.plan for w in group} | {w for rel in relations for w in rel.split()}
        redundant = {"place", "location", "date", "year", "time", "day", "country", "city", "name"} if question.expected & {"PLACE", "DATE", "NORP"} else set()
        question.tail = {k: v for k, v in question.cues.items() if k not in used and k not in CLASSIFIER_NOUNS and k not in COMPARISON_WORDS and k not in redundant}
        for group in question.plan:
            for w in group:
                if w not in question.cues:
                    question.cues[w] = 2.0

    def _compound(self, words, i: int) -> str:
        lemma = words[i][1]
        k = i + 1
        while k < len(words) and words[k][0] in {"in", "-"}:
            k += 1
        if k > i + 1 and k < len(words) and words[k][0] == "law":
            return f"{lemma} in law"
        if i > 0 and words[i - 1][0] in {"paternal", "maternal"}:
            return f"{words[i - 1][0]} {lemma}"
        return lemma

    def _relations_around(self, words, anchor: Anchor) -> list[str]:
        n = len(words)
        start = next((i for i, w in enumerate(words) if w[2] >= anchor.start), n)
        end = next((i for i, w in enumerate(words) if w[2] >= anchor.end), n)
        forward: list[str] = []
        j = end
        if j < n and words[j][0] == "(":
            while j < n and words[j][0] != ")":
                j += 1
            j += 1
        while j < n and words[j][0] in {"'s", "’s", "'", "’"}:
            k = j + 1
            while k < n and words[k][3] in {"ADJ", "DET"} and words[k][0] not in {"paternal", "maternal"}:
                k += 1
            if k < n and words[k][0] in {"paternal", "maternal"}:
                k += 1
            if k < n and words[k][3] in {"NOUN", "PROPN"}:
                phrase = self._compound(words, k)
                if words[k][1] not in ATTRIBUTE_NOUNS and words[k][1] not in CLASSIFIER_NOUNS:
                    forward.append(phrase)
                j = k + 1
                while j < n and words[j][0] in {"-", "in", "law"}:
                    j += 1
            else:
                break
        backward: list[str] = []
        i = start - 1
        while i >= 0 and (words[i][1] in CLASSIFIER_NOUNS or words[i][3] == "DET"):
            i -= 1
        while i >= 1 and words[i][0] == "of":
            k = i - 1
            while k >= 0 and words[k][3] == "DET":
                k -= 1
            if k >= 0 and words[k][0] == "law" and k >= 3 and words[k - 1][0] in {"-", "in"}:
                k2 = k - 1
                while k2 >= 0 and words[k2][0] in {"-", "in"}:
                    k2 -= 1
                backward.append(f"{words[k2][1]} in law")
                i = k2 - 1
            elif k >= 0 and words[k][3] in {"NOUN", "PROPN"}:
                lemma = words[k][1]
                if lemma not in ATTRIBUTE_NOUNS and lemma not in CLASSIFIER_NOUNS:
                    backward.append(self._compound(words, k))
                i = k - 1
                while i >= 0 and words[i][3] in {"DET", "ADJ"}:
                    i -= 1
            else:
                break
        return forward + backward

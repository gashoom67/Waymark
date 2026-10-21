from __future__ import annotations

from dataclasses import dataclass, field

import re

import numpy as np

from waymark.graph.index import KnowledgeGraph
from waymark.nlp.annotator import Annotator
from waymark.reasoning.lexicon import COUNTRIES, DEMONYMS, FEMALE_RELATIONS, GENERIC_BRIDGES, MALE_RELATIONS, RELATED
from waymark.reasoning.values import parse_date, parse_number
from waymark.text import stem_like, surface_key

NUMBER_LABELS = frozenset({"CARDINAL", "QUANTITY", "MONEY", "PERCENT"})
PERSON_RELATIONS = frozenset("""father mother parent son daughter child wife husband spouse marry brother sister sibling director producer
composer performer author writer creator founder owner ceo president leader coach manager star editor designer architect
successor predecessor heir partner screenwriter lyricist singer actor actress host presenter""".split())
ORG_RELATIONS = frozenset("employer label team company university school college publisher".split())
_words = re.compile(r"[A-Za-z]+")
_lifespan_parens = re.compile(r"\((?:[^()]{0,30}?\d{3,4}[^()]{0,40}?)\)")
_child_of = re.compile(r"\b(?:son|daughter|child|children|sons|daughters)\s+of\b[^,;]*$")
_parent_of = re.compile(r"\b(?:father|mother|parent|parents)\s+of\b[^,;]*$")
_of_extension = re.compile(r"\s+of\s+(?:[A-Z][\w'’.-]*)(?:[\s-]+(?:[A-Z][\w'’.-]*|de|la|du|von|van))*")
STAGE_THRESHOLD = 0.78
WORK_CUES = frozenset({"work", "employ", "job", "career"})
_former = re.compile(r"\b(?:previously|formerly|former|used to|until|once|earlier)\b", re.IGNORECASE)
_occupation_at = re.compile(r"\b(?:an?|the)\s+(?:[a-z-]+\s+){0,2}[a-z]+(?:ist|er|or|ian|ant|ent|ee|eer|ive)\s+(?:at|with|for)\s+(?:the\s+)?$", re.IGNORECASE)
ROLE_CUES = {
    "birth": frozenset({"bear", "birth", "born", "birthplace", "old", "young", "birthday"}),
    "death": frozenset({"die", "death", "dead", "decease"}),
}
CLASSIFIER_WORDS = frozenset("""film films movie song album book novel serial series show game play opera single episode documentary
sitcom musical comedy drama thriller western silent animated feature short""".split())

TYPE_FIT: dict[str, dict[str, float]] = {
    "PERSON": {"PERSON": 1.0, "ORG": 0.6, "MISC": 0.35},
    "PLACE": {"PLACE": 1.0, "NORP": 0.5, "ORG": 0.45, "MISC": 0.3},
    "ORG": {"ORG": 1.0, "MISC": 0.4, "PLACE": 0.25, "WORK": 0.2},
    "WORK": {"WORK": 1.0, "MISC": 0.5, "EVENT": 0.35, "ORG": 0.15},
    "NORP": {"NORP": 1.0, "PLACE": 0.7, "MISC": 0.2},
    "EVENT": {"EVENT": 1.0, "MISC": 0.4, "WORK": 0.3},
}


@dataclass
class Hop:
    sentence: int
    source: int
    target: int | None
    gain: float


@dataclass
class Chain:
    score: float
    entity: int
    covered: np.ndarray
    hops: list[Hop] = field(default_factory=list)
    entities: frozenset[int] = frozenset()
    sentences: frozenset[int] = frozenset()
    stage: int = 0

    def coverage(self, weights: np.ndarray) -> float:
        return float(self.covered @ weights) if len(weights) else 0.0


@dataclass
class Candidate:
    text: str
    key: str
    kind: str
    label: str
    score: float
    chain: Chain
    sentence: int
    entity: int | None = None
    value: float | None = None
    support: int = 1


class CueMatcher:
    def __init__(self, graph: KnowledgeGraph, annotator: Annotator, cues: dict[str, float]):
        self.graph = graph
        self.names = list(cues.keys())
        raw = np.asarray(list(cues.values()), dtype=np.float64)
        self.raw_weights = raw
        self.weights = raw / raw.sum() if raw.sum() > 0 else raw
        self.vocab = annotator.nlp.vocab
        self.use_vectors = annotator.has_vectors
        self.cue_vectors = [self._vector(c) for c in self.names]
        self.term_cache: dict[int, np.ndarray] = {}
        self.sent_cache: dict[int, np.ndarray] = {}

    def _vector(self, word: str) -> np.ndarray | None:
        if not self.use_vectors or not self.vocab.has_vector(word):
            return None
        vec = self.vocab.get_vector(word)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else None

    def term(self, tid: int) -> np.ndarray:
        cached = self.term_cache.get(tid)
        if cached is not None:
            return cached
        lemma = self.graph.corpus.lemmas[tid]
        out = np.zeros(len(self.names), dtype=np.float64)
        vec = None
        for i, cue in enumerate(self.names):
            if lemma == cue:
                out[i] = 1.0
                continue
            if lemma in RELATED.get(cue, ()) or cue in RELATED.get(lemma, ()):
                out[i] = 0.85
                continue
            if stem_like(lemma, cue):
                out[i] = 0.8
                continue
            cue_vec = self.cue_vectors[i]
            if cue_vec is None:
                continue
            if vec is None:
                vec = self._vector(lemma)
                if vec is None:
                    vec = np.zeros(0)
            if len(vec):
                sim = float(vec @ cue_vec)
                if sim >= 0.6:
                    out[i] = 0.75 * sim
        self.term_cache[tid] = out
        return out

    def word_hit(self, word: str, cues: tuple[int, ...]) -> bool:
        low = word.lower()
        forms = {low}
        for suffix in ("ing", "ed", "es", "er", "d", "s"):
            if low.endswith(suffix) and len(low) - len(suffix) >= 3:
                forms.add(low[: -len(suffix)])
                forms.add(low[: -len(suffix)] + "e")
        for i in cues:
            cue = self.names[i]
            related = RELATED.get(cue, ())
            for form in forms:
                if form == cue or form in related or stem_like(form, cue):
                    return True
        return False

    def near(self, text: str, start: int, end: int, cues: tuple[int, ...], before: int = 42, after: int = 24) -> float:
        if not cues or start < 0:
            return 0.0
        left = _words.findall(text[max(0, start - before):start])[-5:]
        right = _words.findall(text[end:end + after])[:3]
        best = 0.0
        for dist, word in enumerate(reversed(left)):
            if self.word_hit(word, cues):
                best = max(best, 1.0 / (1.0 + 0.35 * dist))
        for dist, word in enumerate(right):
            if self.word_hit(word, cues):
                best = max(best, 0.7 / (1.0 + 0.35 * dist))
        return best

    def sentence(self, sid: int) -> np.ndarray:
        cached = self.sent_cache.get(sid)
        if cached is not None:
            return cached
        terms = self.graph.corpus.sent_terms[sid]
        if not len(self.names) or not len(terms):
            out = np.zeros(len(self.names), dtype=np.float64)
        else:
            out = np.max(np.vstack([self.term(int(t)) for t in set(terms.tolist())]), axis=0)
        self.sent_cache[sid] = out
        return out


class ChainSearch:
    def __init__(
        self,
        graph: KnowledgeGraph,
        matcher: CueMatcher,
        passage_prior: np.ndarray,
        pool: set[int],
        beam: int = 32,
        sentences_per_entity: int = 120,
    ):
        self.graph = graph
        self.matcher = matcher
        self.prior = passage_prior
        self.pool = pool
        self.beam = beam
        self.limit = sentences_per_entity
        self._sent_cache: dict[int, list[int]] = {}
        self.prefer_countries = False
        self.focus: tuple[int, ...] = ()
        self.present_tense = False
        self.configure([], list(matcher.names))

    def configure(self, plan: list[tuple[str, ...]], tail: list[str], focus: str | None = None) -> None:
        names = self.matcher.names
        index = {n: i for i, n in enumerate(names)}
        plan_words = {w for group in plan for w in group}
        self.focus = (index[focus],) if focus and focus in index and focus not in plan_words else ()
        self.plan = [tuple(index[w] for w in group if w in index) for group in plan]
        self.plan = [g for g in self.plan if g]
        self.plan_words = [tuple(w for w in group if w in index) for group in plan if any(w in index for w in group)]
        tail_idx = [index[w] for w in tail if w in index]
        self.tail_idx = tuple(tail_idx)
        weights = np.zeros(len(names), dtype=np.float64)
        if tail_idx:
            raw = self.matcher.raw_weights[tail_idx]
            weights[tail_idx] = raw / raw.sum() if raw.sum() > 0 else 1.0 / len(tail_idx)
        self.weights = weights
        self.has_tail = bool(tail_idx)

    def _stage_class(self, stage: int) -> str | None:
        if stage >= len(self.plan_words):
            return None
        words = set(self.plan_words[stage])
        if words & PERSON_RELATIONS:
            return "PERSON"
        if words & ORG_RELATIONS:
            return "ORG"
        return None

    def _spans(self, sid: int, eid: int) -> list[tuple[int, int]]:
        corpus = self.graph.corpus
        ent = self.graph.res.mention_entity
        return [corpus.m_span[m] for m in corpus.sent_mentions[sid] if ent[m] == eid and corpus.m_span[m][0] >= 0]

    def _proximity(self, sid: int, eid: int | None, cues: tuple[int, ...], value_span: tuple[int, int] | None = None) -> float:
        if not cues:
            return 0.0
        corpus = self.graph.corpus
        pid = int(corpus.sent_passage[sid])
        passage = corpus.passages[pid]
        sent = passage.sentences[int(corpus.sent_order[sid])]
        if value_span is not None:
            spans = [(sent.start + value_span[0], sent.start + value_span[1])]
        elif eid is not None:
            spans = self._spans(sid, eid)
        else:
            spans = []
        if not spans:
            return 0.0
        best = 0.0
        for start, end in spans:
            lo = max(sent.start, start - 42)
            best = max(best, self.matcher.near(passage.text[lo:min(sent.end, end + 24)], start - lo, end - lo, cues))
        return best if best > 0 else -0.3

    def _stage_gender(self, stage: int) -> float:
        if stage >= len(self.plan_words):
            return 0.0
        words = set(self.plan_words[stage])
        if words <= FEMALE_RELATIONS:
            return -1.0
        if words <= MALE_RELATIONS:
            return 1.0
        return 0.0

    def sentences_for(self, eid: int) -> list[int]:
        cached = self._sent_cache.get(eid)
        if cached is not None:
            return cached
        graph = self.graph
        corpus = graph.corpus
        mentioned = graph.entity_sentences(eid)
        passages: list[int] = []
        seen: set[int] = set()
        for pid in graph.entity_homes[eid].tolist():
            if pid not in seen:
                seen.add(pid)
                passages.append(pid)
        mentioned_passages = {int(corpus.sent_passage[s]) for s in mentioned}
        for pid in sorted(mentioned_passages & self.pool, key=lambda p: -self.prior[p]):
            if pid not in seen:
                seen.add(pid)
                passages.append(pid)
        if len(passages) < 4:
            for pid in sorted(mentioned_passages - seen, key=lambda p: -self.prior[p])[:4]:
                seen.add(pid)
                passages.append(pid)
        mentioned_set = set(mentioned.tolist())
        out: list[int] = []
        for pid in passages:
            topical = graph.passage_topic[pid] == eid
            for sid in corpus.passage_sents[pid]:
                if topical or sid in mentioned_set:
                    out.append(sid)
            if len(out) >= self.limit:
                break
        self._sent_cache[eid] = out[: self.limit]
        return self._sent_cache[eid]

    def sentence_entities(self, sid: int) -> list[int]:
        graph = self.graph
        ents = graph.sentence_entities(sid).tolist()
        topic = int(graph.passage_topic[int(graph.corpus.sent_passage[sid])])
        if topic >= 0 and topic not in ents:
            ents.append(topic)
        return ents

    def bridge_ok(self, eid: int) -> bool:
        res = self.graph.res
        if res.entity_class[eid] == "NORP":
            return False
        if res.entity_kind[eid] == "concept":
            return surface_key(res.entity_name[eid]) not in GENERIC_BRIDGES
        return True

    def run(
        self,
        starts: list[tuple[int, float]],
        expected: frozenset[str],
        max_hops: int,
        forbid: set[int],
    ) -> list[Candidate]:
        graph = self.graph
        corpus = graph.corpus
        n_cues = len(self.matcher.names)
        plan = self.plan
        weights = self.weights
        final_class = self._stage_class(len(plan) - 1) if plan else None
        attribute_mode = bool(plan) and bool(expected) and final_class is not None and not any(
            TYPE_FIT.get(t, {}).get(final_class, 0.0) >= 1.0 for t in expected
        )
        if attribute_mode or (plan and self.has_tail):
            max_hops = min(max_hops, len(plan) + 1)
        elif plan:
            max_hops = min(max_hops, len(plan) + 1)
        beam: list[Chain] = []
        for eid, weight in starts:
            beam.append(Chain(
                score=0.6 * float(np.log1p(weight)),
                entity=eid,
                covered=np.zeros(n_cues),
                entities=frozenset({eid}),
            ))
        found: dict[str, Candidate] = {}
        for depth in range(max_hops):
            grown: list[Chain] = []
            for chain in beam:
                stage = chain.stage
                for sid in self.sentences_for(chain.entity):
                    if sid in chain.sentences:
                        continue
                    cov = self.matcher.sentence(sid)
                    match = float(cov[list(plan[stage])].max()) if stage < len(plan) else 0.0
                    advance = match >= STAGE_THRESHOLD and not self._reversed(sid, chain.entity, stage)
                    nxt_stage = stage + 1 if advance else stage
                    merged = np.maximum(chain.covered, cov)
                    before = float(chain.covered @ weights)
                    total = float(merged @ weights)
                    gain = total - before
                    pid = int(corpus.sent_passage[sid])
                    base = chain.score + 0.3 * float(self.prior[pid]) - 0.4 + (1.2 * match if advance else 0.0) + 0.4 * gain
                    answerable = nxt_stage >= len(plan) and (advance or not plan or self.has_tail)
                    if attribute_mode:
                        answerable = stage >= len(plan)
                    if answerable:
                        self._collect(found, chain, sid, pid, expected, forbid, advance, stage, nxt_stage, base, merged, total, gain)
                    if depth + 1 >= max_hops:
                        continue
                    if plan:
                        if not advance:
                            continue
                    elif gain <= 1e-9 and depth > 0:
                        continue
                    stage_gender = self._stage_gender(stage) if advance else 0.0
                    want = self._stage_class(stage) if advance else None
                    for nxt in self.sentence_entities(sid):
                        if nxt in chain.entities or nxt in forbid or not self.bridge_ok(nxt):
                            continue
                        bonus = 0.2 * graph.specificity(nxt)
                        if want:
                            cls = graph.res.entity_class[nxt]
                            bonus += 0.4 if cls == want else (0.05 if cls == "MISC" else -0.6)
                        if advance:
                            bonus += 0.5 * self._proximity(sid, nxt, plan[stage])
                        if stage_gender and graph.res.entity_class[nxt] in {"PERSON", "MISC"}:
                            bonus += 0.5 * stage_gender * float(graph.entity_gender[nxt])
                        grown.append(Chain(
                            score=base + bonus,
                            entity=nxt,
                            covered=merged,
                            hops=chain.hops + [Hop(sid, chain.entity, nxt, gain)],
                            entities=chain.entities | {nxt},
                            sentences=chain.sentences | {sid},
                            stage=nxt_stage,
                        ))
            if not grown:
                break
            grown.sort(key=lambda c: -(c.score + 2.2 * float(c.covered @ weights) + 1.5 * c.stage))
            dedup: dict[tuple[int, int, int], Chain] = {}
            for chain in grown:
                sig = (chain.entity, chain.stage, int(round(float(chain.covered @ weights) * 20)))
                if sig not in dedup:
                    dedup[sig] = chain
                if len(dedup) >= self.beam:
                    break
            beam = list(dedup.values())
        return sorted(found.values(), key=lambda c: -(c.score + 0.08 * np.log1p(c.support)))

    def _collect(self, found, chain, sid, pid, expected, forbid, advance, stage, nxt_stage, base, merged, total, gain) -> None:
        graph = self.graph
        corpus = graph.corpus
        plan = self.plan
        weights = self.weights
        banned = forbid | set(chain.entities)
        stage_gender = self._stage_gender(stage) if advance else 0.0
        focus = self.tail_idx if self.has_tail else (plan[stage] if advance else ())
        definitional = int(corpus.sent_order[sid]) == 0 and graph.passage_topic[pid] == chain.entity
        roles = self._value_roles(sid) if ("DATE" in expected and self.has_tail) else {}
        for label, fit, text, eid, packed in self._answers(sid, expected, banned, include_topic=advance):
            value, span = packed if packed is not None else (None, None)
            effective = total
            local_gain = gain
            if self._occupation(sid, eid, span):
                boosted = merged.copy()
                for i in self.tail_idx:
                    if self.matcher.names[i] in WORK_CUES:
                        boosted[i] = 1.0
                effective = float(boosted @ weights)
                local_gain = max(gain, effective - float(chain.covered @ weights))
            if span is not None and roles:
                role = roles.get(span)
                if role:
                    boosted = merged.copy()
                    for i in self.tail_idx:
                        if self.matcher.names[i] in ROLE_CUES[role]:
                            boosted[i] = 1.0
                    effective = float(boosted @ weights)
                    local_gain = max(gain, effective - float(chain.covered @ weights))
            occupied = effective > total
            score = base + 2.2 * effective + 0.9 * fit
            if self.has_tail and local_gain <= 1e-9 and effective < 0.99:
                score -= 0.5
            score += 0.45 * (1.0 if occupied else self._proximity(sid, eid, focus, span))
            if self.present_tense and _former.search(corpus.sentence_text(sid)):
                score -= 0.45
            if self.focus:
                hit = float(self.matcher.sentence(sid)[self.focus[0]])
                score += 1.1 * hit - 0.3
                if hit:
                    score += 0.35 * max(0.0, self._proximity(sid, eid, self.focus, span))
            if value is not None and definitional and not plan:
                score += 0.35
                if self._classifier_year(sid, span):
                    score += 0.6
            if eid is not None and stage_gender:
                score += 0.5 * stage_gender * float(graph.entity_gender[eid])
            if eid is not None:
                text = self._expand_name(sid, eid, text)
                if self.prefer_countries:
                    folded = surface_key(text)
                    if DEMONYMS.get(folded, folded) in COUNTRIES:
                        score += 0.6
            key = surface_key(text)
            if not key:
                continue
            existing = found.get(key)
            if existing is not None and existing.score >= score:
                existing.support += 1
                continue
            found[key] = Candidate(
                text=text,
                key=key,
                kind="value" if eid is None else "entity",
                label=label,
                score=score,
                chain=Chain(
                    score=score,
                    entity=chain.entity,
                    covered=merged,
                    hops=chain.hops + [Hop(sid, chain.entity, eid, gain)],
                    entities=chain.entities,
                    sentences=chain.sentences | {sid},
                    stage=nxt_stage,
                ),
                sentence=sid,
                entity=eid,
                value=value,
                support=(existing.support + 1) if existing else 1,
            )

    def _occupation(self, sid: int, eid: int | None, span) -> bool:
        if eid is None or not self.tail_idx:
            return False
        if not any(self.matcher.names[i] in WORK_CUES for i in self.tail_idx):
            return False
        text, start, _ = self._sentence_bounds(sid)
        for m_start, _ in self._spans(sid, eid):
            if _occupation_at.search(text[max(start, m_start - 40):m_start]):
                return True
        return False

    def _sentence_bounds(self, sid: int) -> tuple[str, int, int]:
        corpus = self.graph.corpus
        passage = corpus.passages[int(corpus.sent_passage[sid])]
        sent = passage.sentences[int(corpus.sent_order[sid])]
        return passage.text, sent.start, sent.end

    def _value_roles(self, sid: int) -> dict[tuple[int, int], str]:
        text, start, end = self._sentence_bounds(sid)
        sentence = text[start:end]
        roles: dict[tuple[int, int], str] = {}
        values = [v for v in self.graph.corpus.sent_values[sid] if v[3] == "DATE"]
        for match in _lifespan_parens.finditer(sentence):
            inside = [v for v in values if match.start() <= int(v[0]) and int(v[1]) <= match.end()]
            dash = match.group(0).find("–") if "–" in match.group(0) else match.group(0).find("-")
            for v in inside:
                rel = int(v[0]) - match.start()
                roles[(int(v[0]), int(v[1]))] = "birth" if dash < 0 or rel < dash else "death"
            if len(inside) == 1 and dash < 0:
                low = match.group(0).lower()
                if "died" in low or "d." in low:
                    roles[(int(inside[0][0]), int(inside[0][1]))] = "death"
        return roles

    def _classifier_year(self, sid: int, span: tuple[int, int] | None) -> bool:
        if span is None:
            return False
        text, start, _ = self._sentence_bounds(sid)
        after = text[start + span[1]:start + span[1] + 40].lower()
        words = _words.findall(after)[:4]
        return any(w in CLASSIFIER_WORDS for w in words)

    def _reversed(self, sid: int, source: int, stage: int) -> bool:
        if stage >= len(self.plan_words):
            return False
        words = set(self.plan_words[stage])
        parent = bool(words & {"father", "mother", "parent"})
        child = bool(words & {"son", "daughter", "child"})
        if not (parent or child):
            return False
        text, start, end = self._sentence_bounds(sid)
        spans = self._spans(sid, source)
        if not spans:
            return False
        sentence = text[start:end].lower()
        for s_start, _ in spans:
            rel = s_start - start
            window = sentence[max(0, rel - 30):rel]
            if parent and _child_of.search(window):
                return True
            if child and _parent_of.search(window):
                return True
        return False

    def _expand_name(self, sid: int, eid: int, text: str) -> str:
        spans = self._spans(sid, eid)
        if not spans:
            return text
        passage_text, start, end = self._sentence_bounds(sid)
        m_start, m_end = spans[0]
        tail = passage_text[m_end:end]
        match = _of_extension.match(tail)
        if match and len(match.group(0)) < 40:
            return passage_text[m_start:m_end] + match.group(0).rstrip()
        return text

    def _answers(self, sid: int, expected: frozenset[str], banned: set[int], include_topic: bool = False):
        graph = self.graph
        res = graph.res
        corpus = graph.corpus
        if "DATE" in expected or "NUMBER" in expected:
            for value in corpus.sent_values[sid]:
                label, text = value[3], value[2]
                if "DATE" in expected and label == "DATE":
                    parsed = parse_date(text)
                    if parsed is not None:
                        yield label, 1.0, text, None, (parsed, (int(value[0]), int(value[1])))
                elif "NUMBER" in expected and label in NUMBER_LABELS:
                    parsed = parse_number(text)
                    if parsed is not None:
                        yield label, 1.0, text, None, (parsed, (int(value[0]), int(value[1])))
            return
        if "BOOL" in expected:
            return
        ents = graph.sentence_entities(sid).tolist()
        topic_fit = 1.0
        if include_topic:
            topic = int(graph.passage_topic[int(corpus.sent_passage[sid])])
            if topic >= 0 and topic not in ents:
                ents.append(topic)
                topic_fit = 0.85
        for eid in ents:
            if eid in banned:
                continue
            cls = res.entity_class[eid]
            if expected:
                fit = max(TYPE_FIT.get(t, {}).get(cls, 0.0) for t in expected)
                if fit <= 0:
                    continue
            else:
                fit = 0.5 if res.entity_kind[eid] == "named" else 0.35
            if eid == ents[-1] and topic_fit < 1.0:
                fit *= topic_fit
            yield res.entity_label[eid], fit, res.entity_name[eid], eid, None

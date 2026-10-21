from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from waymark.graph.corpus import Corpus, compatible
from waymark.graph.landmarks import ContextField, build_context_field, row_normalize
from waymark.text import CONNECTORS, acronym_of, looks_like_acronym, name_core

log = logging.getLogger(__name__)

ORG_TAILS = frozenset({
    "inc", "corp", "corporation", "company", "co", "ltd", "llc", "plc", "group", "labs", "lab", "laboratory",
    "laboratories", "holdings", "technologies", "systems", "foundation", "institute", "partners", "studios",
    "records", "press", "media", "airlines", "airways", "motors", "bank", "fc", "club",
})

_acronym_pattern = re.compile(
    r"((?:[A-Z][\w'&-]*|of|and|the|for|in|on|&)(?:\s+(?:[A-Z][\w'&-]*|of|and|the|for|in|on|&)){1,9})\s*\(\s*([A-Z][A-Za-z.]{1,9})\s*\)"
)


@dataclass
class Sense:
    sid: int
    key: str
    cls: str
    mentions: list[int] = field(default_factory=list)
    qualifier: str = ""
    anchored: bool = False


@dataclass
class Resolution:
    mention_entity: np.ndarray
    entity_name: list[str]
    entity_class: list[str]
    entity_label: list[str]
    entity_aliases: list[list[str]]
    entity_qualifier: list[str]
    entity_mentions: list[np.ndarray]
    entity_kind: list[str]
    context: ContextField
    profiles: sparse.csr_matrix
    stats: dict


def _head_tokens(key: str) -> list[str]:
    head = key.split(" of ")[0] if " of " in f" {key} " else key
    return name_core(head)


class EntityResolver:
    def __init__(
        self,
        corpus: Corpus,
        split_threshold: float = 0.05,
        local_margin: float = 0.02,
        global_link_threshold: float = 0.12,
        minority_threshold: float = 0.08,
    ):
        self.corpus = corpus
        self.split_threshold = split_threshold
        self.local_margin = local_margin
        self.global_link_threshold = global_link_threshold
        self.minority_threshold = minority_threshold
        self.field: ContextField | None = None
        self.senses: list[Sense] = []
        self.mention_sense = np.full(corpus.n_mentions, -1, dtype=np.int64)
        self.mention_doc = np.asarray([corpus.passage_doc[p] for p in corpus.sent_passage[corpus.m_sent]], dtype=object) if corpus.n_mentions else np.zeros(0, dtype=object)
        self.stats: Counter[str] = Counter()

    def resolve(self) -> Resolution:
        corpus = self.corpus
        self.field = build_context_field(corpus)
        by_key: dict[str, list[int]] = defaultdict(list)
        concept_keys: dict[str, list[int]] = defaultdict(list)
        for mid in range(corpus.n_mentions):
            if corpus.m_kind[mid] == "concept":
                concept_keys[corpus.m_key[mid]].append(mid)
            else:
                by_key[corpus.m_key[mid]].append(mid)
        long_keys = {k: v for k, v in by_key.items() if self._is_long(k, v)}
        short_keys = {k: v for k, v in by_key.items() if k not in long_keys}
        for key, mids in long_keys.items():
            self._settle_long(key, mids)
        self._absorb_extensions()
        unresolved = self._link_short_forms(short_keys)
        for key, mids in unresolved.items():
            self._settle_standalone(key, mids)
        for key, mids in concept_keys.items():
            sense = self._new_sense(key, "CONCEPT")
            for mid in mids:
                self._attach(mid, sense.sid)
        return self._finalise()

    def _is_long(self, key: str, mids: list[int]) -> bool:
        if any(self.corpus.m_kind[m] == "title" for m in mids):
            return True
        core = name_core(key)
        if len(core) >= 2:
            return True
        return bool(core) and any(ch.isdigit() for ch in core[0])

    def _new_sense(self, key: str, cls: str, qualifier: str = "", anchored: bool = False) -> Sense:
        sense = Sense(sid=len(self.senses), key=key, cls=cls, qualifier=qualifier, anchored=anchored)
        self.senses.append(sense)
        return sense

    def _attach(self, mid: int, sid: int) -> None:
        self.senses[sid].mentions.append(mid)
        self.mention_sense[mid] = sid

    def _majority(self, mids: list[int]) -> str:
        counts = Counter(self.corpus.m_class[m] for m in mids if self.corpus.m_class[m] != "MISC")
        return counts.most_common(1)[0][0] if counts else "MISC"

    def _settle_long(self, key: str, mids: list[int]) -> None:
        corpus = self.corpus
        anchors: dict[str, list[int]] = defaultdict(list)
        for m in mids:
            if corpus.m_kind[m] == "title":
                anchors[corpus.m_qualifier[m]].append(m)
        if len(anchors) >= 2:
            self._split_by_anchors(key, mids, anchors)
            return
        if self._split_by_class(key, mids):
            return
        qualifier = next(iter(anchors)) if anchors else ""
        sense = self._new_sense(key, self._majority(mids), qualifier, anchored=bool(anchors))
        for m in mids:
            self._attach(m, sense.sid)

    def _split_by_anchors(self, key: str, mids: list[int], anchors: dict[str, list[int]]) -> None:
        corpus = self.corpus
        senses = []
        for qualifier, members in anchors.items():
            sense = self._new_sense(key, self._majority(members), qualifier, anchored=True)
            for m in members:
                self._attach(m, sense.sid)
            senses.append(sense)
        rest = [m for m in mids if corpus.m_kind[m] != "title"]
        if rest:
            stacked = sparse.vstack([self.field.centroid(corpus, s.mentions) for s in senses], format="csr")
            sims = (self.field.mention_rows(corpus, rest) @ stacked.T).toarray()
            for row, m in enumerate(rest):
                self._attach(m, senses[int(np.argmax(sims[row]))].sid)
        self.stats["anchored_splits"] += len(senses) - 1

    def _split_by_class(self, key: str, mids: list[int]) -> bool:
        corpus = self.corpus
        groups: dict[str, list[int]] = defaultdict(list)
        for m in mids:
            groups[corpus.m_class[m]].append(m)
        strong = [c for c, ms in groups.items() if c != "MISC" and len(ms) >= 2 and len(ms) >= 0.25 * len(mids)]
        split = [c for c in strong if all(not compatible(c, o) for o in strong if o != c)]
        if len(split) < 2:
            return False
        vectors = {c: self.field.centroid(corpus, groups[c]) for c in split}
        sims = [float(vectors[a].multiply(vectors[b]).sum()) for i, a in enumerate(split) for b in split[i + 1:]]
        if max(sims) >= 0.35:
            return False
        senses = {c: self._new_sense(key, c) for c in split}
        stacked = sparse.vstack([vectors[c] for c in split], format="csr")
        stray = [m for m in mids if corpus.m_class[m] not in senses]
        for m in mids:
            if corpus.m_class[m] in senses:
                self._attach(m, senses[corpus.m_class[m]].sid)
        if stray:
            sims_rows = (self.field.mention_rows(corpus, stray) @ stacked.T).toarray()
            for row, m in enumerate(stray):
                self._attach(m, senses[split[int(np.argmax(sims_rows[row]))]].sid)
        self.stats["class_splits"] += len(split) - 1
        return True

    def _absorb_extensions(self) -> None:
        by_key: dict[str, list[Sense]] = defaultdict(list)
        for s in self.senses:
            by_key[s.key].append(s)
        for s in list(self.senses):
            if " of " not in f" {s.key} " or not s.mentions:
                continue
            head = s.key.split(" of ")[0].strip()
            targets = by_key.get(head)
            if not targets or len(targets) != 1 or len(name_core(head)) < 2:
                continue
            target = targets[0]
            if target.sid == s.sid or not compatible(target.cls, s.cls) or (s.anchored and target.anchored):
                continue
            if "PERSON" not in {target.cls, s.cls}:
                continue
            for m in s.mentions:
                self.mention_sense[m] = target.sid
            target.mentions.extend(s.mentions)
            s.mentions = []
            self.stats["extensions_absorbed"] += 1

    def _profiles(self) -> sparse.csr_matrix:
        corpus = self.corpus
        rows, cols = [], []
        for s in self.senses:
            for m in s.mentions:
                rows.append(s.sid)
                cols.append(int(corpus.m_sent[m]))
        incidence = sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(len(self.senses), corpus.n_sentences),
        )
        return row_normalize(sparse.csr_matrix(incidence @ self.field.sentence_vectors))

    def _name_indexes(self) -> tuple[dict[str, list[int]], dict[str, list[int]], dict[str, list[int]]]:
        person: dict[str, list[int]] = defaultdict(list)
        org: dict[str, list[int]] = defaultdict(list)
        acro: dict[str, list[int]] = defaultdict(list)
        for s in self.senses:
            if not s.mentions:
                continue
            core = name_core(s.key)
            if len(core) < 2:
                continue
            if s.cls in {"PERSON", "MISC"}:
                head = _head_tokens(s.key)
                if head and all(t.isalpha() for t in head):
                    if len(head) >= 2 and " of " not in f" {s.key} ":
                        person[head[-1]].append(s.sid)
                    if s.cls == "PERSON":
                        person[head[0]].append(s.sid)
            if s.cls in {"ORG", "MISC"} and core[0] not in CONNECTORS and all(t in ORG_TAILS for t in core[1:]):
                org[core[0]].append(s.sid)
            if s.cls in {"ORG", "PLACE", "WORK", "EVENT", "MISC"}:
                letters = acronym_of(core)
                if len(letters) >= 2:
                    acro[letters].append(s.sid)
        return person, org, acro

    def _acronym_table(self) -> dict[tuple[str, str], str]:
        table: dict[tuple[str, str], str] = {}
        for passage in self.corpus.passages:
            if "(" not in passage.text:
                continue
            for match in _acronym_pattern.finditer(passage.text):
                long_tokens = name_core(match.group(1))
                short = match.group(2).replace(".", "").lower()
                if len(short) < 2:
                    continue
                trimmed = list(long_tokens)
                while trimmed and len(acronym_of(trimmed)) > len(short):
                    trimmed = trimmed[1:]
                if acronym_of(trimmed) == short:
                    table[(passage.doc_id, short)] = " ".join(trimmed)
        return table

    def _link_short_forms(self, short_keys: dict[str, list[int]]) -> dict[str, list[int]]:
        corpus = self.corpus
        profiles = self._profiles()
        person, org, acro = self._name_indexes()
        expansions = self._acronym_table()
        key_to_sense: dict[str, int] = {}
        for s in self.senses:
            if s.mentions and s.key not in key_to_sense:
                key_to_sense[s.key] = s.sid
        doc_sents: list[dict[str, np.ndarray]] = []
        for s in self.senses:
            grouped: dict[str, list[int]] = defaultdict(list)
            for m in s.mentions:
                grouped[self.mention_doc[m]].append(int(corpus.m_sent[m]))
            doc_sents.append({d: np.unique(np.asarray(v, dtype=np.int64)) for d, v in grouped.items()})

        pairs_m: list[int] = []
        pairs_c: list[int] = []
        unresolved: dict[str, list[int]] = defaultdict(list)
        for key, mids in short_keys.items():
            core = name_core(key)
            if not core:
                unresolved[key].extend(mids)
                continue
            token = core[-1]
            for mid in mids:
                cls = corpus.m_class[mid]
                surface = corpus.m_surface[mid]
                if looks_like_acronym(surface):
                    expansion = expansions.get((self.mention_doc[mid], token))
                    if expansion and expansion in key_to_sense:
                        self._attach(mid, key_to_sense[expansion])
                        self.stats["acronyms_expanded"] += 1
                        continue
                cands: set[int] = set()
                if cls in {"PERSON", "MISC"}:
                    cands.update(person.get(token, ()))
                if cls in {"ORG", "MISC"} and not corpus.m_modifier[mid]:
                    cands.update(org.get(token, ()))
                if looks_like_acronym(surface):
                    cands.update(acro.get(token, ()))
                cands = {c for c in cands if compatible(cls, self.senses[c].cls)}
                if not cands:
                    unresolved[key].append(mid)
                    continue
                for c in cands:
                    pairs_m.append(mid)
                    pairs_c.append(c)
        if not pairs_m:
            return unresolved
        pm = np.asarray(pairs_m, dtype=np.int64)
        pc = np.asarray(pairs_c, dtype=np.int64)
        left = self.field.sentence_vectors[corpus.m_sent[pm]]
        right = profiles[pc]
        sims = np.asarray(left.multiply(right).sum(axis=1)).ravel()
        by_mention: dict[int, list[tuple[float, bool, int]]] = defaultdict(list)
        for mid, cand, sim in zip(pm.tolist(), pc.tolist(), sims.tolist()):
            doc = self.mention_doc[mid]
            sents = doc_sents[cand].get(doc)
            if sents is not None and len(sents):
                sent = int(corpus.m_sent[mid])
                pos = int(np.searchsorted(sents, sent))
                near = [abs(int(sents[i]) - sent) for i in (pos - 1, pos) if 0 <= i < len(sents)]
                gap = min(near) if near else 99
                span = corpus.passage_sents[int(corpus.sent_passage[sent])]
                lo = int(np.searchsorted(sents, span.start))
                same_passage = lo < len(sents) and int(sents[lo]) < span.stop
                score = sim + (0.25 if same_passage else 0.0) + 0.15 / (1.0 + gap / 4.0)
                by_mention[mid].append((score, True, cand))
            else:
                by_mention[mid].append((sim, False, cand))
        for mid, options in by_mention.items():
            target = self._choose(mid, options)
            if target is None:
                unresolved[corpus.m_key[mid]].append(mid)
            else:
                self._attach(mid, target)
                self.stats["short_linked"] += 1
        return unresolved

    def _choose(self, mid: int, options: list[tuple[float, bool, int]]) -> int | None:
        local = sorted(((s, c) for s, is_local, c in options if is_local), reverse=True)
        if local:
            if len(local) == 1:
                return local[0][1]
            if local[0][0] - local[1][0] >= self.local_margin:
                self.stats["context_disambiguated"] += 1
                return local[0][1]
            self.stats["left_ambiguous"] += 1
            return None
        remote = sorted(((s, c) for s, _, c in options), reverse=True)
        best, cand = remote[0]
        runner = remote[1][0] if len(remote) > 1 else 0.0
        if best >= self.global_link_threshold and best - runner >= 0.03:
            return cand
        cls = self.corpus.m_class[mid]
        if len(remote) == 1 and cls == self.senses[cand].cls and cls == "PERSON" and best >= self.global_link_threshold * 0.6:
            return cand
        return None

    def _settle_standalone(self, key: str, mids: list[int]) -> None:
        corpus = self.corpus
        docs: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for m in mids:
            docs[self.mention_doc[m]][corpus.m_class[m]].append(m)
        units: list[tuple[str, list[int]]] = []
        for classes in docs.values():
            named = {c: ms for c, ms in classes.items() if c != "MISC"}
            misc = classes.get("MISC", [])
            if not named:
                units.append(("MISC", list(misc)))
                continue
            doc_units: list[tuple[str, list[int]]] = []
            for c in sorted(named, key=lambda c: -len(named[c])):
                for unit in doc_units:
                    if compatible(c, unit[0]):
                        unit[1].extend(named[c])
                        break
                else:
                    doc_units.append((c, list(named[c])))
            doc_units[0][1].extend(misc)
            units.extend(self._settle_minorities(doc_units))
        clusters = self._cluster_units(units)
        if len(clusters) > 1:
            self.stats["context_splits"] += len(clusters) - 1
        for cls, members in clusters:
            sense = self._new_sense(key, cls if cls != "MISC" else self._majority(members))
            for m in members:
                self._attach(m, sense.sid)

    def _settle_minorities(self, doc_units: list[tuple[str, list[int]]]) -> list[tuple[str, list[int]]]:
        if len(doc_units) < 2:
            return doc_units
        head_cls, head = doc_units[0]
        anchor = self.field.centroid(self.corpus, head)
        kept = [(head_cls, list(head))]
        for cls, members in doc_units[1:]:
            if len(members) * 2 <= len(head):
                sim = float(self.field.centroid(self.corpus, members).multiply(anchor).sum())
                if sim >= self.minority_threshold:
                    kept[0][1].extend(members)
                    self.stats["minority_relabelled"] += 1
                    continue
            kept.append((cls, members))
        return kept

    def _cluster_units(self, units: list[tuple[str, list[int]]]) -> list[tuple[str, list[int]]]:
        if len(units) == 1:
            return units
        corpus = self.corpus
        classes = [u[0] for u in units]
        rows, cols = [], []
        for i, (_, ms) in enumerate(units):
            for m in ms:
                rows.append(i)
                cols.append(int(corpus.m_sent[m]))
        incidence = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(units), corpus.n_sentences))
        vectors = row_normalize(sparse.csr_matrix(incidence @ self.field.sentence_vectors))
        families: dict[str, list[int]] = defaultdict(list)
        floating: list[int] = []
        for i, cls in enumerate(classes):
            if cls == "MISC":
                floating.append(i)
                continue
            for fam in families:
                if compatible(cls, fam):
                    families[fam].append(i)
                    break
            else:
                families[cls].append(i)
        if not families:
            return [("MISC", [m for _, ms in units for m in ms])]
        groups: list[tuple[str, list[int]]] = []
        for fam, idxs in families.items():
            if fam == "PERSON" and len(idxs) > 1:
                sims = (vectors[idxs] @ vectors[idxs].T).toarray()
                for sub in self._average_linkage(sims):
                    groups.append((fam, [idxs[k] for k in sub]))
            else:
                groups.append((fam, idxs))
        if floating:
            centroids = []
            for _, idxs in groups:
                centroids.append(row_normalize(sparse.csr_matrix(vectors[idxs].sum(axis=0))))
            stacked = sparse.vstack(centroids, format="csr")
            sims = (vectors[floating] @ stacked.T).toarray()
            for row, unit in enumerate(floating):
                groups[int(np.argmax(sims[row]))][1].append(unit)
        return [(cls, [m for u in idxs for m in units[u][1]]) for cls, idxs in groups]

    def _average_linkage(self, sims: np.ndarray) -> list[list[int]]:
        n = sims.shape[0]
        if n > 250:
            return self._leader(sims)
        members = [[i] for i in range(n)]
        sums = sims.astype(np.float64).copy()
        sizes = np.ones(n)
        alive = np.ones(n, dtype=bool)
        upper = np.triu(np.ones((n, n), dtype=bool), 1)
        while True:
            mask = upper & alive[:, None] & alive[None, :]
            if not mask.any():
                break
            avg = np.where(mask, sums / np.outer(sizes, sizes), -1.0)
            flat = int(np.argmax(avg))
            i, j = divmod(flat, n)
            if avg[i, j] < self.split_threshold:
                break
            members[i].extend(members[j])
            sums[i, :] += sums[j, :]
            sums[:, i] += sums[:, j]
            sizes[i] += sizes[j]
            alive[j] = False
        return [members[i] for i in np.where(alive)[0]]

    def _leader(self, sims: np.ndarray) -> list[list[int]]:
        leaders: list[int] = []
        assigned: dict[int, list[int]] = {}
        for i in range(sims.shape[0]):
            if leaders:
                scores = sims[i, leaders]
                best = int(np.argmax(scores))
                if scores[best] >= self.split_threshold:
                    assigned[leaders[best]].append(i)
                    continue
            leaders.append(i)
            assigned[i] = [i]
        return [assigned[lead] for lead in leaders]

    def _finalise(self) -> Resolution:
        corpus = self.corpus
        live = [s for s in self.senses if s.mentions]
        for i, s in enumerate(live):
            s.sid = i
        self.senses = live
        mention_entity = np.full(corpus.n_mentions, -1, dtype=np.int64)
        names, classes, labels, aliases, qualifiers, members, kinds = [], [], [], [], [], [], []
        for s in live:
            ms = np.asarray(sorted(s.mentions), dtype=np.int64)
            mention_entity[ms] = s.sid
            surfaces = Counter(corpus.m_surface[m] for m in ms)
            depth = len(name_core(s.key))
            preferred = Counter({sf: c for sf, c in surfaces.items() if len(name_core(sf)) >= depth})
            pool = preferred or surfaces
            names.append(max(pool.items(), key=lambda kv: (kv[1], len(kv[0])))[0])
            classes.append(s.cls)
            label_counts = Counter(corpus.m_label[m] for m in ms if corpus.m_label[m] not in {"MISC", "TITLE"})
            labels.append(label_counts.most_common(1)[0][0] if label_counts else ("CONCEPT" if s.cls == "CONCEPT" else "MISC"))
            aliases.append(sorted({corpus.m_key[m] for m in ms}))
            qualifiers.append(s.qualifier)
            members.append(ms)
            kinds.append("concept" if s.cls == "CONCEPT" else "named")
        missing = int((mention_entity < 0).sum())
        if missing:
            log.warning("%d mentions left unresolved", missing)
        stats = dict(self.stats)
        stats.update({"entities": len(live), "mentions": corpus.n_mentions})
        return Resolution(
            mention_entity=mention_entity,
            entity_name=names,
            entity_class=classes,
            entity_label=labels,
            entity_aliases=aliases,
            entity_qualifier=qualifiers,
            entity_mentions=members,
            entity_kind=kinds,
            context=self.field,
            profiles=self._profiles(),
            stats=stats,
        )

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from functools import lru_cache

HONORIFICS = frozenset({
    "mr", "mrs", "ms", "miss", "dr", "prof", "professor", "sir", "dame", "lady", "lord",
    "rev", "reverend", "fr", "father", "sister", "brother", "sheikh", "sheikha", "shaikh",
    "king", "queen", "prince", "princess", "president", "senator", "gen", "general",
    "capt", "captain", "col", "colonel", "sgt", "st", "jr", "sr", "phd", "md",
})

CONNECTORS = frozenset({"of", "and", "the", "for", "in", "on", "at", "de", "la", "le", "du", "von", "van", "al", "bin", "ibn", "&"})

ARTICLES = frozenset({"the", "a", "an"})

STOPWORDS = frozenset("""
a about above after again against all also am an and any are aren as at be because been before being below
between both but by can cannot could did do does doing down during each else ever few for from further had has
have having he her here hers herself him himself his how however i if in into is it its itself just least less
let like made make many may me might more most much must my myself neither no nor not now of off often on once
one only or other otherwise our ours ourselves out over own per perhaps please rather same shall she should since
so some such than that the their theirs them themselves then there these they this those though through thus to
too toward under until up upon us use used using very via was we were what whatever when where whereas whether
which while who whom whose why will with within without would yet you your yours yourself yourselves
get got go goes going say said says
""".split())

QUESTION_NOISE = frozenset("""
who whom whose what which when where why how much many name named tell find give show list did does do is are was
were be been the of a an document documents paper papers text according mentioned mention describe described
""".split())

_space = re.compile(r"\s+")
_nonword = re.compile(r"[^0-9a-z]+")
_possessive = re.compile(r"(?:'s|’s|'|’)$")


@lru_cache(maxsize=200_000)
def fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.lower()


def squash(text: str) -> str:
    return _space.sub(" ", text).strip()


def tokens_of(text: str) -> list[str]:
    folded = fold(text)
    folded = _possessive.sub("", folded.strip())
    return [tok for tok in _nonword.split(folded) if tok]


def surface_key(text: str) -> str:
    toks = tokens_of(text)
    while toks and toks[0] in ARTICLES:
        toks = toks[1:]
    return " ".join(toks)


def name_core(text: str) -> list[str]:
    return list(_name_core(text))


@lru_cache(maxsize=200_000)
def _name_core(text: str) -> tuple[str, ...]:
    toks = tokens_of(text)
    while toks and (toks[0] in HONORIFICS or toks[0] in ARTICLES):
        toks = toks[1:]
    while toks and toks[-1] in {"jr", "sr", "phd", "md"}:
        toks = toks[:-1]
    return tuple(toks)


def acronym_of(words: Iterable[str]) -> str:
    letters = [w[0] for w in words if w and w not in CONNECTORS]
    return "".join(letters)


def looks_like_acronym(text: str) -> bool:
    raw = text.strip().replace(".", "")
    return 2 <= len(raw) <= 8 and raw.isalpha() and sum(ch.isupper() for ch in raw) >= max(2, len(raw) - 1)


def stem_like(a: str, b: str) -> bool:
    if a == b:
        return True
    if len(a) < 4 or len(b) < 4:
        return False
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if longer.startswith(shorter) and len(shorter) >= 5:
        return True
    prefix = 0
    for x, y in zip(a, b):
        if x != y:
            break
        prefix += 1
    return prefix >= 5 and prefix >= 0.7 * len(shorter)

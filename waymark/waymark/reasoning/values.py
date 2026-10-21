from __future__ import annotations

import re

MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4, "may": 5,
    "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

_year = re.compile(r"(?<![\d.])(\d{3,4})(?![\d.])")
_bc = re.compile(r"\b(?:bc|bce|b\.c\.)\b", re.IGNORECASE)
_day = re.compile(r"(?<!\d)(\d{1,2})(?:st|nd|rd|th)?(?!\d)")
_number = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_scale = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12, "k": 1e3, "m": 1e6, "bn": 1e9}
_words = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100,
}
_lifespan = re.compile(r"\(\s*(?:born\s+)?([^()]*?\d{3,4}[^()]*?)\s*[–—-]\s*([^()]*?\d{3,4}[^()]*?)\s*\)")


def parse_date(text: str) -> float | None:
    lowered = text.lower()
    years = _year.findall(lowered)
    if not years:
        return None
    year = int(years[0])
    if _bc.search(lowered):
        year = -year
    month = 0
    for token in re.findall(r"[a-z]+", lowered):
        if token in MONTHS:
            month = MONTHS[token]
            break
    day = 0
    if month:
        stripped = _year.sub(" ", lowered)
        days = [int(d) for d in _day.findall(stripped) if 1 <= int(d) <= 31]
        if days:
            day = days[0]
    return year * 10000 + month * 100 + day


def parse_number(text: str) -> float | None:
    lowered = text.lower().replace(" ", "")
    match = _number.search(lowered)
    value: float | None = None
    if match:
        try:
            value = float(match.group(0).replace(",", ""))
        except ValueError:
            value = None
    if value is None:
        for word, num in _words.items():
            if re.search(rf"\b{word}\b", lowered):
                value = float(num)
                break
    if value is None:
        return None
    for word, mult in _scale.items():
        if re.search(rf"\b{word}\b", lowered):
            value *= mult
            break
    return value


def lifespan_years(text: str) -> float | None:
    match = _lifespan.search(text)
    if not match:
        return None
    born, died = parse_date(match.group(1)), parse_date(match.group(2))
    if born is None or died is None:
        return None
    return (died - born) / 10000.0

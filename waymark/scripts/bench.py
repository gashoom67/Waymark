from __future__ import annotations

import argparse
import collections
import json
import re
import string
import sys
import time
import urllib.request
from pathlib import Path

from waymark.graph.index import build_graph
from waymark.nlp.annotator import Annotator, PassageInput, PassageNote
from waymark.reasoning.engine import ReasoningEngine

SOURCE = "https://raw.githubusercontent.com/OSU-NLP-Group/HippoRAG/main/reproduce/dataset"
DATASETS = ("2wikimultihopqa", "hotpotqa", "musique")


def fetch(name: str, cache: Path) -> tuple[list[dict], list[dict]]:
    cache.mkdir(parents=True, exist_ok=True)
    out = []
    for suffix in ("", "_corpus"):
        path = cache / f"{name}{suffix}.json"
        if not path.exists():
            print(f"downloading {path.name}", file=sys.stderr)
            urllib.request.urlretrieve(f"{SOURCE}/{name}{suffix}.json", path)
        out.append(json.loads(path.read_text()))
    return out[0], out[1]


def annotate(name: str, corpus: list[dict], cache: Path, model: str) -> list[PassageNote]:
    path = cache / f"{name}.notes.jsonl"
    if path.exists():
        return [PassageNote.unpack(json.loads(line)) for line in path.open()]
    annotator = Annotator(model)
    items = [PassageInput(doc_id=str(i), page=1, text=d["text"], title=d["title"]) for i, d in enumerate(corpus)]
    notes = []
    started = time.perf_counter()
    with path.open("w") as fh:
        for note in annotator.annotate(items, progress=lambda done, total: print(f"\r  annotating {done}/{total}", end="", file=sys.stderr)):
            notes.append(note)
            fh.write(json.dumps(note.pack()) + "\n")
    print(f"\n  annotated in {time.perf_counter() - started:.0f}s", file=sys.stderr)
    return notes


def normalise(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def f1(pred: str, gold: str) -> float:
    p, g = normalise(pred).split(), normalise(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((collections.Counter(p) & collections.Counter(g)).values())
    if not common:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def gold_titles(q: dict) -> set[str]:
    if "supporting_facts" in q:
        return {t for t, _ in q["supporting_facts"]}
    return {p["title"] for p in q["paragraphs"] if p["is_supporting"]}


def run(name: str, limit: int, cache: Path, model: str) -> dict:
    questions, corpus = fetch(name, cache)
    questions = questions[:limit]
    notes = annotate(name, corpus, cache, model)
    started = time.perf_counter()
    graph = build_graph(notes)
    built = time.perf_counter() - started
    engine = ReasoningEngine(graph, Annotator(model, role="query"))
    totals = collections.Counter()
    started = time.perf_counter()
    for q in questions:
        result = engine.ask(q["question"])
        pred = result["answer"]["text"] if result["answer"] else ""
        answers = [q["answer"], *q.get("answer_aliases", [])]
        totals["em"] += max(float(normalise(pred) == normalise(a)) for a in answers)
        totals["f1"] += max(f1(pred, a) for a in answers)
        gold = gold_titles(q)
        titles = [corpus[p]["title"] for p in result["ranked_passages"][:5]]
        totals["r2"] += len(gold & set(titles[:2])) / len(gold)
        totals["r5"] += len(gold & set(titles[:5])) / len(gold)
    n = len(questions)
    return {
        "dataset": name,
        "questions": n,
        "passages": len(corpus),
        "graph_seconds": round(built, 1),
        "ms_per_question": round(1000 * (time.perf_counter() - started) / n, 1),
        "recall@2": round(totals["r2"] / n, 3),
        "recall@5": round(totals["r5"] / n, 3),
        "exact_match": round(totals["em"] / n, 3),
        "f1": round(totals["f1"] / n, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Waymark on the multi-hop QA sets used by HippoRAG")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--cache", type=Path, default=Path("bench"))
    parser.add_argument("--model", default="en_core_web_md")
    args = parser.parse_args()
    rows = [run(name, args.limit, args.cache, args.model) for name in args.datasets]
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

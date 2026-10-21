from __future__ import annotations

import pytest

CASES = [
    ("Which institute funds the lab where the author of Shifting Sands: A Dune Migration Survey trained?", "Al Dhafra Heritage Institute", 3),
    ("Where did the author of Shifting Sands: A Dune Migration Survey train?", "Qasr Al Sarab Field Laboratory", 2),
    ("Who chairs the institute that funds the Qasr Al Sarab Field Laboratory?", "Fatima Al Hameli", 2),
    ("Who funds the Qasr Al Sarab Field Laboratory?", "Al Dhafra Heritage Institute", 1),
    ("When was the lab where Lin Chen trained established?", "2004", 2),
    ("Where was Lin Chen born?", "Kunming", 1),
    ("Where does Wei Chen work?", "Abu Dhabi Investment Office", 1),
    ("Where did Wei Chen previously work?", "Gulf Monetary Bureau", 1),
    ("Who founded the company where Jordan Hale works?", "Rashid Al Suwaidi", 2),
    ("Who installed the weather mast at the Liwa Oasis Research Station?", "Mercury Labs", 1),
    ("Who founded the company that installed the station's weather mast?", "Rashid Al Suwaidi", 2),
    ("Who was the first director of the station whose readings calibrated the probes in the dune survey?", "Salem Al Mazrouei", 2),
]


@pytest.mark.parametrize("question,expected,min_hops", CASES)
def test_multi_hop_answers(engine, question, expected, min_hops):
    result = engine.ask(question)
    assert result["answer"] is not None, question
    assert result["answer"]["text"] == expected
    assert len(result["chains"][0]["hops"]) >= min_hops
    for hop in result["chains"][0]["hops"]:
        assert hop["sentence"]["text"]
        assert hop["sentence"]["page"] >= 1


def test_comparison_uses_dates(engine):
    result = engine.ask("Which was founded earlier, the Liwa Oasis Research Station or the Al Dhafra Heritage Institute?")
    assert result["mode"] == "choose"
    assert result["answer"]["text"] == "Al Dhafra Heritage Institute"


def test_connection_path_crosses_documents(engine):
    result = engine.ask("How is Jordan Hale connected to the Al Dhafra Heritage Institute?")
    assert result["mode"] == "connect"
    names = [h["to"]["name"] for h in result["chains"][0]["hops"]]
    assert names[-1] == "Al Dhafra Heritage Institute"
    assert "Lin Chen" in names
    docs = {h["sentence"]["doc"] for h in result["chains"][0]["hops"]}
    assert len(docs) >= 2


def test_fact_read_from_image_is_used(engine):
    result = engine.ask("Who installed the weather mast at the Liwa Oasis Research Station?")
    origins = {h["sentence"]["origin"] for h in result["chains"][0]["hops"]}
    assert "ocr" in origins


def test_unknown_question_does_not_crash(engine):
    result = engine.ask("What colour is the moon over Antarctica?")
    assert "answer" in result and "chains" in result


def test_queries_are_fast(engine):
    result = engine.ask("Which institute funds the lab where the author of Shifting Sands: A Dune Migration Survey trained?")
    assert result["elapsed_ms"] < 1500


@pytest.mark.parametrize("question,verdict", [
    ("Is Lin Chen a hydrologist?", "yes"),
    ("Is Lin Chen an economist?", "no"),
    ("Is Wei Chen an economist?", "yes"),
])
def test_yes_no_checks(engine, question, verdict):
    result = engine.ask(question)
    assert result["mode"] == "verify"
    assert result["answer"]["text"] == verdict


@pytest.mark.parametrize("question", ["??", "and or but", "12345 67890"])
def test_noise_gets_no_answer(engine, question):
    assert engine.ask(question)["answer"] is None

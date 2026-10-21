from __future__ import annotations

from waymark.text import surface_key


def _entities_named(graph, alias: str) -> list[int]:
    return list(graph.alias_index.get(surface_key(alias), []))


def _doc_entities(graph, doc_id: str, key: str) -> set[int]:
    corpus = graph.corpus
    found = set()
    for mid in range(corpus.n_mentions):
        if corpus.m_key[mid] == key and corpus.mention_doc(mid) == doc_id:
            found.add(int(graph.res.mention_entity[mid]))
    return found


def test_two_chens_stay_apart(graph):
    lin = _entities_named(graph, "Lin Chen")
    wei = _entities_named(graph, "Wei Chen")
    assert len(lin) == 1 and len(wei) == 1
    assert lin[0] != wei[0]


def test_bare_surname_follows_local_context(graph):
    lin = _entities_named(graph, "Lin Chen")[0]
    wei = _entities_named(graph, "Wei Chen")[0]
    assert _doc_entities(graph, "investment_note", "chen") == {wei}
    assert _doc_entities(graph, "shifting_sands_survey", "chen") == {lin}
    assert _doc_entities(graph, "chen_biography", "chen") == {lin}


def test_country_and_person_named_jordan_are_distinct(graph):
    hale = _entities_named(graph, "Jordan Hale")[0]
    country = _doc_entities(graph, "jordan_water_brief", "jordan")
    assert country and hale not in country
    assert all(graph.res.entity_class[e] == "PLACE" for e in country)


def test_element_is_not_merged_into_company(graph):
    labs = _entities_named(graph, "Mercury Labs")[0]
    contamination = _doc_entities(graph, "mercury_contamination", "mercury")
    assert labs not in contamination


def test_pronouns_link_back_to_their_subject(graph):
    lin = _entities_named(graph, "Lin Chen")[0]
    corpus = graph.corpus
    born = [sid for sid in range(corpus.n_sentences) if "born in Kunming" in corpus.sentence_text(sid)]
    assert born
    assert lin in set(graph.sentence_entities(born[0]).tolist())

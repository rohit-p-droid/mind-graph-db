"""Unit tests for NLPModel implementations and semantic extraction."""

from mind_graph_db.nlp import RuleBasedNLPModel, get_nlp_model


def test_rule_based_nlp_entities() -> None:
    model = RuleBasedNLPModel()
    text = "Python is used by Google and OpenAI to build artificial intelligence applications."

    entities = model.extract_entities(text)
    names = [e.name for e in entities]

    assert "Python" in names
    assert "Google" in names or "OpenAI" in names


def test_rule_based_nlp_concepts() -> None:
    model = RuleBasedNLPModel()
    text = "Vector databases enable high performance similarity searches across unstructured text."

    concepts = model.extract_concepts(text)
    assert len(concepts) > 0
    assert "vector" in concepts or "databases" in concepts or "similarity" in concepts


def test_rule_based_nlp_facts() -> None:
    model = RuleBasedNLPModel()
    text = "Mind Graph DB provides hybrid vector search."

    facts = model.extract_facts(text)
    assert len(facts) >= 1
    assert facts[0].subject == "Mind Graph DB"
    assert facts[0].predicate == "provides"


def test_rule_based_nlp_relationship_candidates() -> None:
    model = RuleBasedNLPModel()
    text = "Python is integrated with SQLite to store structured graph data."

    candidates = model.extract_relationship_candidates(text)
    assert len(candidates) >= 1
    assert candidates[0].source_entity.name == "Python"
    assert candidates[0].target_entity.name == "SQLite"


def test_nlp_factory() -> None:
    model = get_nlp_model(fallback_to_rule_based=True)
    res = model.analyze("Test document for factory NLP model")
    assert res is not None
    assert isinstance(res.concepts, list)

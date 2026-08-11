"""Entity extraction, resolution, and ground-truth evaluation tests."""

from pathlib import Path
from mind_graph_db.nlp import RuleBasedNLPModel
from mind_graph_db.testing import SyntheticDataGenerator


def test_entity_resolution_and_canonicalization() -> None:
    nlp = RuleBasedNLPModel()

    # Variation text testing aliases and normalization
    text = "Machine learning techniques and ML process data using deep learning and vector database storage."
    entities = nlp.extract_entities(text)
    names = [e.name for e in entities]

    assert "Machine Learning" in names  # Normalized from Machine learning techniques / ML
    assert "Deep Learning" in names
    assert "Vector Databases" in names  # Normalized from vector database


def test_ground_truth_extraction_precision_recall() -> None:
    nlp = RuleBasedNLPModel()
    generator = SyntheticDataGenerator(seed=300)
    corpus = generator.generate_corpus(profile="quick")

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for gt_doc in corpus.documents:
        extracted = nlp.extract_entities(gt_doc.text)
        extracted_names = {e.name.lower() for e in extracted}
        expected_names = {e.canonical_name.lower() for e in gt_doc.expected_entities}

        for name in extracted_names:
            if name in expected_names:
                true_positives += 1
            else:
                false_positives += 1

        for name in expected_names:
            if name not in extracted_names:
                false_negatives += 1

    precision = true_positives / max(1, (true_positives + false_positives))
    recall = true_positives / max(1, (true_positives + false_negatives))

    # Assert high precision and recall thresholds
    assert precision >= 0.70, f"Precision {precision:.2f} below target 0.70"
    assert recall >= 0.70, f"Recall {recall:.2f} below target 0.70"

"""spaCy open-source pretrained NLP model implementation."""

from typing import List, Optional

from mind_graph_db.config.settings import get_settings
from mind_graph_db.core.types import (
    Entity,
    Fact,
    Relationship,
    RelationshipCandidate,
)
from mind_graph_db.interfaces.nlp import NLPModel


class SpacyNLPModel(NLPModel):
    """NLPModel implementation using open-source spaCy pretrained pipelines."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        """Initialize spaCy NLP model pipeline.

        Args:
            model_name: Name of spaCy model (e.g., 'en_core_web_sm'). Defaults to Settings.
        """
        try:
            import spacy

        except ImportError as err:
            raise ImportError(
                "spaCy library is not installed. Please install it using `pip install spacy`."
            ) from err

        if model_name is None:
            settings = get_settings()
            model_name = settings.nlp_model_name

        self.model_name = model_name
        try:
            self._nlp = spacy.load(self.model_name)
        except Exception:
            # Fallback auto-download attempt or raise clean exception
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except Exception as err:
                raise RuntimeError(
                    f"Failed to load spaCy model '{self.model_name}'. "
                    "Ensure model is downloaded via `python -m spacy download en_core_web_sm`."
                ) from err

    def extract_entities(self, text: str) -> List[Entity]:
        """Extract Named Entities using spaCy NER component."""
        doc = self._nlp(text)
        entities: List[Entity] = []
        seen: set[str] = set()

        for ent in doc.ents:
            clean_name = ent.text.strip()
            if clean_name not in seen and len(clean_name) > 1:
                seen.add(clean_name)
                entities.append(
                    Entity(
                        name=clean_name,
                        type=ent.label_,
                        metadata={
                            "start_char": ent.start_char,
                            "end_char": ent.end_char,
                        },
                    )
                )
        return entities

    def extract_concepts(self, text: str) -> List[str]:
        """Extract noun chunks and key concept phrases."""
        doc = self._nlp(text)
        concepts: List[str] = []
        seen: set[str] = set()

        for chunk in doc.noun_chunks:
            clean_chunk = chunk.text.strip().lower()
            if clean_chunk not in seen and len(clean_chunk) > 2:
                seen.add(clean_chunk)
                concepts.append(clean_chunk)

        return concepts[:10]

    def extract_facts(self, text: str) -> List[Fact]:
        """Extract Subject-Verb-Object (SVO) triples using dependency parsing."""
        doc = self._nlp(text)
        facts: List[Fact] = []

        for token in doc:
            if token.pos_ in ("VERB", "AUX"):
                subjects = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
                objects = [w for w in token.rights if w.dep_ in ("dobj", "pobj", "attr", "acomp")]

                for subj in subjects:
                    for obj in objects:
                        facts.append(
                            Fact(
                                subject=subj.text,
                                predicate=token.lemma_,
                                object=obj.text,
                                confidence=0.9,
                            )
                        )
        return facts

    def extract_relationship_candidates(self, text: str) -> List[RelationshipCandidate]:
        """Extract entity relationship candidates based on sentence dependency structures."""
        doc = self._nlp(text)
        entities = self.extract_entities(text)
        candidates: List[RelationshipCandidate] = []

        if len(entities) < 2:
            return candidates

        for sent in doc.sents:
            sent_text = sent.text
            sent_ents = [e for e in entities if e.name in sent_text]

            if len(sent_ents) >= 2:
                for i in range(len(sent_ents) - 1):
                    src = sent_ents[i]
                    tgt = sent_ents[i + 1]
                    candidates.append(
                        RelationshipCandidate(
                            source_entity=src,
                            target_entity=tgt,
                            relation_type="CONNECTED_TO",
                            confidence=0.8,
                            evidence_text=sent_text.strip(),
                        )
                    )

        return candidates

    def extract_relationships(
        self,
        text: str,
        entities: Optional[List[Entity]] = None,
    ) -> List[Relationship]:
        """Extract relationships and return core Relationship domain objects."""
        candidates = self.extract_relationship_candidates(text)
        return [
            Relationship(
                source_id=c.source_entity.id,
                target_id=c.target_entity.id,
                relation_type=c.relation_type,
                confidence=c.confidence,
            )
            for c in candidates
        ]

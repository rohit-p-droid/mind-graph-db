"""Rule-based heuristic NLP model implementation for zero-dependency offline analysis."""

import re
from typing import List, Optional, Set

from mind_graph_db.core.types import (
    Entity,
    Fact,
    Relationship,
    RelationshipCandidate,
)
from mind_graph_db.interfaces.nlp import NLPModel

# Common English stopwords to ignore in concept extraction
STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what", "which",
    "this", "that", "these", "those", "then", "just", "so", "than", "such", "both",
    "through", "about", "for", "is", "of", "to", "in", "it", "on", "by", "be", "with",
    "are", "from", "at", "as", "into", "has", "have", "had", "will", "was", "were",
}


class RuleBasedNLPModel(NLPModel):
    """Heuristic rule-based NLP engine using regular expressions and syntax patterns."""

    def extract_entities(self, text: str) -> List[Entity]:
        """Extract capitalized named entity candidates from text."""
        entities: List[Entity] = []
        seen_names: Set[str] = set()

        # Match capitalized word sequences (2 to 4 words) or single proper nouns
        pattern = r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\b"
        matches = re.findall(pattern, text)

        for match in matches:
            match_str = match.strip()
            # Ignore single capitalized sentence starter words if in stop words or too short
            if match_str.lower() in STOP_WORDS or len(match_str) < 3:
                continue

            if match_str not in seen_names:
                seen_names.add(match_str)
                # Assign entity type heuristically
                entity_type = "CONCEPT"
                if any(w in match_str for w in ["DB", "Database", "Store", "Engine", "Python"]):
                    entity_type = "TECHNOLOGY"
                elif any(w in match_str for w in ["Corp", "Inc", "Company", "Google", "OpenAI"]):
                    entity_type = "ORGANIZATION"

                entities.append(Entity(name=match_str, type=entity_type))

        return entities

    def extract_concepts(self, text: str) -> List[str]:
        """Extract key non-stopword concept terms and technical phrases."""
        concepts: List[str] = []
        seen: Set[str] = set()

        # Extract words longer than 3 chars that are not stop words
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        for word in words:
            if word not in STOP_WORDS and word not in seen:
                seen.add(word)
                concepts.append(word)

        return concepts[:10]

    def extract_facts(self, text: str) -> List[Fact]:
        """Extract Subject-Verb-Object factual triples using pattern matching."""
        facts: List[Fact] = []
        sentences = re.split(r"[.!?]\s+", text)

        # Regex pattern matching Subject + Predicate + Object in sentences
        pattern = r"([A-Z][a-zA-Z0-9_\s]{2,20})\s+(is|uses|combines|provides|supports|creates|manages|has|contains)\s+([^.!,;]{3,40})"

        for sentence in sentences:
            match = re.search(pattern, sentence)
            if match:
                subj, pred, obj = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
                facts.append(Fact(subject=subj, predicate=pred, object=obj, confidence=0.85))

        return facts

    def extract_relationship_candidates(self, text: str) -> List[RelationshipCandidate]:
        """Extract candidate entity relationships based on sentence co-occurrence."""
        entities = self.extract_entities(text)
        candidates: List[RelationshipCandidate] = []

        if len(entities) < 2:
            return candidates

        sentences = re.split(r"[.!?]\s+", text)
        for sentence in sentences:
            present_entities = [e for e in entities if e.name in sentence]
            if len(present_entities) >= 2:
                # Pair up adjacent entities co-occurring in the sentence
                for i in range(len(present_entities) - 1):
                    src = present_entities[i]
                    tgt = present_entities[i + 1]
                    candidates.append(
                        RelationshipCandidate(
                            source_entity=src,
                            target_entity=tgt,
                            relation_type="CO_OCCURS_WITH",
                            confidence=0.75,
                            evidence_text=sentence.strip(),
                        )
                    )

        return candidates

    def extract_relationships(
        self,
        text: str,
        entities: Optional[List[Entity]] = None,
    ) -> List[Relationship]:
        """Extract relationships and convert to core Relationship models."""
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

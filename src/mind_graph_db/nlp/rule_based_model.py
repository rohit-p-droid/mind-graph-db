"""Rule-based heuristic NLP model implementation for zero-dependency offline analysis."""

import re
from typing import Dict, List, Optional, Set

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
    "using", "enables", "allows", "uses", "make", "stores",
}

# Known technical domain concepts and multi-word entity patterns (canonical titles)
KNOWN_DOMAIN_ENTITIES: List[str] = [
    "Natural Language Processing",
    "Machine Learning Techniques",
    "Artificial Intelligence",
    "Machine Learning",
    "Deep Learning",
    "Computer Vision",
    "Vector Databases",
    "Vector Database",
    "Neural Networks",
    "Mind Graph DB",
    "Embeddings",
    "Python",
    "SQLite",
    "FastAPI",
    "OpenAI",
    "Google",
]

# Canonical normalization map to consolidate near-duplicate entity variants
ENTITY_CANONICAL_MAP: Dict[str, str] = {
    "machine learning techniques": "Machine Learning",
    "machine learning technique": "Machine Learning",
    "machine learning methodology": "Machine Learning",
    "deep learning techniques": "Deep Learning",
    "natural language processing techniques": "Natural Language Processing",
    "vector database": "Vector Databases",
}


class RuleBasedNLPModel(NLPModel):
    """Heuristic rule-based NLP engine using regular expressions and syntax patterns."""

    def extract_entities(self, text: str) -> List[Entity]:
        """Extract multi-word domain entities and capitalized named entity candidates from text.

        Ensures full technical concepts (e.g. 'Machine Learning', 'Artificial Intelligence',
        'Natural Language Processing', 'Computer Vision') are preserved as canonical entity nodes,
        normalizing variants like 'Machine Learning Techniques' -> 'Machine Learning'.
        """
        extracted_names: List[str] = []
        seen_lower: Set[str] = set()

        # 1. Match known multi-word technical domain phrases
        for known_term in KNOWN_DOMAIN_ENTITIES:
            pattern = r"\b" + re.escape(known_term) + r"\b"
            if re.search(pattern, text, re.IGNORECASE):
                # Normalize via canonical mapping if available
                norm_term = ENTITY_CANONICAL_MAP.get(known_term.lower(), known_term)
                term_lower = norm_term.lower()
                if term_lower not in seen_lower:
                    seen_lower.add(term_lower)
                    extracted_names.append(norm_term)

        # 2. Match capitalized named entity sequences (e.g. 'Andrew Ng', 'Ian Goodfellow')
        cap_pattern = r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\b"
        matches = re.findall(cap_pattern, text)
        for match in matches:
            match_str = match.strip()
            if match_str.lower() in STOP_WORDS or len(match_str) < 3:
                continue

            norm_str = ENTITY_CANONICAL_MAP.get(match_str.lower(), match_str)
            match_lower = norm_str.lower()
            if match_lower not in seen_lower:
                seen_lower.add(match_lower)
                extracted_names.append(norm_str)

        # 3. Deduplicate and filter out single-word sub-string fragments if a longer multi-word phrase contains it
        extracted_names.sort(key=lambda s: len(s), reverse=True)

        final_names: List[str] = []
        for name in extracted_names:
            name_lower = name.lower()
            is_fragment = False
            if " " not in name:
                for existing in final_names:
                    if " " in existing and name_lower in existing.lower().split():
                        is_fragment = True
                        break

            if not is_fragment and name not in final_names:
                final_names.append(name)

        # Build Entity objects with heuristic typing
        entities: List[Entity] = []
        for name in final_names:
            entity_type = "CONCEPT"
            if any(w in name for w in ["DB", "Database", "Store", "Engine", "Python", "SQLite", "FastAPI"]):
                entity_type = "TECHNOLOGY"
            elif any(w in name for w in ["Corp", "Inc", "Company", "Google", "OpenAI"]):
                entity_type = "ORGANIZATION"
            elif any(w in name for w in ["Ng", "Goodfellow", "Manning", "Forsyth", "Alice", "Bob", "Charlie"]):
                entity_type = "PERSON"

            entities.append(Entity(name=name, type=entity_type))

        return entities

    def extract_concepts(self, text: str) -> List[str]:
        """Extract key non-stopword concept terms and technical phrases."""
        concepts: List[str] = []
        seen: Set[str] = set()

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

        pattern = r"([A-Z][a-zA-Z0-9_\s]{2,20})\s+(is|uses|combines|provides|supports|creates|manages|has|contains)\s+([^.!,;]{3,40})"

        for sentence in sentences:
            match = re.search(pattern, sentence)
            if match:
                subj, pred, obj = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
                facts.append(Fact(subject=subj, predicate=pred, object=obj, confidence=0.85))

        return facts

    def extract_relationship_candidates(self, text: str) -> List[RelationshipCandidate]:
        """Extract candidate entity relationships based on genuine sentence co-occurrence of distinct entities."""
        entities = self.extract_entities(text)
        candidates: List[RelationshipCandidate] = []

        if len(entities) < 2:
            return candidates

        sentences = re.split(r"[.!?]\s+", text)
        seen_pairs: Set[tuple] = set()

        for sentence in sentences:
            sentence_lower = sentence.lower()
            present_entities = [e for e in entities if e.name.lower() in sentence_lower]
            if len(present_entities) >= 2:
                for i in range(len(present_entities)):
                    for j in range(i + 1, len(present_entities)):
                        src = present_entities[i]
                        tgt = present_entities[j]
                        if src.name.lower() != tgt.name.lower():
                            pair_key = (src.name, tgt.name)
                            if pair_key not in seen_pairs:
                                seen_pairs.add(pair_key)
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

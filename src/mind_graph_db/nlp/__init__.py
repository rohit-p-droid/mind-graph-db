"""NLP models and factory functions for Mind Graph DB."""

from typing import Optional

from mind_graph_db.config.settings import get_settings
from mind_graph_db.interfaces.nlp import NLPModel
from mind_graph_db.nlp.llm_model import LLMNLPModel
from mind_graph_db.nlp.rule_based_model import RuleBasedNLPModel
from mind_graph_db.nlp.spacy_model import SpacyNLPModel


def get_nlp_model(
    model_name: Optional[str] = None,
    fallback_to_rule_based: bool = True,
) -> NLPModel:
    """Factory function to retrieve an NLPModel instance.

    Attempts to instantiate SpacyNLPModel. If spaCy or requested model is missing
    and fallback_to_rule_based is True, returns RuleBasedNLPModel.

    Args:
        model_name: Name of NLP model pipeline.
        fallback_to_rule_based: Whether to fall back to RuleBasedNLPModel if spaCy is unavailable.

    Returns:
        Instance implementing NLPModel interface.
    """
    settings = get_settings()
    target_name = model_name or settings.nlp_model_name

    try:
        return SpacyNLPModel(model_name=target_name)
    except (ImportError, RuntimeError):
        if fallback_to_rule_based:
            return RuleBasedNLPModel()
        raise


__all__ = [
    "NLPModel",
    "RuleBasedNLPModel",
    "SpacyNLPModel",
    "LLMNLPModel",
    "get_nlp_model",
]

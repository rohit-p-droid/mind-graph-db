"""Unit tests for QueryLexer and QueryParser."""

from mind_graph_db.query import QueryLexer, QueryParser, TokenType


def test_query_lexer_tokens() -> None:
    query = 'FIND documents WHERE semantic_match("Tesla") AND metadata.category = "EV"'
    lexer = QueryLexer(query)
    tokens = lexer.tokenize()

    token_types = [t.type for t in tokens if t.type != TokenType.EOF]
    assert TokenType.FIND in token_types
    assert TokenType.WHERE in token_types
    assert TokenType.SEMANTIC_MATCH in token_types
    assert TokenType.AND in token_types


def test_query_parser_full_query() -> None:
    parser = QueryParser()
    query = (
        'FIND documents '
        'WHERE semantic_match("Tesla battery production") AND metadata.category = "EV" '
        'TRAVERSE 2 HOPS WITH RELATIONSHIPS ["MENTIONS", "SIMILAR_TO"] '
        'RETURN documents, entities, relationships'
    )
    ast = parser.parse(query)

    assert ast.target == "documents"
    assert ast.semantic_query == "Tesla battery production"
    assert ast.metadata_filters == {"category": "EV"}
    assert ast.traverse_hops == 2
    assert ast.allowed_relationship_types == ["MENTIONS", "SIMILAR_TO"]
    assert "documents" in ast.return_fields
    assert "entities" in ast.return_fields
    assert "relationships" in ast.return_fields


def test_query_parser_fallback_text() -> None:
    parser = QueryParser()
    query = "Tesla battery production"
    ast = parser.parse(query)

    assert ast.semantic_query == "Tesla battery production"
    assert ast.target == "documents"

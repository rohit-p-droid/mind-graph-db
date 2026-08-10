"""Tokenizer, Lexer, and Parser for Mind Graph DB query language."""

from enum import Enum, auto
import json
import re
from typing import Any, Dict, List, Optional

from mind_graph_db.query.ast import MindQuery


class TokenType(Enum):
    FIND = auto()
    WHERE = auto()
    TRAVERSE = auto()
    HOPS = auto()
    WITH = auto()
    RELATIONSHIPS = auto()
    RETURN = auto()
    AND = auto()
    SEMANTIC_MATCH = auto()
    IDENTIFIER = auto()
    STRING = auto()
    NUMBER = auto()
    EQUALS = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    DOT = auto()
    EOF = auto()


class Token:
    """Token representation."""

    def __init__(self, type_: TokenType, value: str) -> None:
        self.type = type_
        self.value = value

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {repr(self.value)})"


KEYWORDS = {
    "FIND": TokenType.FIND,
    "WHERE": TokenType.WHERE,
    "TRAVERSE": TokenType.TRAVERSE,
    "HOPS": TokenType.HOPS,
    "WITH": TokenType.WITH,
    "RELATIONSHIPS": TokenType.RELATIONSHIPS,
    "RETURN": TokenType.RETURN,
    "AND": TokenType.AND,
    "semantic_match": TokenType.SEMANTIC_MATCH,
}


class QueryLexer:
    """Lexical analyzer for Mind Graph DB query string."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []
        while self.pos < len(self.text):
            char = self.text[self.pos]

            if char.isspace():
                self.pos += 1
                continue

            if char == '"' or char == "'":
                tokens.append(self._read_string(char))
                continue

            if char.isdigit():
                tokens.append(self._read_number())
                continue

            if char.isalpha() or char == "_":
                tokens.append(self._read_identifier())
                continue

            if char == "=":
                tokens.append(Token(TokenType.EQUALS, "="))
                self.pos += 1
                continue

            if char == "(":
                tokens.append(Token(TokenType.LPAREN, "("))
                self.pos += 1
                continue

            if char == ")":
                tokens.append(Token(TokenType.RPAREN, ")"))
                self.pos += 1
                continue

            if char == "[":
                tokens.append(Token(TokenType.LBRACKET, "["))
                self.pos += 1
                continue

            if char == "]":
                tokens.append(Token(TokenType.RBRACKET, "]"))
                self.pos += 1
                continue

            if char == ",":
                tokens.append(Token(TokenType.COMMA, ","))
                self.pos += 1
                continue

            if char == ".":
                tokens.append(Token(TokenType.DOT, "."))
                self.pos += 1
                continue

            self.pos += 1

        tokens.append(Token(TokenType.EOF, ""))
        return tokens

    def _read_string(self, quote: str) -> Token:
        self.pos += 1  # Skip open quote
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] != quote:
            self.pos += 1
        val = self.text[start : self.pos]
        if self.pos < len(self.text):
            self.pos += 1  # Skip close quote
        return Token(TokenType.STRING, val)

    def _read_number(self) -> Token:
        start = self.pos
        while self.pos < len(self.text) and (self.text[self.pos].isdigit() or self.text[self.pos] == "."):
            self.pos += 1
        val = self.text[start : self.pos]
        return Token(TokenType.NUMBER, val)

    def _read_identifier(self) -> Token:
        start = self.pos
        while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] == "_"):
            self.pos += 1
        val = self.text[start : self.pos]
        token_type = KEYWORDS.get(val, TokenType.IDENTIFIER)
        return Token(token_type, val)


class QueryParser:
    """Parser creating MindQuery AST from Mind Graph DB query string."""

    def parse(self, query_text: str) -> MindQuery:
        """Parse input query string into MindQuery AST.

        If string does not start with 'FIND', treats as plain-text semantic query fallback.
        """
        trimmed = query_text.strip()
        if not re.match(r"^FIND\b", trimmed, re.IGNORECASE):
            # Fallback plain text query
            return MindQuery(semantic_query=trimmed)

        lexer = QueryLexer(trimmed)
        tokens = lexer.tokenize()
        idx = 0

        target = "documents"
        semantic_query: Optional[str] = None
        metadata_filters: Dict[str, Any] = {}
        traverse_hops = 0
        allowed_rels: Optional[List[str]] = None
        return_fields: List[str] = ["documents", "entities", "relationships", "scores"]

        # Parse FIND target
        if tokens[idx].type == TokenType.FIND:
            idx += 1
            if tokens[idx].type == TokenType.IDENTIFIER:
                target = tokens[idx].value.lower()
                idx += 1

        # Parse WHERE
        if idx < len(tokens) and tokens[idx].type == TokenType.WHERE:
            idx += 1
            while idx < len(tokens) and tokens[idx].type not in (TokenType.TRAVERSE, TokenType.RETURN, TokenType.EOF):
                if tokens[idx].type == TokenType.SEMANTIC_MATCH:
                    idx += 1
                    if tokens[idx].type == TokenType.LPAREN:
                        idx += 1
                    if tokens[idx].type == TokenType.STRING:
                        semantic_query = tokens[idx].value
                        idx += 1
                    if tokens[idx].type == TokenType.RPAREN:
                        idx += 1
                elif tokens[idx].type in (TokenType.IDENTIFIER, TokenType.DOT):
                    key = ""
                    if tokens[idx].value.lower() == "metadata" and idx + 1 < len(tokens) and tokens[idx + 1].type == TokenType.DOT:
                        idx += 2
                    if tokens[idx].type == TokenType.IDENTIFIER:
                        key = tokens[idx].value
                        idx += 1
                    if idx < len(tokens) and tokens[idx].type == TokenType.EQUALS:
                        idx += 1
                    val: Any = None
                    if idx < len(tokens) and tokens[idx].type == TokenType.STRING:
                        val = tokens[idx].value
                        idx += 1
                    elif idx < len(tokens) and tokens[idx].type == TokenType.NUMBER:
                        num_str = tokens[idx].value
                        val = float(num_str) if "." in num_str else int(num_str)
                        idx += 1
                    if key:
                        metadata_filters[key] = val
                elif tokens[idx].type == TokenType.AND:
                    idx += 1
                else:
                    idx += 1

        # Parse TRAVERSE
        if idx < len(tokens) and tokens[idx].type == TokenType.TRAVERSE:
            idx += 1
            if tokens[idx].type == TokenType.NUMBER:
                traverse_hops = int(tokens[idx].value)
                idx += 1
            if tokens[idx].type == TokenType.HOPS:
                idx += 1

            if idx < len(tokens) and tokens[idx].type == TokenType.WITH:
                idx += 1
                if idx < len(tokens) and tokens[idx].type == TokenType.RELATIONSHIPS:
                    idx += 1
                if idx < len(tokens) and tokens[idx].type == TokenType.LBRACKET:
                    idx += 1
                    allowed_rels = []
                    while idx < len(tokens) and tokens[idx].type != TokenType.RBRACKET:
                        if tokens[idx].type == TokenType.STRING:
                            allowed_rels.append(tokens[idx].value)
                        idx += 1
                    if idx < len(tokens) and tokens[idx].type == TokenType.RBRACKET:
                        idx += 1

        # Parse RETURN
        if idx < len(tokens) and tokens[idx].type == TokenType.RETURN:
            idx += 1
            return_fields = []
            while idx < len(tokens) and tokens[idx].type != TokenType.EOF:
                if tokens[idx].type == TokenType.IDENTIFIER:
                    return_fields.append(tokens[idx].value.lower())
                idx += 1

        return MindQuery(
            target=target,
            semantic_query=semantic_query,
            metadata_filters=metadata_filters,
            traverse_hops=traverse_hops,
            allowed_relationship_types=allowed_rels,
            return_fields=return_fields if return_fields else ["documents", "entities", "relationships", "scores"],
        )

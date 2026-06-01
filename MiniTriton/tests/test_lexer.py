"""
Test suite para el lexer Mini-Triton.
Ejecutar: pytest tests/test_lexer.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lexer import (
    tokenize, Token,
    AT, COLON, COMMA, DOT, EQUALS, EOF,
    ID, LBRACE, LPAREN, MINUS, NUMBER,
    PLUS, RBRACE, RPAREN, SEMICOLON, SLASH, STAR,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def types(source: str) -> list[str]:
    """Devuelve solo los tipos de token (sin EOF)."""
    return [t.type for t in tokenize(source) if t.type != EOF]


def vals(source: str) -> list[str]:
    """Devuelve solo los valores de token (sin EOF)."""
    return [t.value for t in tokenize(source) if t.type != EOF]


# ── Tokens individuales ────────────────────────────────────────────────────────

class TestSingleTokens:
    def test_at(self):
        assert types("@") == [AT]

    def test_dot(self):
        assert types(".") == [DOT]

    def test_colon(self):
        assert types(":") == [COLON]

    def test_comma(self):
        assert types(",") == [COMMA]

    def test_semicolon(self):
        assert types(";") == [SEMICOLON]

    def test_lparen(self):
        assert types("(") == [LPAREN]

    def test_rparen(self):
        assert types(")") == [RPAREN]

    def test_lbrace(self):
        assert types("{") == [LBRACE]

    def test_rbrace(self):
        assert types("}") == [RBRACE]

    def test_equals(self):
        assert types("=") == [EQUALS]

    def test_plus(self):
        assert types("+") == [PLUS]

    def test_minus(self):
        assert types("-") == [MINUS]

    def test_star(self):
        assert types("*") == [STAR]

    def test_slash(self):
        assert types("/") == [SLASH]

    def test_integer_number(self):
        toks = tokenize("42")
        assert toks[0].type == NUMBER
        assert toks[0].value == "42"

    def test_float_number(self):
        toks = tokenize("3.14")
        assert toks[0].type == NUMBER
        assert toks[0].value == "3.14"

    def test_identifier(self):
        toks = tokenize("abc_123")
        assert toks[0].type == ID
        assert toks[0].value == "abc_123"

    def test_eof_always_last(self):
        toks = tokenize("x")
        assert toks[-1].type == EOF


# ── Palabras reservadas tokenizadas como ID ───────────────────────────────────

class TestKeywordsAsID:
    @pytest.mark.parametrize("kw", ["def", "triton", "jit", "tl", "constexpr"])
    def test_keyword_is_id(self, kw):
        toks = tokenize(kw)
        assert toks[0].type == ID
        assert toks[0].value == kw


# ── Secuencias compuestas ──────────────────────────────────────────────────────

class TestSequences:
    def test_decorator(self):
        assert types("@triton.jit") == [AT, ID, DOT, ID]
        assert vals("@triton.jit") == ["@", "triton", ".", "jit"]

    def test_dotted_call(self):
        # tl.load  →  ID DOT ID  (el parser los combina, NO el lexer)
        assert types("tl.load") == [ID, DOT, ID]
        assert vals("tl.load")  == ["tl", ".", "load"]

    def test_def_keyword(self):
        assert types("def foo") == [ID, ID]
        assert vals("def foo")  == ["def", "foo"]

    def test_param_annotation(self):
        # BS: tl.constexpr
        assert types("BS: tl.constexpr") == [ID, COLON, ID, DOT, ID]

    def test_binary_expr(self):
        assert types("x + y * 2") == [ID, PLUS, ID, STAR, NUMBER]

    def test_assignment_stmt(self):
        assert types("z = x + y;") == [ID, EQUALS, ID, PLUS, ID, SEMICOLON]

    def test_full_kernel_tokens(self):
        src = "@triton.jit def add(x, y): { z = x + y; }"
        result = types(src)
        assert result[0] == AT
        assert result[-1] == RBRACE

    def test_call_with_args(self):
        assert types("foo(a, 1)") == [ID, LPAREN, ID, COMMA, NUMBER, RPAREN]


# ── Espacios y comentarios ignorados ──────────────────────────────────────────

class TestSkipped:
    def test_whitespace_ignored(self):
        assert types("  x   y  ") == [ID, ID]

    def test_newlines_ignored(self):
        assert types("x\ny") == [ID, ID]

    def test_line_comment_ignored(self):
        assert types("x # esto es un comentario\ny") == [ID, ID]

    def test_only_comment(self):
        assert types("# solo comentario") == []

    def test_empty_string(self):
        assert types("") == []


# ── Posición (línea / columna) ─────────────────────────────────────────────────

class TestPositions:
    def test_first_token_line_1_col_1(self):
        toks = tokenize("x")
        assert toks[0].line == 1
        assert toks[0].col  == 1

    def test_second_line(self):
        toks = tokenize("x\ny")
        y_tok = [t for t in toks if t.value == "y"][0]
        assert y_tok.line == 2
        assert y_tok.col  == 1

    def test_col_after_spaces(self):
        toks = tokenize("   z")
        assert toks[0].col == 4

    def test_multiline_positions(self):
        src = "@triton.jit\ndef add"
        toks = [t for t in tokenize(src) if t.type != EOF]
        # '@' en línea 1
        assert toks[0].line == 1
        # 'def' en línea 2
        def_tok = next(t for t in toks if t.value == "def")
        assert def_tok.line == 2


# ── Errores léxicos ────────────────────────────────────────────────────────────

class TestLexErrors:
    def test_illegal_char_raises(self):
        with pytest.raises(SyntaxError, match="carácter inesperado"):
            tokenize("x $ y")

    def test_illegal_char_reports_position(self):
        with pytest.raises(SyntaxError, match="línea 1"):
            tokenize("!")


# ── Archivos .tri de muestra ───────────────────────────────────────────────────

class TestTriFiles:
    """Verifica que los archivos del directorio tests/ se tokenan sin errores."""

    TRI_DIR = os.path.join(os.path.dirname(__file__))

    def _load(self, name: str) -> str:
        path = os.path.join(self.TRI_DIR, name)
        with open(path, encoding="utf-8") as f:
            return f.read()

    @pytest.mark.parametrize("filename", [
        "valid_01.tri", "valid_02.tri", "valid_03.tri", "valid_04.tri",
        "valid_05.tri", "valid_06.tri", "valid_07.tri",
        "invalid_01.tri", "invalid_02.tri", "invalid_03.tri",
        "invalid_04.tri", "invalid_05.tri", "invalid_06.tri",
    ])
    def test_tokenizes_without_lex_error(self, filename):
        """Todos los archivos son válidos léxicamente (los errores son sintácticos)."""
        src = self._load(filename)
        toks = tokenize(src)
        assert toks[-1].type == EOF
        assert len(toks) >= 1

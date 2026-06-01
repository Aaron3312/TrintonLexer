#!/usr/bin/env python3
"""
Mini-Triton Lexer
=================
Tokenizador para el lenguaje Mini-Triton (DSL simplificado, no Python real).

Uso como módulo:
    from lexer import tokenize, Token
    tokens = tokenize(source_code)

Uso como CLI:
    python lexer.py <archivo.tri>

TOKEN TYPES
-----------
AT        @
COLON     :
COMMA     ,
DOT       .
EQUALS    =
LBRACE    {
LPAREN    (
MINUS     -
NUMBER    dígitos enteros o decimales  (\\d+(\\.\\d+)?)
PLUS      +
RBRACE    }
RPAREN    )
SEMICOLON ;
SLASH     /
STAR      *
ID        identificador  ([A-Za-z_]\w*)
EOF       fin de entrada (token sintético)

Nota: 'tl.load' se produce como  ID("tl")  DOT  ID("load").
      Palabras clave como 'def' y 'jit' se tokenan como ID y el parser
      las valida por valor, no por tipo.
      Comentarios de línea (#...) y espacios son descartados.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass

# ── Token type constants ───────────────────────────────────────────────────────

AT        = "AT"
COLON     = "COLON"
COMMA     = "COMMA"
DOT       = "DOT"
EQUALS    = "EQUALS"
LBRACE    = "LBRACE"
LPAREN    = "LPAREN"
MINUS     = "MINUS"
NUMBER    = "NUMBER"
PLUS      = "PLUS"
RBRACE    = "RBRACE"
RPAREN    = "RPAREN"
SEMICOLON = "SEMICOLON"
SLASH     = "SLASH"
STAR      = "STAR"
ID        = "ID"
EOF       = "EOF"

ALL_TYPES = (
    AT, COLON, COMMA, DOT, EQUALS, LBRACE, LPAREN, MINUS,
    NUMBER, PLUS, RBRACE, RPAREN, SEMICOLON, SLASH, STAR, ID, EOF,
)


# ── Token dataclass ────────────────────────────────────────────────────────────

@dataclass
class Token:
    """Un token producido por el lexer."""
    type:  str   # uno de los TYPE constants de arriba
    value: str   # texto literal del token en el fuente
    line:  int   # línea base-1
    col:   int   # columna base-1 (-1 para EOF)

    def __repr__(self) -> str:
        return f"Token({self.type:<10} {self.value!r:<15} línea={self.line} col={self.col})"


# ── Expresión regular maestra ──────────────────────────────────────────────────

_MASTER_RE = re.compile(
    r"(?P<SKIP>[ \t\r\n]+|#[^\n]*)"   # espacios y comentarios de línea
    r"|(?P<NUMBER>\d+(?:\.\d+)?)"      # número entero o decimal
    r"|(?P<ID>[A-Za-z_]\w*)"           # identificador / palabra clave
    r"|(?P<AT>@)"
    r"|(?P<DOT>\.)"
    r"|(?P<COLON>:)"
    r"|(?P<COMMA>,)"
    r"|(?P<SEMICOLON>;)"
    r"|(?P<LPAREN>\()"
    r"|(?P<RPAREN>\))"
    r"|(?P<LBRACE>\{)"
    r"|(?P<RBRACE>\})"
    r"|(?P<EQUALS>=)"
    r"|(?P<PLUS>\+)"
    r"|(?P<MINUS>-)"
    r"|(?P<STAR>\*)"
    r"|(?P<SLASH>/)"
    r"|(?P<MISMATCH>.)",
    re.DOTALL,
)


# ── Función principal ──────────────────────────────────────────────────────────

def tokenize(source: str) -> list[Token]:
    """
    Escanea *source* y devuelve la lista completa de tokens.

    El último elemento es siempre Token(EOF, "", ...).
    Lanza SyntaxError si encuentra un carácter ilegal.
    """
    tokens: list[Token] = []
    line = 1
    line_start = 0

    for m in _MASTER_RE.finditer(source):
        kind      = m.lastgroup
        value     = m.group()
        tok_start = m.start()
        col       = tok_start - line_start + 1

        if kind == "SKIP":
            # Actualizar contador de líneas dentro del bloque ignorado
            for i, ch in enumerate(value):
                if ch == "\n":
                    line += 1
                    line_start = tok_start + i + 1
            continue

        if kind == "MISMATCH":
            raise SyntaxError(
                f"carácter inesperado {value!r} en línea {line}, col {col}"
            )

        tokens.append(Token(kind, value, line, col))

    tokens.append(Token(EOF, "", line, -1))
    return tokens


# ── CLI ────────────────────────────────────────────────────────────────────────

def _main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) != 2:
        print(f"Uso: python {sys.argv[0]} <archivo.tri>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    try:
        with open(path, encoding="utf-8") as f:
            source = f.read()
    except OSError as e:
        print(f"Error al abrir archivo: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        tokens = tokenize(source)
    except SyntaxError as e:
        print(f"Error léxico: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"{'#':<5} {'TIPO':<12} {'VALOR':<20} {'LÍNEA':<8} COL")
    print("-" * 55)
    for i, tok in enumerate(tokens):
        val_display = tok.value if tok.value else "(vacío)"
        print(f"{i:<5} {tok.type:<12} {val_display!r:<20} {tok.line:<8} {tok.col}")


if __name__ == "__main__":
    _main()

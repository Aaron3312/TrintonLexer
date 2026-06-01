#!/usr/bin/env python3
"""
Mini-Triton Parser
==================
Parser descendente recursivo LL(1) para el lenguaje Mini-Triton.

Uso como CLI:
    python mini_triton_parser.py <archivo.tri>

Salida (válido):
    VALIDO
    Program
    └── Kernel(name="add", params=[x, y])
        └── Assign
            ├── Name("z")
            └── BinaryOp("+")
                ├── Name("x")
                └── Name("y")

Salida (inválido):
    INVALIDO: <mensaje> en línea N, col M
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from typing import List, Optional

# ── Importar lexer (mismo directorio) ─────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lexer import (
    tokenize, Token,
    AT, COLON, COMMA, DOT, EQUALS, EOF,
    ID, LBRACE, LPAREN, MINUS, NUMBER,
    PLUS, RBRACE, RPAREN, SEMICOLON, SLASH, STAR,
)

# ── Nodos del AST ──────────────────────────────────────────────────────────────

@dataclass
class NumberNode:
    value: str
    line:  int

@dataclass
class NameNode:
    name: str
    line: int

@dataclass
class BinaryOpNode:
    op:    str
    left:  object
    right: object

@dataclass
class CallNode:
    callee: str
    args:   List[object]

@dataclass
class AssignNode:
    target: NameNode
    value:  object

@dataclass
class ExprStmtNode:
    expr: object

@dataclass
class ParamNode:
    name:       str
    annotation: Optional[str] = None   # e.g. "tl.constexpr"

@dataclass
class KernelNode:
    name:   str
    params: List[ParamNode]
    body:   List[object]

@dataclass
class ProgramNode:
    kernel: KernelNode


# ── Error de parsing ───────────────────────────────────────────────────────────

class ParseError(Exception):
    def __init__(self, message: str, line: int, col: int):
        super().__init__(message)
        self.msg  = message
        self.line = line
        self.col  = col

    def __str__(self) -> str:
        return f"{self.msg} en línea {self.line}, col {self.col}"


# ── Parser ─────────────────────────────────────────────────────────────────────

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens    = tokens
        self.i         = 0
        self.lookahead = tokens[0]

    # ── Primitivas ────────────────────────────────────────────────────────────

    def advance(self) -> Token:
        tok = self.tokens[self.i]
        self.i += 1
        self.lookahead = self.tokens[self.i] if self.i < len(self.tokens) else self.tokens[-1]
        return tok

    def peek_type(self, k: int = 1) -> str:
        idx = self.i + k
        return self.tokens[idx].type if idx < len(self.tokens) else EOF

    _DISPLAY = {
        "AT": "@", "COLON": ":", "COMMA": ",", "DOT": ".",
        "EQUALS": "=", "LBRACE": "{", "LPAREN": "(",
        "MINUS": "-", "PLUS": "+", "RBRACE": "}", "RPAREN": ")",
        "SEMICOLON": ";", "SLASH": "/", "STAR": "*",
        "ID": "identificador", "NUMBER": "número", "EOF": "fin de archivo",
    }

    def match(self, expected_type: str) -> Token:
        if self.lookahead.type == expected_type:
            return self.advance()
        want = self._DISPLAY.get(expected_type, expected_type)
        got  = self.lookahead.value or self._DISPLAY.get(self.lookahead.type, self.lookahead.type)
        raise ParseError(
            f"se esperaba '{want}' pero llegó '{got}'",
            self.lookahead.line,
            self.lookahead.col,
        )

    def match_value(self, expected_type: str, expected_value: str) -> Token:
        if self.lookahead.type == expected_type and self.lookahead.value == expected_value:
            return self.advance()
        raise ParseError(
            f"se esperaba '{expected_value}' pero llegó '{self.lookahead.value or self.lookahead.type}'",
            self.lookahead.line,
            self.lookahead.col,
        )

    # ── Reglas de la gramática ────────────────────────────────────────────────

    def parse_program(self) -> ProgramNode:
        decorator = self.parse_decorator()
        kernel    = self.parse_kernel(decorator)
        self.match(EOF)
        return ProgramNode(kernel=kernel)

    def parse_decorator(self) -> str:
        """@triton.jit — valida el decorador y devuelve 'triton.jit'"""
        tok = self.lookahead
        if tok.type != AT:
            raise ParseError(
                f"se esperaba '@triton.jit' pero llegó '{tok.value or tok.type}'",
                tok.line, tok.col,
            )
        self.match(AT)
        ns   = self.match(ID).value
        self.match(DOT)
        fn   = self.match(ID).value
        full = f"{ns}.{fn}"
        if full != "triton.jit":
            raise ParseError(
                f"decorador inválido '@{full}', se esperaba '@triton.jit'",
                tok.line, tok.col,
            )
        return full

    def parse_kernel(self, _decorator: str) -> KernelNode:
        self.match_value(ID, "def")
        name_tok = self.match(ID)
        self.match(LPAREN)
        params = self.parse_params()
        self.match(RPAREN)
        self.match(COLON)
        self.match(LBRACE)
        body = self.parse_stmt_list()
        self.match(RBRACE)
        return KernelNode(name=name_tok.value, params=params, body=body)

    def parse_params(self) -> List[ParamNode]:
        params: List[ParamNode] = []
        if self.lookahead.type == RPAREN:
            return params
        params.append(self.parse_param())
        while self.lookahead.type == COMMA:
            self.advance()
            params.append(self.parse_param())
        return params

    def parse_param(self) -> ParamNode:
        name_tok   = self.match(ID)
        annotation = None
        if self.lookahead.type == COLON:
            self.advance()
            ns   = self.match(ID).value
            self.match(DOT)
            attr = self.match(ID).value
            annotation = f"{ns}.{attr}"
        return ParamNode(name=name_tok.value, annotation=annotation)

    def parse_stmt_list(self) -> List[object]:
        stmts = []
        while self.lookahead.type not in (RBRACE, EOF):
            stmts.append(self.parse_stmt())
        return stmts

    def parse_stmt(self) -> object:
        tok = self.lookahead
        if tok.type == ID and tok.value in ("if", "while", "for", "return", "else", "elif"):
            raise ParseError(
                f"'{tok.value}' no está soportado en Mini-Triton",
                tok.line, tok.col,
            )
        # Lookahead 2: ID '=' → asignación
        if tok.type == ID and self.peek_type(1) == EQUALS:
            return self.parse_assign()
        return self.parse_expr_stmt()

    def parse_assign(self) -> AssignNode:
        name_tok = self.match(ID)
        self.match(EQUALS)
        value = self.parse_expr()
        self.match(SEMICOLON)
        return AssignNode(
            target=NameNode(name=name_tok.value, line=name_tok.line),
            value=value,
        )

    def parse_expr_stmt(self) -> ExprStmtNode:
        expr = self.parse_expr()
        self.match(SEMICOLON)
        return ExprStmtNode(expr=expr)

    # ── Expresiones (jerarquía de precedencia) ────────────────────────────────

    def parse_expr(self) -> object:
        """expr ::= term (('+' | '-') term)*"""
        node = self.parse_term()
        while self.lookahead.type in (PLUS, MINUS):
            op    = self.advance().value
            right = self.parse_term()
            node  = BinaryOpNode(op=op, left=node, right=right)
        return node

    def parse_term(self) -> object:
        """term ::= factor (('*' | '/') factor)*"""
        node = self.parse_factor()
        while self.lookahead.type in (STAR, SLASH):
            op    = self.advance().value
            right = self.parse_factor()
            node  = BinaryOpNode(op=op, left=node, right=right)
        return node

    def parse_factor(self) -> object:
        """factor ::= NUMBER | '(' expr ')' | name_or_call"""
        tok = self.lookahead
        if tok.type == NUMBER:
            self.advance()
            return NumberNode(value=tok.value, line=tok.line)
        if tok.type == LPAREN:
            self.advance()
            node = self.parse_expr()
            self.match(RPAREN)
            return node
        if tok.type == ID:
            return self.parse_name_or_call()
        raise ParseError(
            f"se esperaba un operando pero llegó '{tok.value or tok.type}'",
            tok.line, tok.col,
        )

    def parse_name_or_call(self) -> object:
        """name_or_call ::= ID ('.' ID)? ('(' args ')')?"""
        tok  = self.match(ID)
        name = tok.value

        if self.lookahead.type == DOT:
            self.advance()
            attr = self.match(ID).value
            name = f"{name}.{attr}"

        if self.lookahead.type == LPAREN:
            self.advance()
            args = self.parse_args()
            self.match(RPAREN)
            return CallNode(callee=name, args=args)

        return NameNode(name=name, line=tok.line)

    def parse_args(self) -> List[object]:
        args: List[object] = []
        if self.lookahead.type == RPAREN:
            return args
        self._reject_keyword_arg()
        args.append(self.parse_expr())
        while self.lookahead.type == COMMA:
            self.advance()
            self._reject_keyword_arg()
            args.append(self.parse_expr())
        return args

    def _reject_keyword_arg(self) -> None:
        """Detecta ID '=' dentro de lista de args y lanza ParseError."""
        if self.lookahead.type == ID and self.peek_type(1) == EQUALS:
            tok = self.lookahead
            raise ParseError(
                f"argumento con nombre '{tok.value}=...' no permitido",
                tok.line, tok.col,
            )


# ── Serialización del AST ──────────────────────────────────────────────────────

def ast_to_str(program: ProgramNode) -> str:
    lines: List[str] = ["Program"]
    kernel     = program.kernel
    param_strs = [
        f"{p.name}: {p.annotation}" if p.annotation else p.name
        for p in kernel.params
    ]
    lines.append(f'└── Kernel(name="{kernel.name}", params=[{", ".join(param_strs)}])')
    child_pfx = "    "
    for i, stmt in enumerate(kernel.body):
        is_last = (i == len(kernel.body) - 1)
        lines.append(_node_str(stmt, child_pfx, is_last))
    return "\n".join(lines)


def _node_str(node: object, prefix: str, is_last: bool) -> str:
    branch    = "└── " if is_last else "├── "
    child_pfx = prefix + ("    " if is_last else "│   ")

    if isinstance(node, AssignNode):
        lines = [prefix + branch + "Assign"]
        lines.append(_node_str(node.target, child_pfx, False))
        lines.append(_node_str(node.value,  child_pfx, True))
        return "\n".join(lines)

    if isinstance(node, ExprStmtNode):
        lines = [prefix + branch + "ExprStmt"]
        lines.append(_node_str(node.expr, child_pfx, True))
        return "\n".join(lines)

    if isinstance(node, BinaryOpNode):
        lines = [prefix + branch + f'BinaryOp("{node.op}")']
        lines.append(_node_str(node.left,  child_pfx, False))
        lines.append(_node_str(node.right, child_pfx, True))
        return "\n".join(lines)

    if isinstance(node, CallNode):
        lines = [prefix + branch + f'Call("{node.callee}")']
        for i, arg in enumerate(node.args):
            lines.append(_node_str(arg, child_pfx, i == len(node.args) - 1))
        return "\n".join(lines)

    if isinstance(node, NameNode):
        return prefix + branch + f'Name("{node.name}")'

    if isinstance(node, NumberNode):
        return prefix + branch + f'Number({node.value})'

    return prefix + branch + repr(node)


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
        print(f"INVALIDO: error léxico — {e}")
        sys.exit(1)

    parser = Parser(tokens)
    try:
        program = parser.parse_program()
    except ParseError as e:
        print(f"INVALIDO: {e}")
        sys.exit(1)

    print("VALIDO")
    print(ast_to_str(program))


if __name__ == "__main__":
    _main()
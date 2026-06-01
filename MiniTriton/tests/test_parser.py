"""
Test suite para el parser Mini-Triton.
Ejecutar: pytest tests/test_parser.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lexer import tokenize
from mini_triton_parser import (
    Parser, ParseError,
    ProgramNode, KernelNode, ParamNode,
    AssignNode, ExprStmtNode,
    BinaryOpNode, CallNode, NameNode, NumberNode,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_file(filename: str) -> ProgramNode:
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, encoding="utf-8") as f:
        source = f.read()
    return Parser(tokenize(source)).parse_program()


def parse_src(source: str) -> ProgramNode:
    return Parser(tokenize(source)).parse_program()


# ── Casos válidos ─────────────────────────────────────────────────────────────

class TestValidCases:

    def test_valid_01_simple_assign(self):
        ast = parse_file("valid_01.tri")
        assert isinstance(ast, ProgramNode)
        assert ast.kernel.name == "add"
        assert [p.name for p in ast.kernel.params] == ["x", "y"]
        stmt = ast.kernel.body[0]
        assert isinstance(stmt, AssignNode)
        assert stmt.target.name == "z"
        assert isinstance(stmt.value, BinaryOpNode)
        assert stmt.value.op == "+"

    def test_valid_02_grouped_precedence(self):
        ast = parse_file("valid_02.tri")
        assert ast.kernel.name == "one"
        stmt = ast.kernel.body[0]
        assert isinstance(stmt, AssignNode)
        # (x+1)*2 → BinaryOp("*", BinaryOp("+", ...), Number(2))
        top = stmt.value
        assert isinstance(top, BinaryOpNode)
        assert top.op == "*"
        assert isinstance(top.left, BinaryOpNode)
        assert top.left.op == "+"

    def test_valid_03_operator_precedence(self):
        ast = parse_file("valid_03.tri")
        assert [p.name for p in ast.kernel.params] == ["a", "b", "c"]
        stmt = ast.kernel.body[0]
        assert isinstance(stmt, AssignNode)
        # a + b*c → BinaryOp("+", Name("a"), BinaryOp("*", ...))
        top = stmt.value
        assert isinstance(top, BinaryOpNode)
        assert top.op == "+"
        assert isinstance(top.right, BinaryOpNode)
        assert top.right.op == "*"

    def test_valid_04_expr_stmt_dotted_call(self):
        ast = parse_file("valid_04.tri")
        stmt = ast.kernel.body[0]
        assert isinstance(stmt, ExprStmtNode)
        call = stmt.expr
        assert isinstance(call, CallNode)
        assert call.callee == "tl.load"
        assert len(call.args) == 1
        assert isinstance(call.args[0], NameNode)

    def test_valid_05_call_mixed_args(self):
        ast = parse_file("valid_05.tri")
        stmt = ast.kernel.body[0]
        assert isinstance(stmt, AssignNode)
        call = stmt.value
        assert isinstance(call, CallNode)
        assert call.callee == "foo"
        assert len(call.args) == 3
        assert isinstance(call.args[0], NameNode)
        assert isinstance(call.args[1], NumberNode)
        assert isinstance(call.args[2], BinaryOpNode)

    def test_valid_06_annotated_param(self):
        ast = parse_file("valid_06.tri")
        params = ast.kernel.params
        assert params[0].name == "x"
        assert params[0].annotation is None
        assert params[1].name == "BS"
        assert params[1].annotation == "tl.constexpr"
        stmt = ast.kernel.body[0]
        assert isinstance(stmt, AssignNode)
        assert isinstance(stmt.value, CallNode)
        assert stmt.value.callee == "tl.arange"

    def test_valid_07_full_kernel_b(self):
        ast = parse_file("valid_07.tri")
        assert ast.kernel.name == "kernel_b"
        assert len(ast.kernel.params) == 4
        assert ast.kernel.params[3].annotation == "tl.constexpr"
        assert len(ast.kernel.body) == 3
        # Stmt 0: pid = tl.program_id(0)
        s0 = ast.kernel.body[0]
        assert isinstance(s0, AssignNode)
        assert s0.target.name == "pid"
        assert isinstance(s0.value, CallNode)
        assert s0.value.callee == "tl.program_id"
        # Stmt 2: ExprStmt with tl.store containing tl.load as 2nd arg
        s2 = ast.kernel.body[2]
        assert isinstance(s2, ExprStmtNode)
        store = s2.expr
        assert isinstance(store, CallNode)
        assert store.callee == "tl.store"
        load = store.args[1]
        assert isinstance(load, CallNode)
        assert load.callee == "tl.load"


# ── Casos inválidos ───────────────────────────────────────────────────────────

class TestInvalidCases:

    def _raises(self, filename: str, fragment: str) -> None:
        with pytest.raises(ParseError) as exc_info:
            parse_file(filename)
        assert fragment in str(exc_info.value), (
            f"Esperaba '{fragment}' en el error, pero llegó: {exc_info.value}"
        )

    def test_invalid_01_missing_decorator(self):
        self._raises("invalid_01.tri", "@triton.jit")

    def test_invalid_02_missing_braces(self):
        self._raises("invalid_02.tri", "'{'")

    def test_invalid_03_missing_semicolon(self):
        self._raises("invalid_03.tri", "';'")

    def test_invalid_04_incomplete_expr(self):
        self._raises("invalid_04.tri", "operando")

    def test_invalid_05_keyword_arg(self):
        self._raises("invalid_05.tri", "mask=")

    def test_invalid_06_if_statement(self):
        self._raises("invalid_06.tri", "'if'")


# ── Validación de anotaciones de parámetro ────────────────────────────────────

class TestAnnotationValidation:

    def test_tl_constexpr_accepted(self):
        ast = parse_src("@triton.jit def f(x, BS: tl.constexpr): { y = x; }")
        assert ast.kernel.params[1].annotation == "tl.constexpr"

    def test_arbitrary_annotation_rejected(self):
        with pytest.raises(ParseError, match="tl.constexpr"):
            parse_src("@triton.jit def f(x, BS: foo.bar): { y = x; }")

    def test_wrong_attr_rejected(self):
        with pytest.raises(ParseError, match="tl.constexpr"):
            parse_src("@triton.jit def f(x, BS: tl.something): { y = x; }")

    def test_wrong_namespace_rejected(self):
        with pytest.raises(ParseError, match="tl.constexpr"):
            parse_src("@triton.jit def f(x, BS: torch.Tensor): { y = x; }")

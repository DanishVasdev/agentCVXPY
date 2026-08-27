"""Sandboxed evaluation of LLM-authored CVXPY expression strings.

The LLM never gets to run arbitrary code: it only supplies expression text
(e.g. "3*x[0] + c @ y <= budget") referencing names we already declared as
real cvxpy Variable/Parameter objects. We AST-whitelist the expression
(rejecting anything but names/attributes/calls/operators/literals, and any
dunder access) before eval()'ing it in a namespace with no builtins.
"""

from __future__ import annotations

import ast

import cvxpy as cp
import numpy as np

_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.MatMult,
    ast.Div,
    ast.Pow,
    ast.UAdd,
    ast.USub,
    ast.Compare,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Eq,
    ast.NotEq,
    ast.Call,
    ast.keyword,
    ast.Name,
    ast.Load,
    ast.Attribute,
    ast.Subscript,
    ast.Slice,
    ast.Index,  # py<3.9 compat no-op on newer ast, harmless if unused
    ast.Tuple,
    ast.List,
    ast.Constant,
)


class UnsafeExpressionError(ValueError):
    pass


def _check_node(node: ast.AST, allowed_names: set[str]) -> None:
    for child in ast.walk(node):
        if not isinstance(child, _ALLOWED_NODES):
            raise UnsafeExpressionError(
                f"disallowed syntax in expression: {type(child).__name__}"
            )
        if isinstance(child, ast.Name) and child.id.startswith("_"):
            raise UnsafeExpressionError(f"disallowed identifier: {child.id}")
        if isinstance(child, ast.Attribute) and child.attr.startswith("_"):
            raise UnsafeExpressionError(f"disallowed attribute access: {child.attr}")
        if isinstance(child, ast.Name) and child.id not in allowed_names:
            raise UnsafeExpressionError(
                f"unknown name '{child.id}' -- must be a declared variable/parameter, "
                f"'cp', or 'np'"
            )


def safe_eval(expr: str, namespace: dict):
    """Evaluate expr against namespace (declared vars/params + cp/np), rejecting
    anything outside a safe expression grammar."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise UnsafeExpressionError(f"invalid syntax: {e}") from e

    allowed_names = set(namespace.keys())
    _check_node(tree, allowed_names)

    code = compile(tree, filename="<nl2cvxpygen-expr>", mode="eval")
    return eval(code, {"__builtins__": {}}, namespace)  # noqa: S307 - sandboxed above


def base_namespace() -> dict:
    return {"cp": cp, "np": np}

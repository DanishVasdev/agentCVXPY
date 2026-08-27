import cvxpy as cp
import pytest

from nl2cvxpygen.safe_eval import UnsafeExpressionError, base_namespace, safe_eval


def _namespace():
    ns = base_namespace()
    ns["x"] = cp.Variable(name="x")
    ns["y"] = cp.Variable(name="y")
    return ns


def test_valid_expression_builds_constraint():
    result = safe_eval("2 * x + 3 * y <= 10", _namespace())
    assert isinstance(result, cp.constraints.constraint.Constraint)


def test_valid_expression_builds_objective_term():
    result = safe_eval("cp.sum_squares(x - y)", _namespace())
    assert isinstance(result, cp.Expression)


def test_rejects_dunder_attribute_access():
    with pytest.raises(UnsafeExpressionError):
        safe_eval("x.__class__", _namespace())


def test_rejects_unknown_name():
    with pytest.raises(UnsafeExpressionError):
        safe_eval("os.system('echo hi')", _namespace())


def test_rejects_statements():
    with pytest.raises(UnsafeExpressionError):
        safe_eval("__import__('os').system('echo hi')", _namespace())


def test_rejects_assignment_syntax():
    with pytest.raises(UnsafeExpressionError):
        safe_eval("x = 5", _namespace())

import pytest

from nl2cvxpygen.builder import FormulationError, build_and_solve
from nl2cvxpygen.schema import OptimizationSpec, ParameterSpec, VariableSpec


def test_diet_problem_lp_matches_known_optimum():
    spec = OptimizationSpec(
        name="diet_problem",
        sense="minimize",
        variables=[
            VariableSpec(name="b", shape=[], nonneg=True, description="units of bread"),
            VariableSpec(name="m", shape=[], nonneg=True, description="units of milk"),
        ],
        parameters=[
            ParameterSpec(name="cost_b", shape=[], value=2.0),
            ParameterSpec(name="cost_m", shape=[], value=3.0),
        ],
        objective="cost_b * b + cost_m * m",
        constraints=["4 * b + 3 * m >= 24", "2 * b + 5 * m >= 20"],
    )
    built = build_and_solve(spec)
    assert built.reference_value == pytest.approx(108 / 7, abs=1e-4)
    assert built.variables["b"].value == pytest.approx(30 / 7, abs=1e-4)
    assert built.variables["m"].value == pytest.approx(16 / 7, abs=1e-4)


def test_production_planning_lp_matches_known_optimum():
    spec = OptimizationSpec(
        name="production_planning",
        sense="maximize",
        variables=[
            VariableSpec(name="c", shape=[], nonneg=True, description="chairs produced"),
            VariableSpec(name="t", shape=[], nonneg=True, description="tables produced"),
        ],
        parameters=[
            ParameterSpec(name="profit_c", shape=[], value=15.0),
            ParameterSpec(name="profit_t", shape=[], value=40.0),
        ],
        objective="profit_c * c + profit_t * t",
        constraints=["3 * c + 5 * t <= 120", "c + 4 * t <= 60"],
    )
    built = build_and_solve(spec)
    assert built.reference_value == pytest.approx(5100 / 7, abs=1e-4)
    assert built.variables["c"].value == pytest.approx(180 / 7, abs=1e-4)
    assert built.variables["t"].value == pytest.approx(60 / 7, abs=1e-4)


def test_portfolio_qp_solves_with_matrix_parameter():
    spec = OptimizationSpec(
        name="portfolio_qp",
        sense="minimize",
        variables=[VariableSpec(name="w", shape=[3], nonneg=True, description="portfolio weights")],
        parameters=[
            ParameterSpec(name="mu", shape=[3], value=[0.10, 0.15, 0.08]),
            ParameterSpec(
                name="Sigma",
                shape=[3, 3],
                value=[[0.05, 0.01, 0.00], [0.01, 0.08, 0.02], [0.00, 0.02, 0.03]],
                psd=True,
            ),
        ],
        objective="cp.quad_form(w, Sigma)",
        constraints=["mu @ w >= 0.10", "cp.sum(w) == 1"],
    )
    built = build_and_solve(spec)
    assert built.problem.status == "optimal"
    assert built.variables["w"].value.sum() == pytest.approx(1.0, abs=1e-4)
    assert (built.variables["w"].value >= -1e-6).all()


def test_out_of_scope_problem_raises():
    spec = OptimizationSpec(
        name="knapsack",
        supported=False,
        notes="requires binary variables, out of LP/QP scope",
    )
    with pytest.raises(FormulationError):
        build_and_solve(spec)


def test_unsafe_constraint_is_rejected():
    spec = OptimizationSpec(
        name="malicious",
        variables=[VariableSpec(name="x", shape=[], nonneg=True)],
        objective="x",
        constraints=["x.__class__ <= 1"],
    )
    with pytest.raises(FormulationError):
        build_and_solve(spec)

"""Turn a validated OptimizationSpec into a real, solved cvxpy.Problem."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from nl2cvxpygen.safe_eval import base_namespace, safe_eval
from nl2cvxpygen.schema import OptimizationSpec


class FormulationError(Exception):
    """Raised when the LLM's spec doesn't produce a valid, solvable problem.

    The message is written to be fed straight back to the LLM for a retry.
    """


@dataclass
class BuiltProblem:
    spec: OptimizationSpec
    problem: cp.Problem
    variables: dict[str, cp.Variable]
    parameters: dict[str, cp.Parameter]
    reference_value: float
    reference_solve_time: float


def _shape_tuple(shape: list[int]) -> tuple[int, ...]:
    return tuple(shape)


def build_namespace(spec: OptimizationSpec) -> tuple[dict, dict[str, cp.Variable], dict[str, cp.Parameter]]:
    namespace = base_namespace()
    variables: dict[str, cp.Variable] = {}
    parameters: dict[str, cp.Parameter] = {}

    seen = set()
    for v in spec.variables:
        if v.name in seen:
            raise FormulationError(f"duplicate declared name: {v.name}")
        seen.add(v.name)
        var = cp.Variable(_shape_tuple(v.shape), name=v.name, nonneg=v.nonneg)
        variables[v.name] = var
        namespace[v.name] = var

    for p in spec.parameters:
        if p.name in seen:
            raise FormulationError(f"duplicate declared name: {p.name}")
        seen.add(p.name)
        param = cp.Parameter(_shape_tuple(p.shape), name=p.name, PSD=p.psd)
        try:
            param.value = np.array(p.value, dtype=float).reshape(_shape_tuple(p.shape) or ())
        except (ValueError, TypeError) as e:
            raise FormulationError(
                f"parameter '{p.name}' value does not match declared shape {p.shape}: {e}"
            ) from e
        parameters[p.name] = param
        namespace[p.name] = param

    return namespace, variables, parameters


def build_problem(spec: OptimizationSpec) -> tuple[cp.Problem, dict[str, cp.Variable], dict[str, cp.Parameter]]:
    if not spec.supported:
        raise FormulationError(
            f"problem marked out of scope for LP/QP: {spec.notes or 'no reason given'}"
        )
    if not spec.variables:
        raise FormulationError("spec declares no variables")
    if not spec.objective.strip():
        raise FormulationError("spec has an empty objective")

    namespace, variables, parameters = build_namespace(spec)

    try:
        objective_expr = safe_eval(spec.objective, namespace)
    except Exception as e:
        raise FormulationError(f"objective '{spec.objective}' failed to evaluate: {e}") from e

    if not isinstance(objective_expr, cp.Expression):
        raise FormulationError(
            f"objective '{spec.objective}' did not evaluate to a CVXPY expression"
        )

    constraints = []
    for c in spec.constraints:
        try:
            c_expr = safe_eval(c, namespace)
        except Exception as e:
            raise FormulationError(f"constraint '{c}' failed to evaluate: {e}") from e
        if not isinstance(c_expr, cp.constraints.constraint.Constraint):
            raise FormulationError(f"constraint '{c}' did not evaluate to a CVXPY constraint")
        constraints.append(c_expr)

    objective = cp.Minimize(objective_expr) if spec.sense == "minimize" else cp.Maximize(objective_expr)
    problem = cp.Problem(objective, constraints)

    if not problem.is_dcp():
        raise FormulationError(
            "problem is not DCP-compliant (not a valid convex program) -- "
            f"objective='{spec.objective}', constraints={spec.constraints}"
        )

    return problem, variables, parameters


def build_and_solve(spec: OptimizationSpec) -> BuiltProblem:
    problem, variables, parameters = build_problem(spec)

    try:
        problem.solve()
    except cp.error.SolverError as e:
        raise FormulationError(f"solver failed on reference solve: {e}") from e

    if problem.status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE):
        raise FormulationError(
            "problem is infeasible as formulated -- check constraint bounds/signs "
            f"(constraints={spec.constraints})"
        )
    if problem.status in (cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE):
        raise FormulationError(
            "problem is unbounded as formulated -- objective can be improved without limit, "
            "likely a missing constraint"
        )
    if problem.status != cp.OPTIMAL:
        raise FormulationError(f"reference solve did not reach optimality: status={problem.status}")

    return BuiltProblem(
        spec=spec,
        problem=problem,
        variables=variables,
        parameters=parameters,
        reference_value=problem.value,
        reference_solve_time=problem.solver_stats.solve_time if problem.solver_stats else 0.0,
    )

"""Structured spec the LLM fills in. This is the only thing the LLM produces --
everything downstream (variable/parameter construction, expression evaluation,
problem solving, code generation) is deterministic Python driven off this schema."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Shape = list[int]  # [] = scalar, [n] = vector, [m, n] = matrix


class VariableSpec(BaseModel):
    name: str = Field(description="Python identifier, unique among variables/parameters")
    shape: Shape = Field(default_factory=list, description="[] scalar, [n] vector, [m, n] matrix")
    nonneg: bool = Field(default=False, description="True if this variable must be >= 0")
    description: str = Field(default="", description="What this variable represents")


class ParameterSpec(BaseModel):
    name: str = Field(description="Python identifier, unique among variables/parameters")
    shape: Shape = Field(default_factory=list, description="[] scalar, [n] vector, [m, n] matrix")
    value: float | list[float] | list[list[float]] = Field(
        description="Concrete numeric data extracted from the problem text, matching shape"
    )
    psd: bool = Field(
        default=False,
        description="True if this is a square positive-semidefinite matrix used as the P "
        "argument to cp.quad_form (e.g. a covariance matrix) -- required for CVXPY to accept "
        "it there",
    )
    description: str = Field(default="", description="What this parameter represents")


class OptimizationSpec(BaseModel):
    name: str = Field(description="Short snake_case slug identifying the problem, e.g. diet_problem")
    supported: bool = Field(
        default=True,
        description="False if the problem requires integer/binary variables or non-quadratic "
        "nonlinear terms and therefore cannot be handled by this LP/QP-only tool",
    )
    sense: Literal["minimize", "maximize"] = "minimize"
    variables: list[VariableSpec] = Field(default_factory=list)
    parameters: list[ParameterSpec] = Field(default_factory=list)
    objective: str = Field(
        default="", description="Valid CVXPY/Python expression using only declared names, cp, np"
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Each a valid CVXPY/Python boolean expression (e.g. 'A @ x <= b') "
        "using only declared names, cp, np",
    )
    notes: str = Field(
        default="",
        description="Assumptions made resolving ambiguity in the NL text, or -- if "
        "supported is False -- an explanation of why the problem is out of scope",
    )

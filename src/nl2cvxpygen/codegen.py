"""Drive cvxpygen to generate (and, best-effort, compile+verify) embedded solver
code for a built-and-solved cvxpy.Problem."""

from __future__ import annotations

import keyword
import os
import re
from dataclasses import dataclass
from pathlib import Path

from cvxpygen import cpg

from nl2cvxpygen.builder import BuiltProblem

_SLUG_RE = re.compile(r"[^a-z0-9_]+")


def slugify(name: str) -> str:
    """Produce a valid Python identifier -- cvxpygen imports code_dir as a package."""
    slug = _SLUG_RE.sub("_", name.strip().lower()).strip("_") or "problem"
    if slug[0].isdigit():
        slug = f"p_{slug}"
    if keyword.iskeyword(slug):
        slug = f"{slug}_"
    return slug


@dataclass
class CodegenResult:
    code_dir: Path
    compiled: bool
    compile_error: str | None
    cpg_value: float | None
    matches_reference: bool | None


def generate(built: BuiltProblem, out_dir: Path, solver: str | None, wrapper: bool) -> CodegenResult:
    out_dir = out_dir.resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    # cvxpygen's wrapper=True path imports its own output via
    # importlib.import_module(f'{code_dir}.cpg_solver') with the *current* cwd on
    # sys.path (see cvxpygen/compiler.py::register). That only resolves for a flat,
    # single-segment code_dir -- a nested path like "generated/foo" is not a valid
    # dotted module name. So we chdir into out_dir's parent and pass just the slug.
    code_dir_arg = out_dir.name

    compiled = False
    compile_error = None
    cpg_value = None
    matches_reference = None

    original_cwd = Path.cwd()
    try:
        os.chdir(out_dir.parent)
        try:
            cpg.generate_code(built.problem, code_dir=code_dir_arg, solver=solver, wrapper=wrapper)
        except Exception as e:
            if not wrapper:
                raise
            # Retry once without compiling, so the user still gets inspectable C source
            # even if this machine's compiler toolchain isn't set up correctly.
            compile_error = str(e)
            cpg.generate_code(built.problem, code_dir=code_dir_arg, solver=solver, wrapper=False)
        else:
            if wrapper:
                compiled = True
                try:
                    built.problem.solve(method="CPG")
                    cpg_value = built.problem.value
                    # Embedded solvers (e.g. OSQP) are first-order/ADMM methods solved to
                    # their own default KKT tolerance (OSQP: eps_abs = eps_rel = 1e-3),
                    # not machine precision -- and on problems with several chained
                    # equality constraints (e.g. multi-step dynamics), small per-step KKT
                    # residuals can compound into an objective gap somewhat larger than
                    # that 1e-3 itself. 1% is loose enough to absorb that and still catch
                    # a genuinely wrong formulation, which is typically off by much more.
                    matches_reference = abs(cpg_value - built.reference_value) <= max(
                        1e-2, 1e-2 * abs(built.reference_value)
                    )
                except Exception as e:  # registered solve failing shouldn't crash the CLI
                    compile_error = f"CPG solve failed: {e}"
                    matches_reference = False
    finally:
        os.chdir(original_cwd)

    _write_summary(built, out_dir, compiled, compile_error, cpg_value, matches_reference)

    return CodegenResult(
        code_dir=out_dir,
        compiled=compiled,
        compile_error=compile_error,
        cpg_value=cpg_value,
        matches_reference=matches_reference,
    )


def _write_summary(
    built: BuiltProblem,
    out_dir: Path,
    compiled: bool,
    compile_error: str | None,
    cpg_value: float | None,
    matches_reference: bool | None,
) -> None:
    spec = built.spec
    lines = [f"# {spec.name}", ""]
    if spec.notes:
        lines += ["## Assumptions", spec.notes, ""]

    lines += ["## Variables"]
    for v in spec.variables:
        lines.append(f"- `{v.name}` shape={v.shape or 'scalar'}: {v.description}")
    lines.append("")

    lines += ["## Parameters (problem data)"]
    for p in spec.parameters:
        lines.append(f"- `{p.name}` shape={p.shape or 'scalar'}: {p.description}")
    lines.append("")

    lines += [
        "## Formulation",
        f"{spec.sense} {spec.objective}",
        "subject to:",
    ]
    lines += [f"  {c}" for c in spec.constraints]
    lines.append("")

    lines += [
        "## Solve results",
        f"- Reference (cvxpy) optimal value: {built.reference_value}",
    ]
    if compiled:
        lines.append(f"- CVXPYgen (`method='CPG'`) optimal value: {cpg_value}")
        lines.append(f"- Matches reference within tolerance: {matches_reference}")
    else:
        lines.append("- Not compiled (C source only). Compile error, if any:")
        lines.append(f"  {compile_error}")
    lines.append("")

    lines += [
        "## Reusing the generated code",
        "```python",
        "import cvxpy as cp",
        f"from {out_dir.name} import cpg_solver",
        "",
        "# rebuild the same variables/parameters, set new .value's on the parameters,",
        "# then register + solve:",
        "problem.register_solve('CPG', cpg_solver.cpg_solve)",
        "problem.solve(method='CPG')",
        "```",
    ]

    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

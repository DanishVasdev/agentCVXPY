from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from nl2cvxpygen.builder import BuiltProblem, FormulationError, build_and_solve
from nl2cvxpygen.codegen import generate, slugify
from nl2cvxpygen.llm import GeminiClient
from nl2cvxpygen.schema import OptimizationSpec

app = typer.Typer(add_completion=False)
console = Console()


def _formulate_with_retries(
    client: GeminiClient, problem_text: str, retries: int
) -> tuple[OptimizationSpec, BuiltProblem]:
    spec = client.formulate(problem_text)
    last_error: str | None = None

    for attempt in range(retries + 1):
        if not spec.supported:
            raise FormulationError(spec.notes or "problem marked out of scope")
        try:
            built = build_and_solve(spec)
            return spec, built
        except FormulationError as e:
            last_error = str(e)
            console.print(f"[yellow]Attempt {attempt + 1} failed:[/yellow] {last_error}")
            if attempt == retries:
                raise
            spec = client.fix(spec, last_error)

    raise FormulationError(last_error or "formulation failed")


@app.command()
def solve(
    description: Optional[str] = typer.Argument(None, help="Problem description, in plain English"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Read the problem description from a file"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output directory for generated code"),
    solver: Optional[str] = typer.Option(None, "--solver", help="Force a specific solver (default: auto)"),
    model: Optional[str] = typer.Option(None, "--model", help="Gemini model to use"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Gemini API key (default: GEMINI_API_KEY env var)"),
    no_wrapper: bool = typer.Option(False, "--no-wrapper", help="Only emit C source, skip compiling a Python wrapper"),
    retries: int = typer.Option(2, "--retries", help="Max self-correction retries if formulation fails"),
    show_code: bool = typer.Option(False, "--show-code", help="Print the reconstructed CVXPY formulation"),
):
    """Formulate an NL optimization problem and generate cvxpygen code for it."""
    load_dotenv()

    if file:
        problem_text = file.read_text(encoding="utf-8")
    elif description:
        problem_text = description
    else:
        console.print("[red]Provide a problem description or --file.[/red]")
        raise typer.Exit(1)

    client = GeminiClient(api_key=api_key, model=model)

    console.print("[bold]Formulating problem with Gemini...[/bold]")
    try:
        spec, built = _formulate_with_retries(client, problem_text, retries)
    except FormulationError as e:
        console.print(f"[red]Could not formulate a valid LP/QP from this problem:[/red] {e}")
        raise typer.Exit(1)

    if show_code:
        console.print(_render_formulation(spec))

    out_dir = out or Path("generated") / slugify(spec.name)
    console.print(f"[bold]Generating cvxpygen code into {out_dir}...[/bold]")
    result = generate(built, out_dir, solver=solver, wrapper=not no_wrapper)

    _print_summary(spec, built, result)


def _render_formulation(spec: OptimizationSpec) -> str:
    lines = [f"{spec.sense} {spec.objective}", "subject to:"]
    lines += [f"    {c}" for c in spec.constraints]
    return "\n".join(lines)


def _print_summary(spec: OptimizationSpec, built: BuiltProblem, result) -> None:
    table = Table(title=f"{spec.name} -- solve summary")
    table.add_column("Variable")
    table.add_column("Value")
    for name, var in built.variables.items():
        table.add_row(name, str(var.value))
    console.print(table)

    console.print(f"Reference (cvxpy) optimal value: [bold]{built.reference_value}[/bold]")
    if result.compiled:
        status = "[green]matches reference[/green]" if result.matches_reference else "[red]DOES NOT MATCH[/red]"
        console.print(f"CVXPYgen (compiled) optimal value: [bold]{result.cpg_value}[/bold] ({status})")
    else:
        console.print("[yellow]Not compiled -- C source only.[/yellow]")
        if result.compile_error:
            console.print(f"[yellow]Compile error:[/yellow] {result.compile_error}")

    console.print(f"Output directory: [bold]{result.code_dir}[/bold]")
    if spec.notes:
        console.print(f"Notes: {spec.notes}")


if __name__ == "__main__":
    app()

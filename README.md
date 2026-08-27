# nl2cvxpygen

Turn a natural-language optimization problem into working [cvxpygen](https://github.com/cvxgrp/cvxpygen)
embedded-solver code.

```
"A factory makes chairs and tables to maximize profit..."
        │
        ▼  Gemini (free tier) extracts structure only
   OptimizationSpec (variables, parameters, objective, constraints)
        │
        ▼  deterministic, sandboxed build (no LLM code execution)
   cvxpy.Problem  ──solve()──▶  reference optimal value
        │
        ▼  cvxpygen
   generated C code + compiled Python wrapper, verified against
   the reference solve
```

Scope: **linear and quadratic convex programs (LP/QP) with continuous variables only.**
Problems needing integer/binary variables or general nonlinear terms are detected and
rejected rather than silently mis-formulated.

## Setup

1. Get a free Gemini API key at https://aistudio.google.com/apikey.
2. Copy `.env.example` to `.env` and fill in `GEMINI_API_KEY` (or export it in your shell).
3. Install:

   ```bash
   pip install -e .
   ```

   This reuses your existing `cvxpy`/`cvxpygen` install if present. Compiling the generated
   C code into a Python extension requires a C/C++ compiler (e.g. Visual Studio Build Tools
   on Windows, or `gcc`/`clang` elsewhere). If none is available, use `--no-wrapper` to still
   get the generated C source without compiling it.

## Usage

```bash
nl2cvxpygen solve "A factory makes chairs and tables to maximize profit..."
nl2cvxpygen solve --file examples/diet_problem.txt
nl2cvxpygen solve --file examples/portfolio_qp.txt --out generated/portfolio --show-code
```

Useful flags:

- `--out DIR` — output directory for the generated code (default `generated/<slug>`)
- `--solver NAME` — force a specific solver instead of cvxpygen's auto-choice (OSQP for QPs)
- `--no-wrapper` — skip compilation, only emit C source
- `--retries N` — how many times to ask Gemini to self-correct if the formulation fails to
  build/solve (default 2)
- `--show-code` — print the reconstructed CVXPY formulation

## What you get

Each run writes into the output directory:

- The full `cvxpygen` output (`c/`, `cpp/`, and, if compiled, an importable Python package
  with a registered `'CPG'` solve method).
- `SUMMARY.md` — the variables/parameters/formulation extracted from your problem text, any
  assumptions the model made, and the reference-vs-generated-code solve comparison.

## How correctness is checked

The LLM only ever produces a structured spec (variable/parameter declarations and CVXPY
expression strings) — never a program we execute directly. Expressions are evaluated through
an AST-whitelisted, builtin-free `eval()` against the real `cvxpy` objects we constructed, so
CVXPY's own DCP-aware parser builds the actual problem. Before generating any code, the tool:

1. checks the problem is DCP-compliant,
2. solves it with `cvxpy`'s reference solver and checks the status is optimal,
3. re-prompts the model with the concrete error if any of the above fails (up to `--retries`
   times), and
4. after `cvxpygen` compiles the embedded solver, re-solves with `method='CPG'` and confirms
   the result matches the reference solve within tolerance.

## Project layout

```
src/nl2cvxpygen/
  schema.py     Pydantic OptimizationSpec (the only thing the LLM produces)
  prompts.py    System prompt + few-shot rules for the LLM
  llm.py        Gemini client (structured JSON output, self-correction)
  safe_eval.py  Sandboxed expression evaluator
  builder.py    Spec -> cvxpy.Problem -> reference solve
  codegen.py    cvxpygen invocation, compile+verify, SUMMARY.md
  cli.py        Typer CLI
examples/       Sample NL problem descriptions (LP + QP)
tests/          Offline unit tests (no API key needed)
```

## Tests

```bash
pytest tests/
```

These run fully offline against canned specs — they don't call Gemini.

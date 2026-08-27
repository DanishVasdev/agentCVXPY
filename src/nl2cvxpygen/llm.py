"""Gemini client: NL problem text -> structured OptimizationSpec."""

from __future__ import annotations

import os

from google import genai
from google.genai import types

from nl2cvxpygen.prompts import SYSTEM_PROMPT, build_fix_prompt, build_user_prompt
from nl2cvxpygen.schema import OptimizationSpec

DEFAULT_MODEL = "gemini-3.6-flash"


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No Gemini API key found. Set GEMINI_API_KEY (get a free key at "
                "https://aistudio.google.com/apikey) or pass --api-key."
            )
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        self._client = genai.Client(api_key=api_key)

    def _generate(self, contents: str) -> OptimizationSpec:
        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=OptimizationSpec,
            ),
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, OptimizationSpec):
            return parsed
        return OptimizationSpec.model_validate_json(response.text)

    def formulate(self, problem_text: str) -> OptimizationSpec:
        return self._generate(build_user_prompt(problem_text))

    def fix(self, previous_spec: OptimizationSpec, error_message: str) -> OptimizationSpec:
        prompt = build_fix_prompt(previous_spec.model_dump_json(indent=2), error_message)
        return self._generate(prompt)
